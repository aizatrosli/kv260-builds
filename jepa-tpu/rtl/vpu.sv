// Vector Processing Unit (VPU) for the TPU-KV260 transformer extension.
//
// Works on one vector of `rows` x N int8 lanes (a token of D = rows*N channels, or a row of
// attention logits). Bit-exact with jepa/intops.py:
//   OP_ADD  out = sat8((a*ma + b*mb + 2^15) >>> 16)                          (residual add)
//   OP_LUT  out = gelu_lut[a]                                               (GELU, any int8->int8)
//   OP_LN   S=sum a, V=D*sum a^2 - S^2, sig=max(isqrt V,1), inv=2^24/sig,
//           out = sat8(((D*a - S)*32*inv + 2^23) >>> 24)                    (LayerNorm, no affine)
//   OP_SM   m=max a, e=exp_lut[m-a], Z=sum e, inv=(127<<22)/Z,
//           out = (e*inv + 2^21) >> 22                                       (softmax -> 0..127)
// Flow: command, then `rows` input rows (in_a, and in_b for ADD); `rows` output rows follow.
// Two stages with two vector buffers: the next command and its rows are accepted while the
// previous vector is still in sqrt/divide/emit (cmd_ready/in_ready reflect the front stage).
// LUTs are written through the lut_* port, or pre-loaded from hex files (hard-coded model).
`timescale 1ns/1ps

module vpu #(
  parameter int unsigned N    = 16,
  parameter int unsigned MAXR = 32,           // longest vector: MAXR*N lanes
  parameter string       EXP_INIT  = "",
  parameter string       GELU_INIT = ""
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,
  // command
  input  logic                 cmd_valid_i,
  output logic                 cmd_ready_o,
  input  logic [1:0]           cmd_op_i,
  input  logic [$clog2(MAXR+1)-1:0] cmd_rows_i,
  input  logic signed [17:0]   cmd_ma_i,
  input  logic signed [17:0]   cmd_mb_i,
  // LUT write port
  input  logic                 lut_we_i,
  input  logic                 lut_sel_i,     // 0: exp (16 bit), 1: gelu (8 bit)
  input  logic [7:0]           lut_addr_i,    // exp: index m-a; gelu: a + 128
  input  logic [15:0]          lut_wdata_i,
  // input rows
  input  logic                 in_valid_i,
  output logic                 in_ready_o,
  input  logic [N*8-1:0]       in_a_i,
  input  logic [N*8-1:0]       in_b_i,
  // output rows
  output logic                 out_valid_o,
  input  logic                 out_ready_i,
  output logic [N*8-1:0]       out_data_o,
  output logic                 busy_o
);
  localparam logic [1:0] OP_ADD = 2'd0, OP_LUT = 2'd1, OP_LN = 2'd2, OP_SM = 2'd3;
  localparam int unsigned RW = $clog2(MAXR+1);

  logic [15:0] exp_lut  [256];
  logic [7:0]  gelu_lut [256];
  initial begin
    if (EXP_INIT  != "") $readmemh(EXP_INIT,  exp_lut);
    if (GELU_INIT != "") $readmemh(GELU_INIT, gelu_lut);
  end
  always_ff @(posedge clk_i) if (lut_we_i) begin
    if (lut_sel_i) gelu_lut[lut_addr_i] <= lut_wdata_i[7:0];
    else           exp_lut[lut_addr_i]  <= lut_wdata_i;
  end

  function automatic logic signed [7:0] sat8(input logic signed [63:0] v);
    return (v > 127) ? 8'sd127 : (v < -128) ? -8'sd128 : v[7:0];
  endfunction

  logic [N*8-1:0]  buf8 [2][MAXR];           // two vector buffers (input, or finished ADD/LUT output)
  logic [N*16-1:0] buf16 [MAXR];             // softmax exps (back stage only)

  // ================= front stage: command + row load + statistics =================
  typedef enum logic [1:0] {F_IDLE, F_LOAD, F_HAND} fst_e;
  fst_e fs;
  logic               wsel;
  logic [1:0]         f_op;
  logic [RW-1:0]      f_rows, cnt;
  logic signed [17:0] ma, mb;
  logic signed [31:0] fS;
  logic [31:0]        fSq;
  logic signed [7:0]  fmx;

  logic [N*8-1:0]  add_row, lut_row;
  logic signed [31:0] row_sum;
  logic [31:0]        row_sq;
  logic signed [7:0]  row_max;
  always_comb begin
    row_sum = '0; row_sq = '0; row_max = -8'sd128;
    for (int i = 0; i < N; i++) begin
      logic signed [7:0] a, b;
      a = in_a_i[8*i +: 8]; b = in_b_i[8*i +: 8];
      add_row[8*i +: 8] = sat8((64'(a) * 64'(ma) + 64'(b) * 64'(mb) + 64'sd32768) >>> 16);
      lut_row[8*i +: 8] = gelu_lut[{~a[7], a[6:0]}];
      row_sum += 32'(a);
      row_sq  += 32'(int'(a) * int'(a));
      if (a > row_max) row_max = a;
    end
  end

  // ================= back stage: reductions, sqrt, divide, emit =================
  typedef enum logic [2:0] {B_IDLE, B_EXPSUM, B_SQRT, B_DIV, B_EMIT} bst_e;
  bst_e bs;
  logic               rsel;
  logic [1:0]         op;
  logic [RW-1:0]      rows, xr;
  logic signed [31:0] S;
  logic signed [7:0]  mx;
  logic [31:0]        Z;
  logic [63:0]        inv, rem, rad, root, bitv, num, den, quo;
  logic [6:0]         it;
  wire                hand = (fs == F_HAND) && (bs == B_IDLE);

  logic [N*16-1:0] exp_row;
  logic [31:0]     exp_sum;
  always_comb begin
    exp_sum = '0;
    for (int i = 0; i < N; i++) begin
      logic signed [8:0] d;
      d = 9'(mx) - 9'($signed(buf8[rsel][xr][8*i +: 8]));
      exp_row[16*i +: 16] = exp_lut[d[7:0]];
      exp_sum += 32'(exp_lut[d[7:0]]);
    end
  end

  logic [N*8-1:0] emit_row;
  logic [31:0]    Dlanes;
  assign Dlanes = 32'(rows) * N;
  always_comb begin
    for (int i = 0; i < N; i++) begin
      logic signed [63:0] d, v;
      logic [63:0] e;
      d = 64'(Dlanes) * 64'($signed(buf8[rsel][xr][8*i +: 8])) - 64'(S);
      v = d * 32 * $signed(inv) + 64'sd8388608;
      e = 64'(buf16[xr][16*i +: 16]) * inv + 64'd2097152;
      case (op)
        OP_LN:   emit_row[8*i +: 8] = sat8(v >>> 24);
        OP_SM:   emit_row[8*i +: 8] = 8'(e >> 22);
        default: emit_row[8*i +: 8] = buf8[rsel][xr][8*i +: 8];
      endcase
    end
  end

  assign cmd_ready_o = (fs == F_IDLE);
  assign in_ready_o  = (fs == F_LOAD);
  assign out_valid_o = (bs == B_EMIT);
  assign out_data_o  = emit_row;
  assign busy_o      = (fs != F_IDLE) || (bs != B_IDLE);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      fs <= F_IDLE; wsel <= 1'b0; f_op <= '0; f_rows <= '0; cnt <= '0; ma <= '0; mb <= '0;
      fS <= '0; fSq <= '0; fmx <= '0;
      bs <= B_IDLE; rsel <= 1'b0; op <= '0; rows <= '0; xr <= '0; S <= '0; mx <= '0; Z <= '0;
      inv <= '0; rem <= '0; rad <= '0; root <= '0; bitv <= '0; num <= '0; den <= '0; quo <= '0; it <= '0;
    end else begin
      // ---------------- front
      case (fs)
        F_IDLE: if (cmd_valid_i) begin
          f_op <= cmd_op_i; f_rows <= cmd_rows_i; ma <= cmd_ma_i; mb <= cmd_mb_i;
          cnt <= '0; fS <= '0; fSq <= '0; fmx <= -8'sd128; fs <= F_LOAD;
        end
        F_LOAD: if (in_valid_i) begin
          case (f_op)
            OP_ADD:  buf8[wsel][cnt] <= add_row;
            OP_LUT:  buf8[wsel][cnt] <= lut_row;
            default: buf8[wsel][cnt] <= in_a_i;
          endcase
          fS <= fS + row_sum; fSq <= fSq + row_sq;
          if (row_max > fmx) fmx <= row_max;
          cnt <= cnt + 1'b1;
          if (cnt + 1'b1 == f_rows) fs <= F_HAND;
        end
        F_HAND: if (hand) begin fs <= F_IDLE; wsel <= ~wsel; end
        default: fs <= F_IDLE;
      endcase
      // ---------------- back
      case (bs)
        B_IDLE: if (hand) begin
          rsel <= wsel; op <= f_op; rows <= f_rows; S <= fS; mx <= fmx; Z <= '0; xr <= '0;
          case (f_op)
            OP_SM: bs <= B_EXPSUM;
            OP_LN: begin
              rad <= 64'(32'(f_rows) * N) * 64'(fSq) - 64'(fS) * 64'(fS);
              root <= '0; bitv <= 64'd1 << 32; bs <= B_SQRT;
            end
            default: bs <= B_EMIT;
          endcase
        end
        B_EXPSUM: begin
          buf16[xr] <= exp_row;
          Z <= Z + exp_sum;
          xr <= xr + 1'b1;
          if (xr + 1'b1 == rows) begin
            num <= 64'd127 << 22; den <= 64'(Z + exp_sum); quo <= '0; rem <= 64'd127 << 8; it <= 7'd13;  // Z >= 2^15 -> q < 2^14
            bs <= B_DIV;
          end
        end
        B_SQRT: begin                           // integer square root, 2 result bits per cycle
          logic [63:0] ra, oa, ba, rb, ob;
          ra = rad; oa = root; ba = bitv;
          if (ra >= oa + ba) begin ra = ra - (oa + ba); oa = (oa >> 1) + ba; end else oa = oa >> 1;
          rb = ra; ob = oa;
          if (ba > 1) begin
            if (rb >= ob + (ba >> 2)) begin rb = rb - (ob + (ba >> 2)); ob = (ob >> 1) + (ba >> 2); end else ob = ob >> 1;
          end
          if (ba <= 4) begin
            num <= 64'd1 << 24; den <= (ob == 0) ? 64'd1 : ob; quo <= '0; rem <= '0; it <= 7'd24;
            bs <= B_DIV;
          end else begin
            rad <= rb; root <= ob; bitv <= ba >> 4;
          end
        end
        B_DIV: begin                            // restoring division, 2 quotient bits per cycle
          logic [64:0] ra, rb;
          logic [63:0] rema, qa;
          logic        ga, gb;
          ra = {rem, num[it]};
          ga = (ra >= {1'b0, den});
          rema = ga ? 64'(ra - {1'b0, den}) : ra[63:0];
          qa = quo | (64'(ga) << it);
          if (it == 0) begin
            inv <= qa; xr <= '0; bs <= B_EMIT;
          end else begin
            rb = {rema, num[it - 1]};
            gb = (rb >= {1'b0, den});
            if (it == 1) begin
              inv <= qa | 64'(gb); xr <= '0; bs <= B_EMIT;
            end else begin
              rem <= gb ? 64'(rb - {1'b0, den}) : rb[63:0];
              quo <= qa | (64'(gb) << (it - 1));
              it  <= it - 2;
            end
          end
        end
        B_EMIT: if (out_ready_i) begin
          xr <= xr + 1'b1;
          if (xr + 1'b1 == rows) bs <= B_IDLE;
        end
        default: bs <= B_IDLE;
      endcase
    end
  end
endmodule
