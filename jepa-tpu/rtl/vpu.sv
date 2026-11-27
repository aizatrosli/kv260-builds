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
// Flow: command, then `rows` input rows (in_a, and in_b for ADD), then `rows` output rows.
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

  typedef enum logic [2:0] {S_IDLE, S_LOAD, S_EXPSUM, S_SQRT, S_DIV, S_EMIT} state_e;
  state_e st;

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

  logic [1:0]           op;
  logic [RW-1:0]        rows, cnt;
  logic signed [17:0]   ma, mb;
  logic [N*8-1:0]       buf8  [MAXR];        // stored input / finished int8 output
  logic [N*16-1:0]      buf16 [MAXR];        // softmax: exp values

  // statistics
  logic signed [31:0]   S;                   // sum a        (|S| <= MAXR*N*128)
  logic [31:0]          Sq;                  // sum a^2
  logic signed [7:0]    mx;                  // running max
  logic [31:0]          Z;                   // sum e
  logic [63:0]          inv;                 // reciprocal result

  // iterative isqrt / divider
  logic [63:0] rem, rad, root, bitv;
  logic [63:0] num, den, quo;
  logic [6:0]  it;

  // ---------------- per-row combinational lanes ----------------
  function automatic logic signed [7:0] sat8(input logic signed [63:0] v);
    return (v > 127) ? 8'sd127 : (v < -128) ? -8'sd128 : v[7:0];
  endfunction

  logic [N*8-1:0]  add_row, lut_row, emit_row;
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

  // exp stage: row of logits -> row of exps
  logic [RW-1:0]  xr;                     // row index for EXPSUM / EMIT
  logic [N*16-1:0] exp_row;
  logic [31:0]     exp_sum;
  always_comb begin
    exp_sum = '0;
    for (int i = 0; i < N; i++) begin
      logic signed [8:0] d;
      d = 9'(mx) - 9'($signed(buf8[xr][8*i +: 8]));
      exp_row[16*i +: 16] = exp_lut[d[7:0]];
      exp_sum += 32'(exp_lut[d[7:0]]);
    end
  end

  // emit stage
  logic [31:0] Dlanes;
  assign Dlanes = 32'(rows) * N;
  always_comb begin
    for (int i = 0; i < N; i++) begin
      logic signed [63:0] d, v;
      logic [63:0] e;
      d = 64'(Dlanes) * 64'($signed(buf8[xr][8*i +: 8])) - 64'(S);
      v = d * 32 * $signed(inv) + 64'sd8388608;
      e = 64'(buf16[xr][16*i +: 16]) * inv + 64'd2097152;
      case (op)
        OP_LN:   emit_row[8*i +: 8] = sat8(v >>> 24);
        OP_SM:   emit_row[8*i +: 8] = 8'(e >> 22);
        default: emit_row[8*i +: 8] = buf8[xr][8*i +: 8];
      endcase
    end
  end

  assign cmd_ready_o = (st == S_IDLE);
  assign in_ready_o  = (st == S_LOAD);
  assign out_valid_o = (st == S_EMIT);
  assign out_data_o  = emit_row;
  assign busy_o      = (st != S_IDLE);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      st <= S_IDLE; cnt <= '0; xr <= '0; op <= '0; rows <= '0; ma <= '0; mb <= '0;
      S <= '0; Sq <= '0; mx <= '0; Z <= '0; inv <= '0;
      rem <= '0; rad <= '0; root <= '0; bitv <= '0; num <= '0; den <= '0; quo <= '0; it <= '0;
    end else begin
      case (st)
        S_IDLE: if (cmd_valid_i) begin
          op <= cmd_op_i; rows <= cmd_rows_i; ma <= cmd_ma_i; mb <= cmd_mb_i;
          cnt <= '0; S <= '0; Sq <= '0; mx <= -8'sd128; Z <= '0;
          st <= S_LOAD;
        end
        S_LOAD: if (in_valid_i) begin
          case (op)
            OP_ADD:  buf8[cnt] <= add_row;
            OP_LUT:  buf8[cnt] <= lut_row;
            default: buf8[cnt] <= in_a_i;
          endcase
          S  <= S + row_sum;
          Sq <= Sq + row_sq;
          if (row_max > mx) mx <= row_max;
          cnt <= cnt + 1'b1;
          if (cnt + 1'b1 == rows) begin
            xr <= '0;
            case (op)
              OP_SM: st <= S_EXPSUM;
              OP_LN: begin
                // V = D*Sq - S^2 (S, Sq still missing this row: fold it in here)
                rad  <= 64'(Dlanes) * 64'(Sq + row_sq) - 64'($signed(S + row_sum)) * 64'($signed(S + row_sum));
                rem  <= '0; root <= '0; bitv <= 64'd1 << 40; st <= S_SQRT;
              end
              default: st <= S_EMIT;
            endcase
          end
        end
        S_EXPSUM: begin
          buf16[xr] <= exp_row;
          Z <= Z + exp_sum;
          xr <= xr + 1'b1;
          if (xr + 1'b1 == rows) begin
            num <= 64'd127 << 22; den <= 64'(Z + exp_sum); quo <= '0; rem <= '0; it <= 7'd29;
            st <= S_DIV;
          end
        end
        S_SQRT: begin                           // bit-by-bit integer square root of rad
          if (bitv == 0) begin
            num <= 64'd1 << 24; den <= (root == 0) ? 64'd1 : root; quo <= '0; rem <= '0; it <= 7'd29;
            st <= S_DIV;
          end else begin
            if (rad >= root + bitv) begin
              rad  <= rad - (root + bitv);
              root <= (root >> 1) + bitv;
            end else begin
              root <= root >> 1;
            end
            bitv <= bitv >> 2;
          end
        end
        S_DIV: begin                            // restoring division, 1 bit per cycle
          logic [64:0] r2;
          r2 = {rem, num[it]};
          if (r2 >= {1'b0, den}) begin
            rem <= 64'(r2 - {1'b0, den}); quo <= quo | (64'd1 << it);
          end else begin
            rem <= r2[63:0];
          end
          if (it == 0) begin
            inv <= (r2 >= {1'b0, den}) ? (quo | 64'd1) : quo;
            xr <= '0; st <= S_EMIT;
          end
          it <= it - 1'b1;
        end
        S_EMIT: if (out_ready_i) begin
          xr <= xr + 1'b1;
          if (xr + 1'b1 == rows) st <= S_IDLE;
        end
        default: st <= S_IDLE;
      endcase
    end
  end
endmodule
