// features: F0..F7 of SPEC.md from the top of both book sides and the trade-flow register.
//
// Fixed-latency, fully pipelined, never stalls (one new vector may enter every cycle; the
// book only produces one every >= 2 cycles). Validity travels in a shift register; flow
// control is done upstream with credits so nothing here needs a ready signal.
//
// Exact SPEC semantics:
//   F0 b1, F1 a1 (0 if side empty)              F2 spread = a1-b1 mod 2^32 if both sides, else 0
//   F3/F4 sum of the top-5 volumes mod 2^32      F6 flow (register lives in tob_engine)
//   F5 imb = trunc0(((B-A)<<15) / (B+A)), B,A unsigned 32, B+A 33 bit, 0 if B+A == 0
//      computed as sign(B-A) * floor(|B-A|*2^15 / (B+A)): |q| <= 2^15 so a 16-bit quotient
//   F7 micro = floor(sp_u * bv1 / (bv1+av1)), sp_u = F2 as unsigned, 64-bit product,
//      33-bit divisor, 0 if F2 == 0 or bv1+av1 == 0; the quotient is <= sp_u so 32 bits
//
// Pipeline (cycle c0 = in_valid_i, the cycle after the book committed):
//   c0 -> S1  spread, pairwise top-5 partial sums, copies of b1/a1/bv1/av1/flow
//   c1 -> S2  B, A (3-input adds), bv1+av1, product stage 1 (sp_u * bv1)
//   c2 -> S3  |B-A| and sign (two parallel subtractors + mux), B+A, product stage 2, 3*(bv1+av1)
//   c3 -> S4  product stage 3, |B-A|<<15, 3*(B+A)          (3 regs after the operands: DSP48E2
//                                                           A/B, M and P registers, 4 DSPs)
//   c4 .. c19 radix-4 divider stages: micro 16 (32-bit q), imb 8 (16-bit q, then delayed 8)
//   c20 -> out  sign/zero fix-up; out_valid_o is high in c21  => LATENCY = 21 cycles
`timescale 1ns/1ps

module features (
  input  wire         clk_i,
  input  wire         rst_ni,
  input  wire         in_valid_i,
  input  wire [31:0]  tag_i,           // carried through unchanged (rx cycle stamp)
  input  wire         bid_ok_i,        // bid side non-empty
  input  wire         ask_ok_i,
  input  wire [31:0]  bid_p0_i,        // best bid price (0 if empty)
  input  wire [31:0]  ask_p0_i,
  input  wire [159:0] bid_v_i,         // bid volumes, levels 1..5 (0 if missing)
  input  wire [159:0] ask_v_i,
  input  wire [31:0]  flow_i,
  output reg          out_valid_o,
  output reg  [255:0] out_feat_o,      // F_k at [32*k +: 32]
  output reg  [31:0]  out_tag_o
);
  localparam QM = 32, QI = 16;         // quotient widths: micro, imbalance
  localparam LM = QM / 2, LI = QI / 2; // radix-4 divider latencies
  localparam TD = 4;                   // cycle in which divider inputs are visible
  localparam LAT = TD + LM + 1;        // in_valid -> out_valid

  // ---------------- S1 ----------------
  reg [31:0] s1_b1, s1_a1, s1_sp, s1_bv1, s1_av1, s1_flow, s1_tag;
  reg [31:0] s1_b01, s1_b23, s1_b4, s1_a01, s1_a23, s1_a4;
  always @(posedge clk_i) begin
    s1_b1   <= bid_ok_i ? bid_p0_i : 32'd0;
    s1_a1   <= ask_ok_i ? ask_p0_i : 32'd0;
    s1_sp   <= (bid_ok_i && ask_ok_i) ? (ask_p0_i - bid_p0_i) : 32'd0;
    s1_bv1  <= bid_v_i[31:0];
    s1_av1  <= ask_v_i[31:0];
    s1_b01  <= bid_v_i[31:0]   + bid_v_i[63:32];
    s1_b23  <= bid_v_i[95:64]  + bid_v_i[127:96];
    s1_b4   <= bid_v_i[159:128];
    s1_a01  <= ask_v_i[31:0]   + ask_v_i[63:32];
    s1_a23  <= ask_v_i[95:64]  + ask_v_i[127:96];
    s1_a4   <= ask_v_i[159:128];
    s1_flow <= flow_i;
    s1_tag  <= tag_i;
  end

  // ---------------- S2 ----------------
  reg [31:0] s2_B, s2_A, s2_b1, s2_a1, s2_sp, s2_flow, s2_tag;
  reg [32:0] s2_dm;
  reg        s2_zmic;
  reg [63:0] p1;
  always @(posedge clk_i) begin
    s2_B    <= s1_b01 + s1_b23 + s1_b4;
    s2_A    <= s1_a01 + s1_a23 + s1_a4;
    s2_dm   <= {1'b0, s1_bv1} + {1'b0, s1_av1};
    s2_zmic <= (s1_sp == 32'd0) || ((s1_bv1 | s1_av1) == 32'd0);
    p1      <= {32'd0, s1_sp} * {32'd0, s1_bv1};
    s2_b1 <= s1_b1; s2_a1 <= s1_a1; s2_sp <= s1_sp; s2_flow <= s1_flow; s2_tag <= s1_tag;
  end

  // ---------------- S3 ----------------
  reg [31:0] s3_absd;
  reg        s3_neg, s3_zimb;
  reg [32:0] s3_den, s3_dm;
  reg [34:0] s3_dm3;
  reg [63:0] p2;
  wire [32:0] d_ba = {1'b0, s2_B} - {1'b0, s2_A};
  /* verilator lint_off UNUSEDSIGNAL */
  wire [32:0] d_ab = {1'b0, s2_A} - {1'b0, s2_B};           // only used when B < A: [32] unused
  /* verilator lint_on UNUSEDSIGNAL */
  always @(posedge clk_i) begin
    s3_neg  <= d_ba[32];                          // B < A
    s3_absd <= d_ba[32] ? d_ab[31:0] : d_ba[31:0];
    s3_den  <= {1'b0, s2_B} + {1'b0, s2_A};
    s3_zimb <= (s2_B == 32'd0) && (s2_A == 32'd0);
    s3_dm   <= s2_dm;
    s3_dm3  <= {2'b00, s2_dm} + {1'b0, s2_dm, 1'b0};
    p2      <= p1;
  end

  // ---------------- S4: divider operands ----------------
  reg [63:0] p3;
  reg [32:0] s4_dm, s4_den;
  reg [34:0] s4_dm3, s4_den3;
  reg [47:0] s4_inum;
  always @(posedge clk_i) begin
    p3      <= p2;
    s4_dm   <= s3_dm;
    s4_dm3  <= s3_dm3;
    s4_inum <= {1'b0, s3_absd, 15'd0};
    s4_den  <= s3_den;
    s4_den3 <= {2'b00, s3_den} + {1'b0, s3_den, 1'b0};
  end

  wire [QM-1:0] q_mic;
  wire [QI-1:0] q_imb, q_imb_d;
  div_r4_pipe #(.NW(64), .DW(33), .QW(QM)) u_div_mic (
    .clk_i(clk_i), .n_i(p3), .d_i(s4_dm), .d3_i(s4_dm3), .q_o(q_mic));
  div_r4_pipe #(.NW(48), .DW(33), .QW(QI)) u_div_imb (
    .clk_i(clk_i), .n_i(s4_inum), .d_i(s4_den), .d3_i(s4_den3), .q_o(q_imb));

  // ---------------- alignment to cycle TD+LM ----------------
  wire [31:0] b1_d, a1_d, sp_d, B_d, A_d, flow_d, tag_d;
  wire        zmic_d, neg_d, zimb_d;
  delay_sr #(.W(7*32+1), .D(TD + LM - 2)) u_dl_s2 (
    .clk_i(clk_i), .d_i({s2_b1, s2_a1, s2_sp, s2_B, s2_A, s2_flow, s2_tag, s2_zmic}),
    .q_o({b1_d, a1_d, sp_d, B_d, A_d, flow_d, tag_d, zmic_d}));
  delay_sr #(.W(2), .D(TD + LM - 3)) u_dl_s3 (
    .clk_i(clk_i), .d_i({s3_neg, s3_zimb}), .q_o({neg_d, zimb_d}));
  delay_sr #(.W(QI), .D(LM - LI)) u_dl_imb (
    .clk_i(clk_i), .d_i(q_imb), .q_o(q_imb_d));

  wire [31:0] qi32 = {{(32-QI){1'b0}}, q_imb_d};
  wire [31:0] imb  = zimb_d ? 32'd0 : neg_d ? (32'd0 - qi32) : qi32;
  wire [31:0] mic  = zmic_d ? 32'd0 : q_mic;

  // ---------------- output ----------------
  reg [LAT-2:0] vld;                   // vld[k] high in cycle c(k+1)
  always @(posedge clk_i) begin
    if (!rst_ni) begin
      vld <= {(LAT-1){1'b0}}; out_valid_o <= 1'b0;
    end else begin
      vld <= {vld[LAT-3:0], in_valid_i};
      out_valid_o <= vld[LAT-2];
    end
  end
  always @(posedge clk_i) begin
    out_feat_o <= {mic, flow_d, imb, A_d, B_d, sp_d, a1_d, b1_d};
    out_tag_o  <= tag_d;
  end
endmodule
