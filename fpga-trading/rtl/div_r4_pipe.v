// div_r4_pipe: fully pipelined radix-4 restoring divider, q = floor(n / d), one result per cycle.
//
// Precondition (guaranteed by the caller): q < 2^QW, i.e. (n >> QW) < d, and NW-QW < DW.
// Then every partial remainder is < d and fits DW bits. d == 0 gives garbage; the caller
// masks it (both SPEC features define x/0 as 0).
//
// Each stage retires 2 quotient bits: r = {rem, next 2 numerator bits} (< 4d), compared against
// d, 2d, 3d with three parallel (DW+3)-bit subtractors; the largest non-negative one wins.
// That is one carry chain + a 4:1 mux per stage, comfortably inside 4 ns on UltraScale+.
// 3d is supplied by the caller (computed one stage earlier, off the critical path).
//
// Latency: QW/2 cycles from (n_i, d_i, d3_i) to q_o (all inputs must be registered and held
// in step: n_i, d_i, d3_i sampled together). No handshake: the pipeline never stalls; the
// caller tracks validity with a parallel shift register.
`timescale 1ns/1ps

module div_r4_pipe #(
  parameter NW = 64,                  // numerator width
  parameter DW = 33,                  // divisor width
  parameter QW = 32                   // quotient width (even)
) (
  input  wire          clk_i,
  input  wire [NW-1:0] n_i,
  input  wire [DW-1:0] d_i,
  input  wire [DW+1:0] d3_i,          // 3 * d_i
  output wire [QW-1:0] q_o
);
  localparam S = QW / 2;

  // stage k (1..S) registers: remainder, {unconsumed numerator bits | quotient bits}, d, 3d
  reg [DW-1:0] rem [1:S];
  reg [QW-1:0] nq  [1:S];
  reg [DW-1:0] dd  [1:S];
  reg [DW+1:0] d3  [1:S];

  genvar k;
  generate
    for (k = 1; k <= S; k = k + 1) begin : g_stage
      wire [DW-1:0] rem_p, dd_p;
      wire [QW-1:0] nq_p;
      wire [DW+1:0] d3_p;
      if (k == 1) begin : g_first
        assign rem_p = {{(DW-(NW-QW)){1'b0}}, n_i[NW-1:QW]};
        assign nq_p  = n_i[QW-1:0];
        assign dd_p  = d_i;
        assign d3_p  = d3_i;
      end else begin : g_next
        assign rem_p = rem[k-1];
        assign nq_p  = nq[k-1];
        assign dd_p  = dd[k-1];
        assign d3_p  = d3[k-1];
      end
      wire [DW+1:0] r  = {rem_p, nq_p[QW-1:QW-2]};
      wire [DW+2:0] t1 = {1'b0, r} - {3'b000, dd_p};
      wire [DW+2:0] t2 = {1'b0, r} - {2'b00, dd_p, 1'b0};
      wire [DW+2:0] t3 = {1'b0, r} - {1'b0, d3_p};
      wire [1:0]    dig = !t3[DW+2] ? 2'd3 : !t2[DW+2] ? 2'd2 : !t1[DW+2] ? 2'd1 : 2'd0;
      wire [DW-1:0] rn  = !t3[DW+2] ? t3[DW-1:0] : !t2[DW+2] ? t2[DW-1:0] :
                          !t1[DW+2] ? t1[DW-1:0] : r[DW-1:0];
      always @(posedge clk_i) begin
        rem[k] <= rn;
        nq[k]  <= {nq_p[QW-3:0], dig};
        dd[k]  <= dd_p;
        d3[k]  <= d3_p;
      end
    end
  endgenerate

  assign q_o = nq[S];
endmodule
