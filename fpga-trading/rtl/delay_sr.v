// delay_sr: fixed W-bit delay line of D cycles (D >= 1). No reset on purpose, so Vivado maps it
// to SRL16/SRL32 shift-register LUTs (one LUT per bit per 32 cycles) instead of flip-flops.
`timescale 1ns/1ps

module delay_sr #(
  parameter W = 32,
  parameter D = 1
) (
  input  wire         clk_i,
  input  wire [W-1:0] d_i,
  output wire [W-1:0] q_o
);
  reg [W-1:0] sr [0:D-1];
  integer i;
  always @(posedge clk_i) begin
    sr[0] <= d_i;
    for (i = 1; i < D; i = i + 1) sr[i] <= sr[i-1];
  end
  assign q_o = sr[D-1];
endmodule
