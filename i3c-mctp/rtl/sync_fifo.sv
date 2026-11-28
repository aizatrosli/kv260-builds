// Simple synchronous FIFO (first-word-fall-through), power-of-two depth.
`timescale 1ns/1ps
module sync_fifo #(parameter int W = 32, parameter int DEPTH = 512) (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         clr_i,
  input  logic         wr_i,
  input  logic [W-1:0] wdata_i,
  output logic         full_o,
  input  logic         rd_i,
  output logic [W-1:0] rdata_o,
  output logic         empty_o,
  output logic [$clog2(DEPTH):0] count_o
);
  localparam int AW = $clog2(DEPTH);
  logic [W-1:0] mem [DEPTH];
  logic [AW:0]  wp, rp;
  assign count_o = wp - rp;
  assign empty_o = (wp == rp);
  assign full_o  = (count_o == DEPTH[AW:0]);
  assign rdata_o = mem[rp[AW-1:0]];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin wp <= '0; rp <= '0; end
    else if (clr_i) begin wp <= '0; rp <= '0; end
    else begin
      if (wr_i && !full_o) wp <= wp + 1'b1;
      if (rd_i && !empty_o) rp <= rp + 1'b1;
    end
  end
  always_ff @(posedge clk_i) if (wr_i && !full_o) mem[wp[AW-1:0]] <= wdata_i;
endmodule
