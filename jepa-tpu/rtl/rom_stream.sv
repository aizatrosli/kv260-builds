// rom_stream: a read-only memory replayed as a valid/ready word stream. Used to hard-code the
// JEPA encoder: one instance holds the weight-tile words (replaces the host-filled Weight FIFO),
// another holds the program (replaces the Instruction FIFO). On the FPGA, INIT_FILE becomes the
// BRAM/URAM initial contents in the bitstream; no host, no DDR, no DMA.
// start_i rewinds to word 0 and streams `len_i` words (1 word/cycle when ready is high).
`timescale 1ns/1ps
module rom_stream #(
  parameter int unsigned W     = 128,
  parameter int unsigned DEPTH = 1024,
  parameter string       INIT_FILE = ""
) (
  input  logic                      clk_i,
  input  logic                      rst_ni,
  input  logic                      start_i,
  input  logic [$clog2(DEPTH):0]    len_i,
  output logic                      valid_o,
  input  logic                      ready_i,
  output logic [W-1:0]              data_o,
  output logic                      done_o
);
  (* rom_style = "block" *) logic [W-1:0] mem [DEPTH];
  initial if (INIT_FILE != "") $readmemh(INIT_FILE, mem);

  logic [$clog2(DEPTH):0] rd, len;
  logic                   have;              // data_o holds word rd-1
  logic                   active;
  wire                    pop  = valid_o && ready_i;
  wire                    load = active && (rd < len) && (!have || pop);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin rd <= '0; len <= '0; have <= 1'b0; active <= 1'b0; end
    else if (start_i) begin rd <= '0; len <= len_i; have <= 1'b0; active <= 1'b1; end
    else begin
      if (load) rd <= rd + 1'b1;
      if (load) have <= 1'b1; else if (pop) have <= 1'b0;
      if (active && rd == len && (!have || pop)) active <= 1'b0;
    end
  end
  always_ff @(posedge clk_i) if (load) data_o <= mem[rd[$clog2(DEPTH)-1:0]];   // synchronous BRAM read
  assign valid_o = have;
  assign done_o  = !active && !have;
endmodule
