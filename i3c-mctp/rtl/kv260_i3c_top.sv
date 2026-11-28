// KV260 pad wrapper for the I3C/MCTP generator. Use it as a module reference in a Vivado block design
// next to the Zynq UltraScale+ PS (connect s_axi_* to M_AXI_HPM0_FPD through an AXI SmartConnect,
// clk to pl_clk0 = 100 MHz, rst_n to the proc_sys_reset peripheral_aresetn).
// Pins go to PMOD J2, see constraints/kv260_pmod_i3c.xdc and docs/README.md (level translation!).
`timescale 1ns/1ps
module kv260_i3c_top (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [7:0]  s_axi_awaddr,  input logic s_axi_awvalid, output logic s_axi_awready,
  input  logic [31:0] s_axi_wdata,   input logic [3:0] s_axi_wstrb, input logic s_axi_wvalid, output logic s_axi_wready,
  output logic [1:0]  s_axi_bresp,   output logic s_axi_bvalid, input logic s_axi_bready,
  input  logic [7:0]  s_axi_araddr,  input logic s_axi_arvalid, output logic s_axi_arready,
  output logic [31:0] s_axi_rdata,   output logic [1:0] s_axi_rresp, output logic s_axi_rvalid, input logic s_axi_rready,
  output wire         i3c_scl,       // pmod[0], LOC H12: Push-Pull
  inout  wire         i3c_sda,       // pmod[1], LOC E10: Push-Pull / Open-Drain
  inout  wire         i3c_sda_pu,    // pmod[2], LOC D10: drives '1' through Rp to SDA in OD phases, else Z
  output wire         trig           // pmod[3], LOC C11: high while a frame is on the bus
);
  logic scl, sda_oe, sda_o, sda_pu_en, frame;
  i3c_mctp_axil u_axil (
    .clk_i(clk), .rst_ni(rst_n),
    .s_axi_awaddr, .s_axi_awvalid, .s_axi_awready, .s_axi_wdata, .s_axi_wstrb, .s_axi_wvalid, .s_axi_wready,
    .s_axi_bresp, .s_axi_bvalid, .s_axi_bready, .s_axi_araddr, .s_axi_arvalid, .s_axi_arready,
    .s_axi_rdata, .s_axi_rresp, .s_axi_rvalid, .s_axi_rready,
    .scl_o(scl), .sda_oe_o(sda_oe), .sda_o(sda_o), .sda_i(i3c_sda), .sda_pu_en_o(sda_pu_en), .frame_o(frame));
  // Vivado infers IOBUF / OBUFT from these tri-state assigns (UG571: HD bank supports 3-state outputs).
  assign i3c_scl    = scl;
  assign i3c_sda    = sda_oe ? sda_o : 1'bz;
  assign i3c_sda_pu = sda_pu_en ? 1'b1 : 1'bz;
  assign trig       = frame;
endmodule
