// Register-level test of kv260_i3c_top: drives the AXI4-Lite map exactly as the PS software does
// (sw/kv260_driver.py uses the same offsets) and checks the bus with the behavioural Target.
`timescale 1ns/1ps
module tb_axil;
  import crc8_pec_pkg::*;
  logic clk = 0, rst_n = 0;
  always #5 clk = ~clk;

  logic [7:0] awaddr = 0, araddr = 0; logic awvalid = 0, wvalid = 0, bready = 1, arvalid = 0, rready = 1;
  logic [31:0] wdata = 0; logic awready, wready, bvalid, arready, rvalid; logic [1:0] bresp, rresp; logic [31:0] rdata;
  wire scl, sda, sda_pu, trig;
  logic t_d0, t_d1;

  kv260_i3c_top dut (.clk, .rst_n,
    .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
    .i3c_scl(scl), .i3c_sda(sda), .i3c_sda_pu(sda_pu), .trig(trig));

  // board: Target open-drain/push-pull onto SDA; pull-up modelled as weak 1
  assign (weak0, weak1) sda = 1'b1;
  assign sda = t_d0 ? 1'b0 : (t_d1 ? 1'b1 : 1'bz);

  i3c_target_bfm #(.PID(48'h04D2_0000_3003), .NAME("T")) tgt (.scl, .sda, .sda_d0(t_d0), .sda_d1(t_d1));
  i3c_timing_mon mon (.scl, .sda, .od_i(sda_pu === 1'b1), .ctrl_drv_i(dut.sda_oe));

  int errors = 0;
  task automatic check(input bit c, input string m);
    if (!c) begin errors++; $display("FAIL: %s", m); end else $display("ok: %s", m);
  endtask

  task automatic axw(input logic [7:0] a, input logic [31:0] d);
    @(negedge clk); awaddr = a; awvalid = 1; wdata = d; wvalid = 1;
    fork
      begin do @(posedge clk); while (!awready); @(negedge clk); awvalid = 0; end
      begin do @(posedge clk); while (!wready);  @(negedge clk); wvalid = 0; end
    join
    do @(posedge clk); while (!bvalid);
  endtask
  task automatic axr(input logic [7:0] a, output logic [31:0] d);
    @(negedge clk); araddr = a; arvalid = 1;
    do @(posedge clk); while (!arready);
    @(negedge clk); arvalid = 0;
    while (!rvalid) @(posedge clk);
    d = rdata;
  endtask
  task automatic wait_free();
    logic [31:0] s;
    repeat (20) @(posedge clk);
    do axr(8'h08, s); while (!s[0] || s[1]);
  endtask

  localparam logic [3:0] OP_START = 1, OP_STOP = 2, OP_ADDR = 3, OP_WRITE = 4, OP_DAA = 8;
  function automatic logic [31:0] C(input logic [3:0] op, input logic [7:0] fl, input logic [15:0] d);
    return {op, fl, 4'd0, d};
  endfunction

  initial begin
    logic [31:0] r;
    byte unsigned msg[];
    repeat (5) @(posedge clk); rst_n = 1;
    axr(8'h00, r); check(r == 32'h13C3_0100, $sformatf("ID = %08x", r));
    wait_free();
    // ENTDAA via CMD FIFO
    axw(8'h20, C(OP_START, 0, 0)); axw(8'h20, C(OP_ADDR, 8'h01, 16'h00FC)); axw(8'h20, C(OP_WRITE, 0, 16'h0007));
    axw(8'h20, C(OP_DAA, 0, 16'h0010)); axw(8'h20, C(OP_DAA, 0, 16'h0011)); axw(8'h20, C(OP_STOP, 0, 0));
    wait_free();
    check(tgt.has_da && tgt.da == 7'h10, "Target got DA 0x10 through the register interface");
    begin
      int n; n = 0;
      do begin axr(8'h24, r); if (r != 0) n++; end while (r != 0);
      check(n == 11, $sformatf("%0d responses popped (7E ACK + 8 PID/BCR/DCR + 2 DAA)", n));
    end
    // MCTP: 70-byte PLDM-typed message, TU 64 -> 2 packets
    msg = new[70]; msg[0] = 8'h01; for (int i = 1; i < 70; i++) msg[i] = 8'(i);
    foreach (msg[i]) axw(8'h40, {24'd0, msg[i]});
    axw(8'h34, {16'd70, 8'h08, 8'h20});                         // len, src EID 0x08, dst EID 0x20
    axw(8'h30, {8'd64, 2'b00, 1'b1, 1'b0, 4'h1, 2'b00, 2'd3, 3'd6, 1'b1, 1'b0, 7'h10});
    axw(8'h38, 32'h1);
    wait_free();
    axr(8'h3C, r); check(r == 32'h0000_0002, $sformatf("MCTP_STAT = %08x (2 ACKed, 0 NACKed)", r));
    axr(8'h38, r); check(r[5:4] == 2'd1, $sformatf("next seq = %0d", r[5:4]));
    check(tgt.wframes.size() == 2, $sformatf("%0d packets received", tgt.wframes.size()));
    begin
      byte unsigned got[$]; logic [7:0] c;
      foreach (tgt.wframes[k]) begin
        byte unsigned f[$]; f = tgt.wframes[k];
        c = crc8_pec(8'h00, {7'h10, 1'b0});
        for (int i = 0; i < f.size() - 1; i++) c = crc8_pec(c, f[i]);
        check(c == f[f.size() - 1], $sformatf("packet %0d PEC 0x%02x", k, c));
        check(f[0] == 8'h01 && f[1] == 8'h20 && f[2] == 8'h08 &&
              f[3] == {k == 0, k == 1, 2'(3 + k), 1'b1, 3'd6}, $sformatf("packet %0d header %02x %02x %02x %02x", k, f[0], f[1], f[2], f[3]));
        for (int i = 4; i < f.size() - 1; i++) got.push_back(f[i]);
      end
      begin
        bit same; same = (got.size() == msg.size());
        foreach (msg[i]) if (same && got[i] != msg[i]) same = 0;
        check(same, "payload reassembles to the message");
      end
    end
    mon.report();
    check(mon.errors == 0, "no timing violations");
    if (errors == 0) $display("tb_axil: ALL PASS"); else $display("tb_axil: %0d FAILURES", errors);
    $finish;
  end
  initial begin #5ms; $display("tb_axil: TIMEOUT"); $finish; end
endmodule
