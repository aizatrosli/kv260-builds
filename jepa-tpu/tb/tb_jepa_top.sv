// tb_jepa_top: end-to-end hard-coded JEPA encoder. tpu_fabric (UB/ACC memories + AXI window)
// + jepa_top (tpu_core + sequencer + VPU + ROMs). The host only writes the input image into
// the UB, pulses run, waits for the interrupt and reads the encoder output back, which must
// equal exp_out.hex (compile_jepa.Machine == quantize.int_forward) bit for bit.
//   +ROM=<dir from compile_jepa.py>
`timescale 1ns/1ps
`include "axi/typedef.svh"

module tb_jepa_top;
  import tpu_pkg::*;
  localparam time TCK = 10ns, TA = 2ns;
  localparam int unsigned N = 16;

  logic clk = 1'b0, rst_n = 1'b0;
  always #(TCK/2) clk = ~clk;

  fs_req_t  mreq [2];
  fs_resp_t mrsp [2];
  fm_req_t  hp0_req;
  fm_resp_t hp0_rsp;
  assign hp0_rsp = '0;

  logic ub_rd_en, ub_wr_en, acc_rd_en, acc_wr_en;
  logic [15:0] ub_rd_addr, ub_wr_addr;
  logic [10:0] acc_rd_addr, acc_wr_addr;
  data_t ub_rd_data, ub_wr_data;
  logic [15:0] ub_wr_be;
  acc_row_t acc_rd_data, acc_wr_data;
  logic [3:0] acc_wr_be;
  logic [3:0] fab_err;

  tpu_fabric i_fab (
    .clk_i(clk), .rst_ni(rst_n),
    .dma_req_i(mreq[0]), .dma_rsp_o(mrsp[0]), .hpm_req_i(mreq[1]), .hpm_rsp_o(mrsp[1]),
    .hp0_req_o(hp0_req), .hp0_rsp_i(hp0_rsp),
    .ub_rd_en_i(ub_rd_en), .ub_rd_addr_i(ub_rd_addr), .ub_rd_data_o(ub_rd_data),
    .ub_wr_en_i(ub_wr_en), .ub_wr_addr_i(ub_wr_addr), .ub_wr_data_i(ub_wr_data), .ub_wr_be_i(ub_wr_be),
    .acc_rd_en_i(acc_rd_en), .acc_rd_addr_i(acc_rd_addr), .acc_rd_data_o(acc_rd_data),
    .acc_wr_en_i(acc_wr_en), .acc_wr_addr_i(acc_wr_addr), .acc_wr_data_i(acc_wr_data), .acc_wr_be_i(acc_wr_be),
    .wf_valid_o(), .wf_ready_i(1'b0), .wf_data_o(), .if_valid_o(), .if_ready_i(1'b0), .if_data_o(),
    .ub_owner_tpu_i(1'b0), .acc_owner_tpu_i(1'b0), .wf_dbg_pop_i(1'b0), .if_dbg_pop_i(1'b0),
    .wf_flush_i(1'b0), .if_flush_i(1'b0), .err_clr_i(1'b0),
    .err_o(fab_err), .wf_level_o(), .if_level_o()
  );

  logic run = 0, busy, halted, cfg_err, irq;
  logic [7:0] cfg_err_op;
  logic [31:0] irq_tag;
  logic [31:0] perf [8];
  logic [31:0] sperf [4];

  logic wl_valid = 0, wl_ready, wl_last = 0, w_ready;
  logic [127:0] wl_data = '0;
  jepa_top #(.N(N)) i_jepa (
    .clk_i(clk), .rst_ni(rst_n), .run_i(run),
    .wload_valid_i(wl_valid), .wload_ready_o(wl_ready), .wload_data_i(wl_data), .wload_last_i(wl_last),
    .weights_ready_o(w_ready),
    .ub_rd_en_o(ub_rd_en), .ub_rd_addr_o(ub_rd_addr), .ub_rd_data_i(ub_rd_data),
    .ub_wr_en_o(ub_wr_en), .ub_wr_addr_o(ub_wr_addr), .ub_wr_data_o(ub_wr_data), .ub_wr_be_o(ub_wr_be),
    .acc_rd_en_o(acc_rd_en), .acc_rd_addr_o(acc_rd_addr), .acc_rd_data_i(acc_rd_data),
    .acc_wr_en_o(acc_wr_en), .acc_wr_addr_o(acc_wr_addr), .acc_wr_data_o(acc_wr_data), .acc_wr_be_o(acc_wr_be),
    .busy_o(busy), .halted_o(halted), .cfg_err_o(cfg_err), .cfg_err_op_o(cfg_err_op),
    .irq_o(irq), .irq_tag_o(irq_tag), .perf_o(perf), .seq_perf_o(sperf));

  int unsigned errors = 0;
  task automatic fail(input string m); $display("[ERROR] %s", m); errors++; endtask

  task automatic axi_write(input int p, input logic [63:0] a, input data_t d);
    mreq[p].aw = '0; mreq[p].aw.addr = a; mreq[p].aw.size = 3'd4; mreq[p].aw.burst = axi_pkg::BURST_INCR;
    mreq[p].aw_valid = 1;
    mreq[p].w = '0; mreq[p].w.data = d; mreq[p].w.strb = '1; mreq[p].w.last = 1; mreq[p].w_valid = 1;
    mreq[p].b_ready = 1;
    fork
      begin do @(posedge clk); while (!mrsp[p].aw_ready); #TA mreq[p].aw_valid = 0; end
      begin do @(posedge clk); while (!mrsp[p].w_ready);  #TA mreq[p].w_valid  = 0; end
    join
    while (!mrsp[p].b_valid) @(posedge clk);
    if (mrsp[p].b.resp != axi_pkg::RESP_OKAY) fail($sformatf("write %h resp %0d", a, mrsp[p].b.resp));
    @(posedge clk); #TA mreq[p].b_ready = 0;
  endtask
  task automatic axi_read(input int p, input logic [63:0] a, output data_t d);
    mreq[p].ar = '0; mreq[p].ar.addr = a; mreq[p].ar.size = 3'd4; mreq[p].ar.burst = axi_pkg::BURST_INCR;
    mreq[p].ar_valid = 1; mreq[p].r_ready = 1;
    do @(posedge clk); while (!mrsp[p].ar_ready);
    #TA mreq[p].ar_valid = 0;
    while (!mrsp[p].r_valid) @(posedge clk);
    d = mrsp[p].r.data;
    @(posedge clk); #TA mreq[p].r_ready = 0;
  endtask

  logic [127:0] ub_init [0:4095];
  logic [127:0] wimg [0:131071];      // weight image (boot partition / DDR on the board)
  logic [127:0] exp_out [0:16383];
  int m_in, m_out_base, m_out_rows, m_prog, m_wrom, m_prom, m_lrom;
  string rom;

  initial begin
    int fd, v; string key;
    longint t0, t1;
    mreq[0] = '0; mreq[1] = '0;
    if (!$value$plusargs("ROM=%s", rom)) rom = "rom_e2e";
    fd = $fopen({rom, "/meta.txt"}, "r");
    if (fd == 0) $fatal(1, "no %s/meta.txt", rom);
    while ($fscanf(fd, "%s %d", key, v) == 2)
      case (key)
        "in_rows": m_in = v; "out_base": m_out_base = v; "out_rows": m_out_rows = v;
        "prog": m_prog = v; "wrom": m_wrom = v; "prom": m_prom = v; "lrom": m_lrom = v;
        default: ;
      endcase
    $fclose(fd);
    // "bitstream" contents: the hard-coded model
    $readmemh({rom, "/prog.hex"}, i_jepa.prog_rom, 0, m_prog - 1);
    $readmemh({rom, "/wrom.hex"}, wimg,            0, m_wrom - 1);
    $readmemh({rom, "/prom.hex"}, i_jepa.prom,     0, m_prom - 1);
    $readmemh({rom, "/lrom.hex"}, i_jepa.lrom,     0, m_lrom - 1);
    $readmemh({rom, "/ub_init.hex"}, ub_init, 0, m_in - 1);
    $readmemh({rom, "/exp_out.hex"}, exp_out, 0, m_out_rows - 1);

    repeat (5) @(posedge clk); #TA rst_n = 1; repeat (50) @(posedge clk); #TA;   // drive between edges
    // the image: the host writes it into the UB over AXI (HPM0 port of tpu_fabric)
    for (int r = 0; r < m_in; r++) axi_write(1, UB_BASE + 64'(r) * 16, ub_init[r]);
    // boot: stream the weight image into the URAM, with random source gaps (DMA/flash)
    begin
      longint tb0; int gap_pct; tb0 = $time;
      if (!$value$plusargs("GAP_PCT=%d", gap_pct)) gap_pct = 10;
      for (int k = 0; k < m_wrom; k++) begin
        while ($urandom_range(99) < gap_pct) begin wl_valid = 0; @(posedge clk); #TA; end
        wl_valid = 1; wl_data = wimg[k]; wl_last = (k == m_wrom - 1);
        do @(posedge clk); while (!wl_ready);
        #TA;
      end
      wl_valid = 0; wl_last = 0;
      @(posedge clk); #TA;
      if (!w_ready) fail("weights_ready not set after boot load");
      begin int nb = 0; for (int k = 0; k < m_wrom; k++) if (i_jepa.wrom[k] !== wimg[k]) begin if (nb < 3) $display("  wrom %0d got %h exp %h", k, i_jepa.wrom[k], wimg[k]); nb++; end
        if (nb) fail($sformatf("%0d weight words wrong after boot", nb)); end
      $display("tb_jepa_top: boot load %0d words in %0d cycles (%0.2f ms @100MHz)", m_wrom, ($time - tb0) / TCK, real'($time - tb0) / 1e6);
    end
    @(posedge clk); #TA run = 1; t0 = $time;
    begin
      int c = 0;
      while (!irq && !cfg_err && c < 20_000_000) begin @(posedge clk); c++; end
      if (cfg_err) fail($sformatf("cfg_err op=%h at pc=%0d", cfg_err_op, i_jepa.pc));
      if (!irq) begin fail($sformatf("timeout: seq state %0d pc %0d", i_jepa.st, i_jepa.pc)); $finish; end
    end
    t1 = $time;
    repeat (20) @(posedge clk);
    if (irq_tag != 32'h7E9A0001) fail($sformatf("irq tag %h", irq_tag));
    begin
      int bad = 0; data_t d;
      for (int r = 0; r < m_out_rows; r++) begin
        axi_read(1, UB_BASE + 64'(m_out_base + r) * 16, d);
        if (d !== exp_out[r]) begin
          if (bad < 5) $display("  out row %0d: got %h exp %h", r, d, exp_out[r]);
          bad++;
        end
      end
      if (bad) fail($sformatf("%0d of %0d output rows wrong", bad, m_out_rows));
    end
    $display("tb_jepa_top: encoder run %0d cycles (%0.1f us @100MHz) | core: array=%0d wstall=%0d nonmx=%0d sync=%0d instr=%0d | seq: vec=%0d wub=%0d drain=%0d rom=%0d",
             (t1 - t0) / TCK, real'(t1 - t0) / 1000.0, perf[1], perf[2], perf[4], perf[5], perf[7],
             sperf[0], sperf[1], sperf[2], sperf[3]);
    if (errors == 0) $display("JEPA_E2E_PASS"); else $display("JEPA_E2E_FAIL errors=%0d", errors);
    $finish;
  end
endmodule
