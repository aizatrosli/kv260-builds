// End-to-end test of the I3C/MCTP generator against behavioural I3C Targets.
//   +OUT=<dir>   where frames/messages are dumped for the Python checker (default build)
//   +VCD         dump waves
`timescale 1ns/1ps
module tb_i3c_mctp;
  import crc8_pec_pkg::*;
  localparam logic [3:0] OP_START = 1, OP_STOP = 2, OP_ADDR = 3, OP_WRITE = 4, OP_READ = 5,
                         OP_RAWWR = 6, OP_RAWRD = 7, OP_DAA = 8, OP_WAIT = 9, OP_PATTERN = 10;
  localparam logic [7:0] F_ARB = 8'h01, F_IBIACK = 8'h02, F_NACKSTOP = 8'h04, F_LAST = 8'h01, F_PARINV = 8'h01;

  logic clk = 0, rst_n = 0;
  always #5 clk = ~clk;   // 100 MHz

  // ---------------- timing configuration (clk cycles of 10 ns), overridable by plusargs ----------------
  // Defaults meet MIPI I3C Basic v1.1.1 Tables 86/87 for a pure I3C bus at fSCL = 12.5 MHz.
  logic [7:0] t_low_pp = 4, t_high_pp = 4, t_low_od = 25, t_high_od = 4, t_high_init = 25,
              t_cas = 5, t_cbp = 3, t_pat = 4;
  initial begin
    int v;
    if ($value$plusargs("T_LOW_PP=%d", v))    t_low_pp = 8'(v);
    if ($value$plusargs("T_HIGH_PP=%d", v))   t_high_pp = 8'(v);
    if ($value$plusargs("T_LOW_OD=%d", v))    t_low_od = 8'(v);
    if ($value$plusargs("T_HIGH_OD=%d", v))   t_high_od = 8'(v);
    if ($value$plusargs("T_HIGH_INIT=%d", v)) t_high_init = 8'(v);
    if ($value$plusargs("T_CAS=%d", v))       t_cas = 8'(v);
    if ($value$plusargs("T_CBP=%d", v))       t_cbp = 8'(v);
  end

  // ---------------- DUT ----------------
  logic        hv = 0, hr; logic [31:0] hcmd = 0;
  logic        rv; logic [31:0] rsp;
  logic        mstart = 0, mbusy; logic [6:0] mda = 0; logic [7:0] mdst = 0, msrc = 0, mtu = 64;
  logic        mto = 0; logic [2:0] mtag = 0; logic [1:0] mseq = 0, mseq_next; logic [15:0] mlen = 0;
  logic [3:0]  mver = 4'h1; logic mpec_bad = 0, muse7e = 1; logic [15:0] mpkts, mnack;
  logic        msg_v = 0, msg_r; logic [7:0] msg_d = 0;
  logic        bus_free, scl, sda_oe, sda_o, sda_pu;
  logic        ibi_en = 0, ibi_ack = 1; logic [3:0] ibi_bytes = 1;
  wire         sda;

  i3c_mctp_gen dut (
    .clk_i(clk), .rst_ni(rst_n),
    .cfg_t_low_pp(t_low_pp), .cfg_t_high_pp(t_high_pp), .cfg_t_low_od(t_low_od), .cfg_t_high_od(t_high_od),
    .cfg_t_high_init(t_high_init), .cfg_t_cas(t_cas), .cfg_t_cbp(t_cbp), .cfg_t_pat(t_pat), .cfg_t_buf(16'd100), .cfg_t_stall(16'd9000),
    .cfg_ibi_en(ibi_en), .cfg_ibi_ack(ibi_ack), .cfg_ibi_bytes(ibi_bytes),
    .hcmd_valid_i(hv), .hcmd_i(hcmd), .hcmd_ready_o(hr), .rsp_valid_o(rv), .rsp_o(rsp),
    .mctp_start_i(mstart), .mctp_busy_o(mbusy), .mctp_da_i(mda), .mctp_dst_eid_i(mdst),
    .mctp_src_eid_i(msrc), .mctp_to_i(mto), .mctp_tag_i(mtag), .mctp_seq_i(mseq), .mctp_tu_i(mtu),
    .mctp_len_i(mlen), .mctp_hdr_ver_i(mver), .mctp_pec_corrupt_i(mpec_bad), .mctp_use_7e_i(muse7e),
    .mctp_seq_next_o(mseq_next), .mctp_pkts_o(mpkts), .mctp_nack_o(mnack),
    .msg_valid_i(msg_v), .msg_data_i(msg_d), .msg_ready_o(msg_r),
    .bus_free_o(bus_free),
    .scl_o(scl), .sda_oe_o(sda_oe), .sda_o(sda_o), .sda_pu_en_o(sda_pu), .sda_i(sda));

  // ---------------- bus: wired-AND with pull-up / high-keeper ----------------
  logic a_d0, a_d1, b_d0, b_d1;
  wire  c_d0 = sda_oe & ~sda_o, c_d1 = sda_oe & sda_o;
  assign sda = (c_d0 | a_d0 | b_d0) ? 1'b0 : 1'b1;   // pull-up/keeper makes released SDA high
  int contention = 0;
  always @(c_d0 or c_d1 or a_d0 or a_d1 or b_d0 or b_d1)
    if ((c_d1 && (a_d0 || b_d0)) || (c_d0 && (a_d1 || b_d1)) || (a_d1 && b_d0) || (a_d0 && b_d1)) begin
      #0.5;   // ignore zero-width overlaps inside one timestep
      if ((c_d1 && (a_d0 || b_d0)) || (c_d0 && (a_d1 || b_d1)) || (a_d1 && b_d0) || (a_d0 && b_d1)) begin
        contention++; $display("[%0t] BUS CONTENTION c=%b%b a=%b%b b=%b%b", $time, c_d0, c_d1, a_d0, a_d1, b_d0, b_d1);
      end
    end

  // PIDs chosen so that A wins ENTDAA arbitration first (lower value wins, §5.1.4.2)
  i3c_target_bfm #(.PID(48'h04D2_0000_1001), .BCR(8'h06), .DCR(8'hCC), .NAME("A")) tA (.scl, .sda, .sda_d0(a_d0), .sda_d1(a_d1));
  i3c_target_bfm #(.PID(48'h04D2_0000_2002), .BCR(8'h06), .DCR(8'hCC), .NAME("B")) tB (.scl, .sda, .sda_d0(b_d0), .sda_d1(b_d1));

  i3c_timing_mon mon (.scl, .sda, .od_i(sda_pu), .ctrl_drv_i(sda_oe));

  // ---------------- response capture ----------------
  logic [31:0] rq[$];
  always @(posedge clk) if (rv) rq.push_back(rsp);

  int errors = 0;
  task automatic check(input bit cond, input string m);
    if (!cond) begin errors++; $display("[%0t] FAIL: %s", $time, m); end
    else $display("[%0t] ok: %s", $time, m);
  endtask

  task automatic cmd(input logic [3:0] op, input logic [7:0] fl = 0, input logic [15:0] d = 0, input logic [3:0] nb = 0);
    @(negedge clk); hv = 1; hcmd = {op, fl, nb, d};
    do @(posedge clk); while (!hr);
    @(negedge clk); hv = 0;
  endtask

  task automatic wait_idle();
    repeat (5) @(posedge clk);
    while (!bus_free || mbusy) @(posedge clk);
    repeat (20) @(posedge clk);
  endtask

  function automatic logic [31:0] pop();
    if (rq.size() == 0) return 32'hDEAD_DEAD;
    return rq.pop_front();
  endfunction

  // send one MCTP message through the packet sequencer
  task automatic send_mctp(input byte unsigned msg[], input logic [6:0] da, input logic [7:0] dst, src,
                           input logic to, input logic [2:0] tag, input logic [1:0] seq, input logic [7:0] tu);
    @(negedge clk);
    mda = da; mdst = dst; msrc = src; mto = to; mtag = tag; mseq = seq; mtu = tu; mlen = msg.size();
    mstart = 1;
    do @(posedge clk); while (!mbusy);
    @(negedge clk); mstart = 0;
    for (int i = 0; i < msg.size(); i++) begin
      @(negedge clk); msg_v = 1; msg_d = msg[i];
      do @(posedge clk); while (!msg_r);
    end
    @(negedge clk); msg_v = 0;
    while (mbusy) @(posedge clk);
  endtask

  // dump helpers for the Python checker
  int fd;
  task automatic dump_bytes(input string tag, input byte unsigned b[$]);
    string s; s = tag;
    foreach (b[i]) s = {s, $sformatf(" %02x", b[i])};
    $fdisplay(fd, "%s", s);
  endtask

  string outdir;
  initial begin
    byte unsigned m1[], m2[], m3[];
    byte unsigned q[$];
    logic [31:0] r;
    if (!$value$plusargs("OUT=%s", outdir)) outdir = "build";
    if ($test$plusargs("VCD")) begin $dumpfile({outdir, "/tb.vcd"}); $dumpvars(0, tb_i3c_mctp); end
    fd = $fopen({outdir, "/sim_dump.txt"}, "w");

    // ---- PEC check value (CRC-8/SMBUS) ----
    begin
      logic [7:0] c; c = 0;
      c = 0; for (int i = 0; i < 9; i++) c = crc8_pec(c, 8'h31 + 8'(i));
      check(c == 8'hF4, $sformatf("crc8_pec(\"123456789\") = 0x%02x (expect 0xF4)", c));
    end

    repeat (5) @(posedge clk); rst_n = 1;
    wait_idle();

    // ---- 1. ENTDAA (I3C Basic §5.1.4.2) ----
    $display("--- ENTDAA");
    cmd(OP_START); cmd(OP_ADDR, F_ARB, {8'd0, 7'h7E, 1'b0}); cmd(OP_WRITE, 0, 16'h0007);
    cmd(OP_DAA, 0, 16'h000A); cmd(OP_DAA, 0, 16'h000B); cmd(OP_DAA, 0, 16'h000C); cmd(OP_STOP);
    wait_idle();
    r = pop(); check(r[31:28] == 3 && r[20] && r[7:0] == 8'hFC, $sformatf("7E/W ACKed (rsp %08x)", r));
    begin
      logic [63:0] w; w = 0;
      for (int i = 0; i < 8; i++) begin r = pop(); w = {w[55:0], r[7:0]}; end
      check(w == {48'h04D2_0000_1001, 8'h06, 8'hCC}, $sformatf("round 1 PID/BCR/DCR = %016x", w));
      r = pop(); check(r[31:28] == 8 && r[21:20] == 2'b11 && r[6:0] == 7'h0A, $sformatf("round 1 DA ACK (rsp %08x)", r));
      w = 0;
      for (int i = 0; i < 8; i++) begin r = pop(); w = {w[55:0], r[7:0]}; end
      check(w == {48'h04D2_0000_2002, 8'h06, 8'hCC}, $sformatf("round 2 PID/BCR/DCR = %016x", w));
      r = pop(); check(r[31:28] == 8 && r[21:20] == 2'b11 && r[6:0] == 7'h0B, $sformatf("round 2 DA ACK (rsp %08x)", r));
      r = pop(); check(r[31:28] == 8 && r[21:20] == 2'b00, $sformatf("round 3: no Target left (rsp %08x)", r));
    end
    check(tA.has_da && tA.da == 7'h0A && tB.has_da && tB.da == 7'h0B, "Targets hold DA 0x0A / 0x0B");

    // ---- 2. Broadcast SETMWL = 69 (DSP0233 §5.4.2 baseline, I3C §5.1.9.3.5) ----
    $display("--- SETMWL broadcast");
    cmd(OP_START); cmd(OP_ADDR, F_ARB, {8'd0, 7'h7E, 1'b0}); cmd(OP_WRITE, 0, 16'h0009);
    cmd(OP_WRITE, 0, 16'h0000); cmd(OP_WRITE, 0, 16'h0045); cmd(OP_STOP);
    wait_idle(); r = pop();
    check(tA.mwl == 16'd69 && tB.mwl == 16'd69, $sformatf("SETMWL seen by A=%0d B=%0d", tA.mwl, tB.mwl));

    // ---- 3. Direct GETSTATUS to B (I3C Table 27) ----
    $display("--- GETSTATUS direct");
    tB.status = 16'h0007;   // pending interrupt = 7 (DSP0233 §5.2.2.4 default)
    cmd(OP_START); cmd(OP_ADDR, F_ARB, {8'd0, 7'h7E, 1'b0}); cmd(OP_WRITE, 0, 16'h0090);
    cmd(OP_START); cmd(OP_ADDR, F_NACKSTOP, {8'd0, 7'h0B, 1'b1}); cmd(OP_READ); cmd(OP_READ, F_LAST); cmd(OP_STOP);
    wait_idle();
    r = pop(); r = pop(); check(r[31:28] == 3 && r[20], $sformatf("GETSTATUS addr ACK (rsp %08x)", r));
    begin
      logic [31:0] r1, r2; r1 = pop(); r2 = pop();
      check(r1[7:0] == 8'h00 && r1[20] == 1 && r2[7:0] == 8'h07 && r2[20] == 0,
            $sformatf("GETSTATUS = %02x%02x, T-bits %b%b (expect 0007, T=1 then End-of-Data 0)", r1[7:0], r2[7:0], r1[20], r2[20]));
    end

    // ---- 4. MCTP: 150-byte vendor-defined message to A, TU 64 -> 3 packets ----
    $display("--- MCTP multi-packet write");
    m1 = new[150];
    m1[0] = 8'h7F;                                    // IC=0, type 0x7F vendor defined IANA (DSP0239)
    for (int i = 1; i < 150; i++) m1[i] = 8'((i * 37 + 11) & 8'hFF);
    tA.wframes = {};
    send_mctp(m1, 7'h0A, 8'h1D, 8'h08, 1'b1, 3'd5, 2'd2, 8'd64);
    wait_idle();
    check(tA.wframes.size() == 3, $sformatf("A received %0d frames (expect 3)", tA.wframes.size()));
    check(mpkts == 3 && mnack == 0, $sformatf("sequencer: %0d ACKed, %0d NACKed", mpkts, mnack));
    check(mseq_next == 2'd1, $sformatf("seq_next = %0d (2,3,0 used -> 1)", mseq_next));
    begin q = {}; foreach (m1[i]) q.push_back(m1[i]); dump_bytes("MSG 0a 1d 08 1 5 2 64", q); end
    foreach (tA.wframes[i]) dump_bytes("WR 0a", tA.wframes[i]);
    rq = {};

    // ---- 5. MCTP control: Set Endpoint ID (0x1D) to A, then Get EID to B, direct address after S ----
    $display("--- MCTP control requests");
    m2 = '{8'h00, 8'h81, 8'h01, 8'h00, 8'h1D};       // DSP0236 §13.4 Table 14
    tA.wframes = {};
    send_mctp(m2, 7'h0A, 8'h00, 8'h08, 1'b1, 3'd0, 2'd0, 8'd64);
    muse7e = 0;
    m3 = '{8'h00, 8'h82, 8'h02};                      // Get EID, IID 2 (§13.5 Table 15)
    tB.wframes = {};
    send_mctp(m3, 7'h0B, 8'h00, 8'h08, 1'b1, 3'd1, 2'd0, 8'd64);
    muse7e = 1;
    wait_idle();
    check(tA.wframes.size() == 1 && tB.wframes.size() == 1, "single-packet control messages delivered");
    begin q = {}; foreach (m2[i]) q.push_back(m2[i]); dump_bytes("MSG 0a 00 08 1 0 0 64", q); end
    dump_bytes("WR 0a", tA.wframes[0]);
    begin q = {}; foreach (m3[i]) q.push_back(m3[i]); dump_bytes("MSG 0b 00 08 1 1 0 64", q); end
    dump_bytes("WR 0b", tB.wframes[0]);
    rq = {};

    // ---- 6. Negative: corrupted PEC and bad T-bit parity ----
    $display("--- error injection");
    tA.wframes = {};
    mpec_bad = 1; send_mctp(m3, 7'h0A, 8'h00, 8'h08, 1'b1, 3'd2, 2'd0, 8'd64); mpec_bad = 0;
    wait_idle();
    begin
      byte unsigned f[$]; logic [7:0] c; f = tA.wframes[0];
      c = crc8_pec(8'h00, {7'h0A, 1'b0});
      for (int i = 0; i < f.size() - 1; i++) c = crc8_pec(c, f[i]);
      check(c != f[f.size() - 1], "corrupted PEC is detected by a receiver");
      dump_bytes("BADPEC 0a", f);
    end
    begin
      int pe; pe = tA.parity_errors;
      cmd(OP_START); cmd(OP_ADDR, F_ARB | F_NACKSTOP, {8'd0, 7'h0A, 1'b0}); cmd(OP_WRITE, F_PARINV, 16'h0055); cmd(OP_STOP);
      wait_idle();
      check(tA.parity_errors == pe + 1, "inverted T-bit flagged by Target as parity error");
    end
    rq = {};

    // ---- 7. NACK of an absent address (DSP0233 §5.2.1.2) ----
    $display("--- NACK + flush");
    send_mctp(m3, 7'h21, 8'h00, 8'h08, 1'b1, 3'd3, 2'd0, 8'd64);
    wait_idle();
    check(mnack == 1 && mpkts == 0, $sformatf("absent target: %0d NACK, %0d ACK", mnack, mpkts));
    begin
      int nflush; nflush = 0;
      while (rq.size()) begin r = pop(); if (r[31:28] == 4'hF && r[21]) nflush++; end
      check(nflush >= 1, $sformatf("rest of packet flushed (%0d commands up to STOP)", nflush));
    end

    // ---- 8. IBI from a free bus with MDB 0xAE, then private read of the MCTP packet ----
    $display("--- IBI + MCTP read");
    begin
      byte unsigned rp[$];
      // Get EID response from B (EID 0x1E), DSP0236 §13.5 Table 15; built with the packet rules
      byte unsigned body[$]; logic [7:0] c;
      body = '{8'h01, 8'h08, 8'h1E, 8'hC1,  8'h00, 8'h02, 8'h02, 8'h00, 8'h1E, 8'h00, 8'h00};
      c = crc8_pec(8'h00, {7'h0B, 1'b1});
      foreach (body[i]) c = crc8_pec(c, body[i]);
      rp = body; rp.push_back(c);
      tB.rdq.push_back(rp);
      tB.ibi_payload = '{8'hAE};
      ibi_en = 1; ibi_ack = 1; ibi_bytes = 1;
      tB.ibi_on_idle = 1; tB.ibi_req = 1;
      wait (tB.ibi_count == 1);
      wait_idle();
      r = pop(); check(r[31:28] == 4'hB && r[20] && r[7:0] == {7'h0B, 1'b1}, $sformatf("IBI from 0x0B ACKed (rsp %08x)", r));
      r = pop(); check(r[31:28] == 5 && r[7:0] == 8'hAE && r[20] == 0, $sformatf("MDB = 0x%02x (expect 0xAE, T=0)", r[7:0]));
      // Controller reads the packet: S + addr/R (arbitrable), bytes until T=0 (DSP0233 §5.2.2.1 (b))
      cmd(OP_START); cmd(OP_ADDR, F_ARB | F_NACKSTOP, {8'd0, 7'h0B, 1'b1});
      // ask for 5 bytes more than the packet: the Target's T=0 ends it, the extra READs are flushed
      for (int i = 0; i < rp.size() + 5; i++) cmd(OP_READ, (i == rp.size() + 4) ? F_LAST : 8'h00);
      cmd(OP_STOP);
      wait_idle();
      r = pop(); check(r[31:28] == 3 && r[20], "read address ACKed");
      q = {};
      for (int i = 0; i < rp.size(); i++) begin
        r = pop(); q.push_back(r[7:0]);
        if (i == rp.size() - 1) check(r[20] == 0, "last byte carries T=0 End-of-Data");
      end
      check(q == rp, "read packet matches what the Target sent");
      begin
        int nflush; nflush = 0;
        while (rq.size()) begin r = pop(); if (r[31:28] == 4'hF && r[21] && r[3:0] == OP_READ) nflush++; end
        check(nflush == 5, $sformatf("%0d surplus READs flushed after End-of-Data (expect 5)", nflush));
      end
      dump_bytes("RD 0b", q);
      ibi_en = 0;
    end
    rq = {};

    // ---- 9. IBI that wins arbitration against a Controller header after S ----
    $display("--- IBI in arbitration");
    tA.ibi_payload = '{8'hAE};
    tA.ibi_on_idle = 0; tA.ibi_req = 1;
    cmd(OP_START); cmd(OP_ADDR, F_ARB | F_IBIACK, {8'd0, 7'h7E, 1'b0}); cmd(OP_WRITE, 0, 16'h0090); cmd(OP_STOP);
    wait_idle();
    r = pop(); check(r[31:28] == 4'hB && r[7:0] == {7'h0A, 1'b1} && r[20], $sformatf("Target 0x0A won header with IBI (rsp %08x)", r));
    r = pop(); check(r[31:28] == 5 && r[7:0] == 8'hAE, "MDB 0xAE read");
    begin
      int nflush; nflush = 0;
      while (rq.size()) begin r = pop(); if (r[31:28] == 4'hF && r[21]) nflush++; end
      check(nflush == 2, $sformatf("rest of Controller frame flushed (%0d)", nflush));
    end
    check(tA.ibi_count == 1, "A counted one accepted IBI");

    // ---- 10. HDR Exit Pattern and Target Reset Pattern ----
    $display("--- patterns");
    cmd(OP_START); cmd(OP_ADDR, F_ARB, {8'd0, 7'h7E, 1'b0}); cmd(OP_WRITE, 0, 16'h002A); cmd(OP_WRITE, 0, 16'h0001);
    cmd(OP_PATTERN, 0, 16'd14); cmd(OP_START); cmd(OP_STOP);
    cmd(OP_START); cmd(OP_ADDR, F_ARB, {8'd0, 7'h7E, 1'b0}); cmd(OP_PATTERN, 0, 16'd7); cmd(OP_STOP);
    wait_idle();
    check(mon.pat_counts.size() == 2 && mon.pat_counts[0] == 14 && mon.pat_counts[1] == 7,
          $sformatf("patterns on the wire: %p SDA toggles with SCL low (expect 14, 7)", mon.pat_counts));
    rq = {};

    // ---- 11. Host underrun: Controller Clock Stalling with SCL Low (I3C Basic §5.1.2.5, Table 10) ----
    $display("--- clock stalling");
    begin
      int pe; byte unsigned f[$]; real ml;
      tA.wframes = {};
      cmd(OP_START); cmd(OP_ADDR, F_ARB | F_NACKSTOP, {8'd0, 7'h0A, 1'b0}); cmd(OP_WRITE, 0, 16'h0011);
      #3us;                                   // host late by 3 us: stall in the write parity bit
      cmd(OP_WRITE, 0, 16'h0022); cmd(OP_STOP);
      wait_idle();
      check(tA.wframes.size() == 1 && tA.wframes[0].size() == 2 && tA.wframes[0][0] == 8'h11 && tA.wframes[0][1] == 8'h22,
            "stalled frame delivered intact");
      check(mon.max_low >= 2000.0, $sformatf("host gap absorbed as one SCL Low stall of %.0f ns", mon.max_low));
      rq = {};
      // host late by 150 us: stall limit (90 us) hit, frame aborted with P, late commands flushed
      tA.wframes = {};
      cmd(OP_START); cmd(OP_ADDR, F_ARB | F_NACKSTOP, {8'd0, 7'h0A, 1'b0}); cmd(OP_WRITE, 0, 16'h0033);
      #150us;
      cmd(OP_WRITE, 0, 16'h0044); cmd(OP_STOP);
      wait_idle();
      r = pop(); r = pop();
      check(r[31:28] == 4'hF && r[22], $sformatf("stall limit reported (rsp %08x)", r));
      check(tA.wframes.size() == 1 && tA.wframes[0].size() == 1, "aborted frame ended after the stalled byte");
      rq = {};
    end

    // ---- summary ----
    mon.report();
    check(mon.errors == 0, "no timing violations");
    check(contention == 0, "no bus contention");
    check(tA.parity_errors == 1 && tB.parity_errors == 0, "only the injected parity error was seen");
    $fclose(fd);
    if (errors == 0) $display("tb_i3c_mctp: ALL PASS");
    else $display("tb_i3c_mctp: %0d FAILURES", errors);
    $finish;
  end

  initial begin #20ms; $display("tb_i3c_mctp: TIMEOUT"); $finish; end
endmodule
