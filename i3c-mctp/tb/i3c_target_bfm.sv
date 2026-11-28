// Behavioural I3C Basic SDR Target, for simulation only (event driven, needs --timing).
// Written from MIPI I3C Basic v1.1.1 (Public Release Edition), independent of the RTL Controller.
//
// Supports: S/Sr/P detection; 7'h7E broadcast ACK; private write/read to its Dynamic Address;
// T-bit odd-parity check on writes (§5.1.2.3.3); End-of-Data T-bit on reads (§5.1.2.3.4);
// ENTDAA with 64-bit PID+BCR+DCR arbitration and PAR check (§5.1.4.2); RSTDAA; broadcast SETMWL/
// SETMRL capture; direct GETSTATUS (§5.1.9.3.15, Table 27); IBI with MDB (§5.1.6.2), both from a
// free bus (Target pulls SDA low) and by arbitrating in a Controller's header after S.
//
// Drive: sda_d0 = pull SDA low, sda_d1 = drive SDA high (Push-Pull). Both 0 = High-Z.
`timescale 1ns/1ps
module i3c_target_bfm #(
  parameter logic [47:0] PID = 48'h0123_4567_89AB,
  parameter logic [7:0]  BCR = 8'h06,     // IBI capable (bit1), IBI payload/MDB (bit2) (Table 5)
  parameter logic [7:0]  DCR = 8'hCC,     // MCTP endpoint (DSP0233 §5.4.1)
  parameter string       NAME = "tgt",
  parameter real         T_SCO = 5.0      // clock-to-data after SCL fall (<= 12 ns, Table 87)
) (
  input  logic scl,
  input  logic sda,
  output logic sda_d0,
  output logic sda_d1
);
  // ---------------- control / observation from the testbench ----------------
  logic [6:0]  da = 7'h00;        // Dynamic Address (0 = none)
  logic        has_da = 1'b0;
  bit          nack_writes = 0;   // emulate full buffer (DSP0233 §5.2.1.2)
  // read data queue: each entry is one private-read transfer
  byte unsigned rdq[$][$];
  // IBI request
  bit          ibi_req = 0;       // request an IBI
  bit          ibi_on_idle = 1;   // 1: pull SDA low on a free bus; 0: only compete after a Controller S
  byte unsigned ibi_payload[$];   // MDB first
  int          ibi_count = 0;     // IBIs accepted by the Controller
  int          ibi_nacked = 0;
  // received frames: each private write transfer (bytes after the address)
  byte unsigned wframes[$][$];
  int          parity_errors = 0;
  int          daa_par_errors = 0;
  logic [15:0] mwl = 16'd0, mrl = 16'd0;
  logic [15:0] status = 16'h0000;   // GETSTATUS reply
  int          errors = 0;

  // ---------------- bus state ----------------
  typedef enum {IDLE, HDR, ACKSLOT, WDATA, RDATA, DAA_PID, DAA_DA, DAA_ACK, IBI_ACK, IBI_DATA, IGNORE} st_e;
  st_e st = IDLE;
  int          bitn = 0;
  logic [8:0]  sh = '0;
  logic        after_sr = 1'b0;
  logic        in_frame = 1'b0;       // between S and P
  logic        bcast_ccc = 1'b0;      // current message is a 7'h7E write
  logic        ccc_seen = 1'b0;
  logic [7:0]  ccc = 8'h00;           // CCC code of this frame (0 = none)
  int          ccc_bytes = 0;
  logic        daa_armed = 1'b0;      // ENTDAA received in this frame
  logic        daa_lost = 1'b0;
  logic [63:0] daa_word;
  logic        competing = 1'b0;      // arbitrating for an IBI in this header
  logic        lost = 1'b0;
  logic [7:0]  hdr;
  byte unsigned cur_w[$];
  byte unsigned cur_r[$];
  int          ridx = 0;
  int          ibi_idx = 0;
  logic        get_status = 1'b0;     // direct GETSTATUS addressed to us
  logic        direct_ccc = 1'b0;     // a direct CCC is active in this frame
  logic        drive_ack = 1'b0;
  logic        idle_pull = 1'b0;      // pulled SDA low on a free bus

  initial begin sda_d0 = 1'b0; sda_d1 = 1'b0; end

  task automatic rel();   sda_d0 = 1'b0; sda_d1 = 1'b0; endtask
  task automatic d0();    sda_d0 = 1'b1; sda_d1 = 1'b0; endtask
  task automatic d1();    sda_d0 = 1'b0; sda_d1 = 1'b1; endtask
  task automatic od(input logic b); if (b) rel(); else d0(); endtask

  function automatic logic par_odd(input logic [7:0] b); return ~^b; endfunction

  task automatic end_wframe();
    if (st == WDATA && !bcast_ccc && !direct_ccc && cur_w.size() > 0) wframes.push_back(cur_w);
    cur_w = {};
  endtask

  // START / Repeated START: SDA falls while SCL is high
  always @(negedge sda) if (scl === 1'b1) begin
    if (st == IDLE && !in_frame) after_sr = 1'b0; else after_sr = 1'b1;
    end_wframe();
    if (!in_frame) begin ccc = 8'h00; ccc_seen = 1'b0; daa_armed = 1'b0; direct_ccc = 1'b0; end
    in_frame = 1'b1;
    st = HDR; bitn = 0; sh = '0; lost = 1'b0;
    competing = !after_sr && ibi_req && has_da && (idle_pull || !ibi_on_idle);
    if (!idle_pull) rel();
    idle_pull = 1'b0;
  end

  // STOP: SDA rises while SCL is high
  always @(posedge sda) if (scl === 1'b1) begin
    end_wframe();
    st = IDLE; in_frame = 1'b0; rel();
    daa_armed = 1'b0; ccc = 8'h00; ccc_seen = 1'b0; direct_ccc = 1'b0;
  end

  // Free-bus IBI request: after Bus Available (tAVAL >= 1 us, Table 86) pull SDA low (§5.1.2.2).
  realtime last_act = 0;
  always @(scl or sda) last_act = $realtime;
  initial forever begin
    #50;
    if (ibi_req && ibi_on_idle && has_da && !in_frame && !idle_pull && scl === 1'b1 && sda === 1'b1 &&
        ($realtime - last_act) >= 1000.0) begin
      idle_pull = 1'b1; d0();
    end
  end

  // ---------------- SCL falling edge: put the next bit on SDA ----------------
  always @(negedge scl) begin
    #(T_SCO);
    unique case (st)
      HDR: begin
        if (competing && !lost) begin
          logic [7:0] mine; mine = {da, 1'b1};
          od(mine[7 - bitn]);
        end else if (!(idle_pull)) rel();
      end
      ACKSLOT: begin
        if (drive_ack) d0(); else rel();
      end
      WDATA: rel();
      RDATA: begin
        if (bitn < 8) begin
          logic [7:0] b; b = cur_r[ridx];
          if (b[7 - bitn]) d1(); else d0();
        end else begin
          // T-bit: 0 = End-of-Data, 1 = more data (§5.1.2.3.4)
          if (ridx + 1 < cur_r.size()) d1(); else d0();
        end
      end
      DAA_PID: begin
        if (!daa_lost) od(daa_word[63 - bitn]); else rel();
      end
      DAA_DA:  rel();
      DAA_ACK: if (drive_ack) d0(); else rel();
      IBI_DATA: begin
        if (bitn < 8) begin
          logic [7:0] b; b = ibi_payload[ibi_idx];
          if (b[7 - bitn]) d1(); else d0();
        end else begin
          if (ibi_idx + 1 < ibi_payload.size()) d1(); else d0();
        end
      end
      IBI_ACK: rel();
      default: rel();
    endcase
  end

  // ---------------- SCL rising edge: sample ----------------
  always @(posedge scl) begin
    logic s; s = sda;
    unique case (st)
      HDR: begin
        if (competing && !lost) begin
          logic [7:0] mine; mine = {da, 1'b1};
          if (mine[7 - bitn] && !s) lost = 1'b1;             // §5.1.2.2.1
        end
        sh = {sh[7:0], s}; bitn++;
        if (bitn == 8) begin
          hdr = sh[7:0];
          decide_ack();
          bitn = 0;
        end
      end
      ACKSLOT: begin
        logic acked; acked = !s;
        // write ACK: release as soon as SCL rises (§5.1.2.3.1). Read ACK: keep SDA low until SCL falls
        // (the Target then drives data), otherwise the release would look like a STOP.
        if (competing && !lost && hdr == {da, 1'b1}) begin
          // we won arbitration with an IBI; the Controller decides in this slot
          if (acked && ibi_payload.size() > 0 && BCR[2]) begin
            #1 d0();                                       // §5.1.2.3.2: Target drives Low after SCL rises
            ibi_count++; st = IBI_DATA; bitn = 0; ibi_idx = 0;
          end else if (acked) begin ibi_count++; ibi_req = 0; st = IGNORE; end
          else begin ibi_nacked++; st = IGNORE; end
          competing = 1'b0;
        end else if (!drive_ack) begin
          st = IGNORE;
        end else if (hdr[0] == 1'b0) begin
          #1 rel();
          st = WDATA; bitn = 0; cur_w = {};
        end else begin
          // read: hold low through SCL high
          if (daa_armed && hdr == {7'h7E, 1'b1}) begin
            st = DAA_PID; bitn = 0; daa_lost = 1'b0; daa_word = {PID, BCR, DCR};
          end else begin
            st = RDATA; bitn = 0; ridx = 0;
          end
        end
      end
      WDATA: begin
        sh = {sh[7:0], s}; bitn++;
        if (bitn == 9) begin
          logic [7:0] b; b = sh[8:1];
          if (sh[0] != par_odd(b)) begin
            parity_errors++;
            $display("[%0t] %s: T-bit parity error on 0x%02x", $time, NAME, b);
          end
          if (bcast_ccc && !ccc_seen) begin
            ccc = b; ccc_seen = 1'b1; ccc_bytes = 0;
            if (b == 8'h07) daa_armed = 1'b1;                               // ENTDAA
            if (b == 8'h06) begin has_da = 1'b0; da = 7'h00; end            // RSTDAA
            if (b[7]) direct_ccc = 1'b1;
          end else if (bcast_ccc) begin
            if (ccc == 8'h09) begin mwl = (ccc_bytes == 0) ? {b, 8'h00} : {mwl[15:8], b}; end  // SETMWL
            if (ccc == 8'h0A) begin
              if (ccc_bytes == 0) mrl = {b, 8'h00}; else if (ccc_bytes == 1) mrl = {mrl[15:8], b};
            end
            ccc_bytes++;
          end else begin
            cur_w.push_back(b);
          end
          bitn = 0;
        end
      end
      RDATA: begin
        if (bitn < 8) bitn++;
        else begin
          // T-bit just went out. T=1 and the Controller pulled SDA low after SCL rose: that is an Sr
          // (caught by the negedge sda block). T=0: release after SCL rises, Controller holds SDA low.
          #1 rel();
          if (ridx + 1 < cur_r.size()) begin ridx++; bitn = 0; end
          else st = IGNORE;
        end
      end
      DAA_PID: begin
        if (!daa_lost && daa_word[63 - bitn] && !s) daa_lost = 1'b1;
        bitn++;
        if (bitn == 64) begin
          if (daa_lost) st = IGNORE; else begin st = DAA_DA; bitn = 0; sh = '0; end
        end
      end
      DAA_DA: begin
        sh = {sh[7:0], s}; bitn++;
        if (bitn == 8) begin
          logic [6:0] a; a = sh[7:1];
          if (sh[0] != ~^a) begin
            daa_par_errors++; drive_ack = 1'b0;
            $display("[%0t] %s: ENTDAA PAR error, addr 0x%02x", $time, NAME, a);
          end else begin
            drive_ack = 1'b1; da = a; has_da = 1'b1;
            $display("[%0t] %s: ENTDAA assigned DA 0x%02x", $time, NAME, a);
          end
          st = DAA_ACK; bitn = 0;
        end
      end
      DAA_ACK: begin
        st = IGNORE;   // keep the ACK low until SCL falls; next is Sr
      end
      IBI_DATA: begin
        if (bitn < 8) bitn++;
        else begin
          #1 rel();
          if (ibi_idx + 1 < ibi_payload.size()) begin ibi_idx++; bitn = 0; end
          else begin st = IGNORE; ibi_req = 0; end
        end
      end
      default: ;
    endcase
  end

  // ACK decision for the header just received (called at the 8th SCL rise).
  task automatic decide_ack();
    logic [6:0] a; logic rnw;
    a = hdr[7:1]; rnw = hdr[0];
    drive_ack = 1'b0;
    bcast_ccc = 1'b0;
    if (competing && !lost && hdr == {da, 1'b1}) begin
      st = ACKSLOT;  // our IBI won; the Controller ACKs or NACKs
      return;
    end
    if (a == 7'h7E && !rnw) begin
      drive_ack = 1'b1; bcast_ccc = 1'b1; ccc_seen = 1'b0;
    end else if (a == 7'h7E && rnw) begin
      drive_ack = daa_armed && !has_da;
    end else if (has_da && a == da && !rnw) begin
      drive_ack = !nack_writes;
    end else if (has_da && a == da && rnw) begin
      if (direct_ccc && ccc == 8'h90) begin
        drive_ack = 1'b1; cur_r = {}; cur_r.push_back(status[15:8]); cur_r.push_back(status[7:0]);
      end else if (rdq.size() > 0) begin
        drive_ack = 1'b1; cur_r = rdq.pop_front();
      end
    end
    st = ACKSLOT;
  endtask

  // Keep the 7'h7E CCC context across Sr: once a frame's CCC byte was seen, a later 7'h7E/W after
  // Sr starts a new CCC (§5.1.9.2 Figs 30-31), so reset ccc_seen on each 7'h7E/W header.
endmodule
