// I3C SDR Controller signal engine (command driven).
//
// Generates SCL/SDA waveforms for I3C Basic SDR mode. All references are to
// "MIPI Alliance Specification for I3C Basic, v1.1.1 (9 June 2021), Public Release Edition".
//
// Pins (connect through an IOBUF on the FPGA, see kv260_i3c_top.sv):
//   scl_o              SCL is always driven push-pull by the Controller (Table 82 Note 6).
//   sda_oe_o/sda_o     SDA driver. oe=0 is High-Z (open-drain '1' or hand-over to a Target).
//   sda_i              SDA pad input (asynchronous; synchronised here with 2 flops).
//   sda_pu_en_o        1 while the bus is in an Open-Drain phase or idle. Drive a switched pull-up
//                      with it: the Controller shall provide an active OD pull-up during OD phases
//                      and the pull-up shall be off during Push-Pull (§5.1.3.1; Table 82 Notes 4, 7).
//
// Timing: every bit is SCL low for t_low cycles then SCL high for t_high cycles. SDA changes one clock
// after the SCL falling edge (Controller tHD_PP, Table 87) and is sampled on the last low cycle, i.e.
// on the SCL rising edge (Target data is valid tSCO <= 12 ns after SCL fall, Table 87; arbitration
// samples on SCL rise, §5.1.2.2.1). The sample is taken from the synchronised input, so it reflects
// the pad 2 clocks earlier. All periods are runtime registers (cfg_*) in clk_i cycles; defaults that
// meet Tables 86/87 at 100 MHz are given in i3c_mctp_regs.sv and checked by tb/i3c_timing_mon.sv.
//
// Commands (cmd_i, 32 bit): [31:28] op, [27:20] flags, [19:16] nbits, [15:0] data
//   OP_START    S when the bus is free, Sr otherwise (§5.1.2).
//   OP_STOP     P, then bus free for cfg_t_buf (§5.1.3.2.1).
//   OP_ADDR     data[7:1]=address, data[0]=RnW. flags: F_ARB = arbitrable header after S, all bits
//               Open Drain (§5.1.2.2, §5.1.2.2.1); otherwise Push-Pull header after Sr with an
//               Open-Drain ACK (§5.1.2.2.4). F_IBIACK = ACK an In-Band Interrupt won in arbitration
//               (§5.1.6.2). F_NACKSTOP = on NACK, emit P and discard host commands up to the next STOP.
//               The ninth bit is the ACK. A Target ACK of a write is handed over as in §5.1.2.3.1.
//               The first arbitrable header after reset uses cfg_t_high_init (tHIGH_INIT, Table 86).
//   OP_WRITE    8 data bits MSB first + T-bit = odd parity, T = XOR(Data[7:0], 1) (§5.1.2.3.3).
//               F_PARINV inverts the T-bit (negative testing).
//   OP_READ     8 bits from the Target + T-bit End-of-Data handling (§5.1.2.3.4): T=0 means the
//               Target ended; the Controller drives SDA low after SCL rises. T=1 means more data; with
//               F_LAST the Controller aborts by pulling SDA low while SCL is high (an Sr); without
//               F_LAST it continues. Either way the next command must be STOP or START when T=0/LAST.
//   OP_RAWWR    nbits (1..8) of data[7:0] MSB first. F_OD selects Open-Drain drive and timing.
//   OP_RAWRD    nbits (1..8) with SDA released, Open-Drain timing.
//   OP_DAA      One ENTDAA round (§5.1.4.2): Sr, 7'h7E/R (Push-Pull header, OD ACK); if ACKed read 64
//               bits PID+BCR+DCR in Open Drain, send {DA[6:0], PAR} with PAR = ~XOR(DA) in Open Drain,
//               read the ACK. data[6:0] = DA. Once a round sees NACK on 7'h7E/R, later rounds are
//               no-ops until the next STOP.
//   OP_WAIT     hold the bus as it is for data[15:0] cycles.
//   OP_PATTERN  pull SCL low, drive SDA high, then toggle SDA data[7:0] times every cfg_t_pat cycles
//               with SCL low. 7 toggles = HDR Exit Pattern (4 SDA falls, §5.2.1.1.1, follow with STOP);
//               14 toggles = Target Reset Pattern (§5.1.11.3, follow with START then STOP).
//
// Responses (rsp_o, 32 bit): [31:28] kind, [27:20] flags, [15:0] data
//   RSP_ADDR    flags[0]=ACK seen, flags[1]=arbitration lost, data[7:0] = header seen on the wire.
//   RSP_IBI     IBI header won by a Target in arbitration; flags[0]=Controller ACKed it.
//   RSP_READ    flags[0]=T-bit, data[7:0] = byte (also used for IBI payload / MDB bytes).
//   RSP_RAWRD   data[7:0] = bits read, right aligned.
//   RSP_DAA     flags[0]=7'h7E/R ACKed, flags[1]=DA ACKed. Followed (if ACKed) by 8 RSP_RAWRD bytes
//               of PID[47:0], BCR, DCR, pushed before this response.
//   RSP_ERR     flags[2]=clock-stall limit reached, frame aborted with P (data = op).
//               flags[1]=flushed host command (data = its op). Host commands are flushed up to the
//               next STOP after: arbitration lost to an IBI, NACK with F_NACKSTOP, or a Target ending a
//               read with T=0 before the READ marked F_LAST.
//
// Not implemented (see docs/README.md): HDR modes, Controller-role hand-off, Hot-Join acceptance,
// the optional A6 Push-Pull header shortcut (§5.1.2.2.2), legacy I2C (Fm/Fm+) transfers.
`timescale 1ns/1ps
module i3c_sdr_engine (
  input  logic        clk_i,
  input  logic        rst_ni,
  // configuration (cycles of clk_i)
  input  logic [7:0]  cfg_t_low_pp,
  input  logic [7:0]  cfg_t_high_pp,
  input  logic [7:0]  cfg_t_low_od,
  input  logic [7:0]  cfg_t_high_od,
  input  logic [7:0]  cfg_t_high_init,
  input  logic [7:0]  cfg_t_cas,      // S/Sr to first SCL fall (tCAS, tCASr)
  input  logic [7:0]  cfg_t_cbp,      // SCL rise to P or Sr (tCBP, tCBSr)
  input  logic [15:0] cfg_t_buf,      // bus free after P
  input  logic [7:0]  cfg_t_pat,      // spacing of pattern edges (>= tDIG_H)
  input  logic [15:0] cfg_t_stall,    // max Controller clock stall (SCL low, 9th bit), <= 100 us (Table 10)
  input  logic        cfg_ibi_en,     // react to a Target START while the bus is free (§5.1.2.2)
  input  logic        cfg_ibi_ack,    // ACK IBIs detected while the bus is free
  input  logic [3:0]  cfg_ibi_bytes,  // bytes to read after an ACKed IBI (MDB + payload), 0 = none
  // command stream
  input  logic        cmd_valid_i,
  input  logic [31:0] cmd_i,
  output logic        cmd_ready_o,
  // response stream (no back-pressure: overflow is flagged)
  output logic        rsp_valid_o,
  output logic [31:0] rsp_o,
  // status
  output logic        bus_free_o,
  output logic        busy_o,
  // pins
  output logic        scl_o,
  output logic        sda_oe_o,
  output logic        sda_o,
  input  logic        sda_i,
  output logic        sda_pu_en_o
);
  localparam logic [3:0] OP_NOP = 4'd0, OP_START = 4'd1, OP_STOP = 4'd2, OP_ADDR = 4'd3,
                         OP_WRITE = 4'd4, OP_READ = 4'd5, OP_RAWWR = 4'd6, OP_RAWRD = 4'd7,
                         OP_DAA = 4'd8, OP_WAIT = 4'd9, OP_PATTERN = 4'd10;
  localparam logic [3:0] RSP_ADDR = 4'd3, RSP_READ = 4'd5, RSP_RAWRD = 4'd7, RSP_DAA = 4'd8,
                         RSP_IBI = 4'd11, RSP_ERR = 4'd15;
  // flags
  localparam int F_ARB = 0, F_IBIACK = 1, F_NACKSTOP = 2;   // OP_ADDR
  localparam int F_PARINV = 0;                               // OP_WRITE, OP_DAA
  localparam int F_LAST = 0;                                 // OP_READ
  localparam int F_OD = 0;                                   // OP_RAWWR

  // SDA drive codes
  localparam logic [1:0] D_REL = 2'd0, D_0 = 2'd1, D_1 = 2'd2;

  // ---------------- input synchroniser ----------------
  logic [1:0] sda_sync;
  always_ff @(posedge clk_i or negedge rst_ni)
    if (!rst_ni) sda_sync <= 2'b11; else sda_sync <= {sda_sync[0], sda_i};
  wire sda_s = sda_sync[1];

  // ---------------- state ----------------
  typedef enum logic [3:0] {
    S_RESET,     // bus free wait after reset
    S_FREE,      // bus free (SCL=1, SDA released)
    S_FETCH,     // bus held, waiting for the next command
    S_SEQ,       // multi-step line sequence (START/STOP/PATTERN/WAIT)
    S_BIT_START, // first SCL fall of a bit group
    S_BIT_LO,
    S_BIT_HI,
    S_BYTE_DONE  // decide what follows a completed bit group
  } st_e;
  st_e st;

  // current operation
  logic [3:0]  op;
  logic [7:0]  fl;
  logic [15:0] dat;
  logic [6:0]  daa_da;     // ENTDAA: address to assign this round
  logic        daa_pinv;   // ENTDAA: invert PAR (negative testing)
  logic [3:0]  nbits;
  logic        host_op;      // op came from the host (vs. internally generated)

  // line state
  logic        scl_q, sda_oe_q, sda_o_q, pu_q;
  logic        in_start;     // an S/Sr was just created (SDA fell with SCL high)
  logic        init_done;    // first arbitrable header after reset done (tHIGH_INIT)
  logic        flush;        // discard host commands up to the next STOP
  logic        daa_end;      // ENTDAA: 7'h7E/R was NACKed
  logic        bus_act;      // between S and P

  // timer
  logic [15:0] cnt;
  logic [3:0]  step;
  logic [15:0] stall_cnt;    // cycles spent stalling SCL low in the current bit
  logic        stall_to;     // stall limit reached: abort the frame

  // bit-group context
  logic [3:0]  bi;           // bit index within group (0 = first transmitted)
  logic [3:0]  blen;         // bits in group (9 for ADDR/WRITE/READ)
  logic        b_od;         // Open-Drain timing
  logic        b_init;       // use tHIGH_INIT
  wire         bit_od = b_od || (op == OP_ADDR && bi == 4'd8);  // ACK/NACK is always Open Drain
  logic [8:0]  rx;           // bits seen on the wire
  logic        arb_lost;
  logic        samp;
  logic [1:0]  d_hi, d_end;  // drive for the SCL-high phase and its last cycle

  // ENTDAA / IBI sub-sequencing
  typedef enum logic [3:0] {
    X_NONE, X_DAA_SR, X_DAA_HDR, X_DAA_PID, X_DAA_DA, X_DAA_ACK,
    X_IBI_START, X_IBI_HDR, X_IBI_READ, X_IBI_STOP, X_ABORT_STOP
  } sub_e;
  sub_e        sub;
  // stall in the 9th bit of a host byte (or the ENTDAA ACK) while the next host command is missing
  wire         want_stall = (cfg_t_stall != 16'd0) && !cmd_valid_i && (
                   (host_op && sub == X_NONE && blen == 4'd9 && bi == 4'd8 && !flush &&
                    !(op == OP_ADDR && arb_lost)) ||
                   (sub == X_DAA_ACK));
  logic [3:0]  subn;         // byte counter in sub-sequences
  logic        daa_hdr_ack;
  logic        ibi_acked;
  logic        ibi_auto;     // IBI handling came from a bus-free Target START

  // responses
  logic        rsp_v;
  logic [31:0] rsp_d;
  assign rsp_valid_o = rsp_v;
  assign rsp_o       = rsp_d;

  assign scl_o       = scl_q;
  assign sda_oe_o    = sda_oe_q;
  assign sda_o       = sda_o_q;
  assign sda_pu_en_o = pu_q;
  assign bus_free_o  = (st == S_FREE);
  assign busy_o      = !(st == S_FREE || st == S_FETCH) || (sub != X_NONE);

  // Accept a host command only when nothing internal is pending.
  assign cmd_ready_o = (st == S_FREE || st == S_FETCH) && (sub == X_NONE) && !rsp_hold;
  logic rsp_hold;  // unused hook for back-pressure (kept 0)
  assign rsp_hold = 1'b0;

  wire [3:0] c_op    = cmd_i[31:28];
  wire [7:0] c_fl    = cmd_i[27:20];
  wire [3:0] c_nb    = cmd_i[19:16];
  wire [15:0] c_dat  = cmd_i[15:0];

  function automatic logic [7:0] t_low (input logic od); return od ? cfg_t_low_od  : cfg_t_low_pp;  endfunction
  function automatic logic [7:0] t_high(input logic od, input logic ini);
    return ini ? cfg_t_high_init : (od ? cfg_t_high_od : cfg_t_high_pp);
  endfunction

  task automatic drive(input logic [1:0] d);
    case (d)
      D_0:     begin sda_oe_q <= 1'b1; sda_o_q <= 1'b0; end
      D_1:     begin sda_oe_q <= 1'b1; sda_o_q <= 1'b1; end
      default: begin sda_oe_q <= 1'b0; sda_o_q <= 1'b1; end
    endcase
  endtask

  task automatic push_rsp(input logic [3:0] k, input logic [7:0] f, input logic [15:0] d);
    rsp_v <= 1'b1; rsp_d <= {k, f, 4'h0, d};
  endtask

  // Start a bit group. Lines: SCL is high on entry (or low after OP_PATTERN; then the first SCL low
  // phase just continues).
  task automatic begin_group(input logic [3:0] n, input logic od, input logic ini);
    blen <= n; bi <= 4'd0; b_od <= od; b_init <= ini; rx <= '0; arb_lost <= 1'b0;
    in_start <= 1'b0;
    st <= S_BIT_START; cnt <= '0;
  endtask

  // --- drive for bit `bi` during SCL low, per operation ---
  function automatic logic [1:0] lo_drive(input logic [3:0] i);
    logic [1:0] d;
    d = D_REL;
    unique case (op)
      OP_ADDR: begin
        if (i < 4'd8) begin
          logic b; b = dat[7 - i];
          if (fl[F_ARB]) d = (arb_lost || b) ? D_REL : D_0;     // arbitrable: Open Drain (§5.1.2.2.1)
          else           d = b ? D_1 : D_0;                      // after Sr: Push-Pull (§5.1.2.2.4)
        end else begin
          // ACK slot: Controller ACKs an IBI it won't refuse (§5.1.6.2), otherwise released.
          d = (arb_lost && rx[0] && ibi_ack_now()) ? D_0 : D_REL;  // rx[0] = Target's RnW
        end
      end
      OP_WRITE: d = (i < 4'd8) ? (dat[7 - i] ? D_1 : D_0)
                               : (((~^dat[7:0]) ^ fl[F_PARINV]) ? D_1 : D_0);  // T = XOR(D,1)
      OP_RAWWR: begin
        logic b; b = dat[7 - i];
        d = fl[F_OD] ? (b ? D_REL : D_0) : (b ? D_1 : D_0);
      end
      default: d = D_REL;                                         // READ, RAWRD, DAA reads
    endcase
    return d;
  endfunction

  // SCL falls for bit i. A released SDA is released right at the SCL fall so the Target can drive
  // (ACK, read data) without overlap; a driven value changes one clock later (Controller hold time).
  task automatic fall_to(input logic [3:0] i);
    scl_q <= 1'b0; st <= S_BIT_LO; cnt <= '0;
    pu_q  <= b_od || (op == OP_ADDR && i == 4'd8);
    if (lo_drive(i) == D_REL) drive(D_REL);
  endtask

  // IBI ACK decision (for an arbitrable header lost to a Target with RnW=1).
  function automatic logic ibi_ack_now();
    return ibi_auto ? cfg_ibi_ack : fl[F_IBIACK];
  endfunction

  // ---------------- main FSM ----------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      st <= S_RESET; op <= OP_NOP; fl <= '0; dat <= '0; daa_da <= '0; daa_pinv <= 1'b0; nbits <= '0; host_op <= 1'b0;
      scl_q <= 1'b1; sda_oe_q <= 1'b0; sda_o_q <= 1'b1; pu_q <= 1'b1;
      in_start <= 1'b0; bus_act <= 1'b0; init_done <= 1'b0; flush <= 1'b0; daa_end <= 1'b0;
      cnt <= '0; step <= '0; stall_cnt <= '0; stall_to <= 1'b0; bi <= '0; blen <= '0; b_od <= 1'b0; b_init <= 1'b0; rx <= '0;
      arb_lost <= 1'b0; samp <= 1'b1; d_hi <= D_REL; d_end <= D_REL;
      sub <= X_NONE; subn <= '0; daa_hdr_ack <= 1'b0; ibi_acked <= 1'b0; ibi_auto <= 1'b0;
      rsp_v <= 1'b0; rsp_d <= '0;
    end else begin
      rsp_v <= 1'b0;
      unique case (st)
        // ---------------------------------------------------------------
        S_RESET: begin
          cnt <= cnt + 16'd1;
          if (cnt >= cfg_t_buf) begin st <= S_FREE; cnt <= '0; end
        end

        S_FREE, S_FETCH: begin
          if (sub != X_NONE) begin
            // internally generated operation
            host_op <= 1'b0;
            unique case (sub)
              X_DAA_SR:  begin op <= OP_START; step <= '0; cnt <= '0; st <= S_SEQ; end
              X_DAA_HDR: begin op <= OP_ADDR; fl <= '0; dat <= {8'd0, 7'h7E, 1'b1};
                               begin_group(4'd9, 1'b0, 1'b0); end
              X_DAA_PID: begin op <= OP_RAWRD; fl <= '0; begin_group(4'd8, 1'b1, 1'b0); end
              X_DAA_DA:  begin op <= OP_RAWWR; fl <= 8'h01;
                               dat <= {8'd0, daa_da, (~^daa_da) ^ daa_pinv};  // PAR = ~XOR(DA) (§5.1.4.2)
                               begin_group(4'd8, 1'b1, 1'b0); end
              X_DAA_ACK: begin op <= OP_RAWRD; fl <= '0; begin_group(4'd1, 1'b1, 1'b0); end
              X_IBI_START: begin op <= OP_START; step <= '0; cnt <= '0; st <= S_SEQ; end
              X_IBI_HDR: begin op <= OP_ADDR; fl <= 8'h01; dat <= {8'd0, 7'h7E, 1'b0};
                               begin_group(4'd9, 1'b1, !init_done); init_done <= 1'b1; end
              X_IBI_READ: begin op <= OP_READ; fl <= {7'd0, (subn == 4'd1)};
                               begin_group(4'd9, 1'b0, 1'b0); end
              X_IBI_STOP, X_ABORT_STOP: begin op <= OP_STOP; step <= '0; cnt <= '0; st <= S_SEQ; end
              default: sub <= X_NONE;
            endcase
          end else if (cmd_valid_i) begin
            host_op <= 1'b1;
            op <= c_op; fl <= c_fl; dat <= c_dat; nbits <= c_nb;
            if (flush) begin
              // discard host commands of an aborted transfer, up to and including its STOP
              // A STOP ends the flush. If the bus is still held (Target ended a read early with T=0),
              // the STOP is executed; if the engine already released the bus, it is dropped.
              if (c_op == OP_STOP) begin
                flush <= 1'b0;
                if (st == S_FETCH) begin step <= '0; cnt <= '0; st <= S_SEQ; end
                else push_rsp(RSP_ERR, 8'h02, {12'd0, c_op});
              end else push_rsp(RSP_ERR, 8'h02, {12'd0, c_op});
            end else begin
              unique case (c_op)
                OP_START, OP_STOP, OP_WAIT, OP_PATTERN: begin step <= '0; cnt <= '0; st <= S_SEQ; end
                OP_ADDR: begin
                  begin_group(4'd9, c_fl[F_ARB], c_fl[F_ARB] && !init_done);
                  if (c_fl[F_ARB]) init_done <= 1'b1;
                end
                OP_WRITE, OP_READ: begin_group(4'd9, 1'b0, 1'b0);
                OP_RAWWR: begin_group((c_nb == 0 || c_nb > 8) ? 4'd8 : c_nb, c_fl[F_OD], 1'b0);
                OP_RAWRD: begin_group((c_nb == 0 || c_nb > 8) ? 4'd8 : c_nb, 1'b1, 1'b0);
                OP_DAA: begin
                  if (daa_end) push_rsp(RSP_DAA, 8'h00, 16'd0);
                  else begin sub <= X_DAA_SR; subn <= '0; daa_da <= c_dat[6:0]; daa_pinv <= c_fl[F_PARINV]; end
                end
                default: ;   // NOP
              endcase
            end
          end else if (st == S_FREE && cfg_ibi_en && !sda_s) begin
            // Target pulled SDA low during Bus Available/Idle: Controller also pulls SDA low and
            // starts clocking within tCAS (§5.1.2.2, §5.1.6.2).
            ibi_auto <= 1'b1; sub <= X_IBI_START;
          end
        end

        // ---------------------------------------------------------------
        // Multi-step line sequences. `cnt` counts cycles in the current step.
        S_SEQ: begin
          cnt <= cnt + 16'd1;
          unique case (op)
            OP_WAIT: if (cnt >= dat) st <= S_FETCH;
            OP_START: begin
              unique case (step)
                4'd0: begin
                  if (in_start) begin st <= S_FETCH; end              // already have S/Sr
                  else if (sub == X_IBI_START) begin
                    // Target already pulled SDA low on a free bus: Controller also pulls it low (§5.1.2.2)
                    drive(D_0); step <= 4'd3; cnt <= '0;
                  end else if (scl_q && ((!sda_oe_q && sda_s) || (sda_oe_q && sda_o_q))) begin
                    // SDA high with SCL high: SDA falls -> S (bus free, Open Drain) or Sr (Push-Pull).
                    drive(D_0); pu_q <= !bus_act; step <= 4'd3; cnt <= '0;
                  end else begin
                    // SDA low (or SCL low): SCL low, release SDA, SCL high, then SDA falls.
                    scl_q <= 1'b0; step <= 4'd1; cnt <= '0;
                  end
                end
                4'd1: begin
                  if (cnt == 16'd1) begin drive(D_REL); pu_q <= 1'b1; end
                  if (cnt >= {8'd0, cfg_t_low_od}) begin scl_q <= 1'b1; step <= 4'd2; cnt <= '0; end
                end
                4'd2: if (cnt >= {8'd0, cfg_t_cbp}) begin                       // tCBSr
                  drive(D_0); pu_q <= 1'b0; step <= 4'd3; cnt <= '0;
                end
                4'd3: if (cnt >= {8'd0, cfg_t_cas}) begin                       // tCAS / tCASr
                  in_start <= 1'b1; bus_act <= 1'b1; st <= S_FETCH;
                  if (sub == X_DAA_SR)    sub <= X_DAA_HDR;
                  if (sub == X_IBI_START) sub <= X_IBI_HDR;
                end
                default: st <= S_FETCH;
              endcase
            end
            OP_STOP: begin
              unique case (step)
                4'd0: begin
                  if (scl_q && sda_oe_q && !sda_o_q) begin step <= 4'd2; cnt <= '0; end  // SDA low, SCL high
                  else if (!scl_q) begin pu_q <= 1'b0; step <= 4'd1; cnt <= '0; end
                  else begin scl_q <= 1'b0; pu_q <= 1'b0; step <= 4'd1; cnt <= '0; end
                end
                4'd1: begin   // SCL low: drive SDA low, then SCL high
                  if (cnt == 16'd1) begin drive(D_0); pu_q <= 1'b0; end
                  if (cnt >= {8'd0, cfg_t_low_pp}) begin scl_q <= 1'b1; step <= 4'd2; cnt <= '0; end
                end
                4'd2: if (cnt >= {8'd0, cfg_t_cbp}) begin                       // tCBP
                  drive(D_1); step <= 4'd3; cnt <= '0;                           // SDA rises: P
                end
                4'd3: if (cnt >= 16'd1) begin
                  drive(D_REL); pu_q <= 1'b1; step <= 4'd4; cnt <= '0;           // hand to pull-up/keeper
                end
                4'd4: if (cnt >= cfg_t_buf) begin                                // bus free time
                  st <= S_FREE; in_start <= 1'b0; bus_act <= 1'b0; daa_end <= 1'b0; ibi_auto <= 1'b0;
                  if (sub == X_IBI_STOP || sub == X_ABORT_STOP) sub <= X_NONE;
                end
                default: st <= S_FREE;
              endcase
            end
            OP_PATTERN: begin
              unique case (step)
                4'd0: begin scl_q <= 1'b0; step <= 4'd1; cnt <= '0; end
                4'd1: if (cnt >= {8'd0, cfg_t_pat}) begin drive(D_1); pu_q <= 1'b0; subn <= '0; step <= 4'd2; cnt <= '0; end
                4'd2: if (cnt >= {8'd0, cfg_t_pat}) begin
                  if (subn == dat[3:0]) begin
                    in_start <= 1'b0; st <= S_FETCH;          // leaves SCL low
                  end else begin
                    sda_o_q <= ~sda_o_q; subn <= subn + 4'd1; cnt <= '0;
                  end
                end
                default: st <= S_FETCH;
              endcase
            end
            default: st <= S_FETCH;
          endcase
        end

        // ---------------------------------------------------------------
        S_BIT_START: fall_to(4'd0);

        S_BIT_LO: begin
          cnt <= cnt + 16'd1;
          if (cnt == 16'd1) begin
            drive(lo_drive(bi));
            pu_q <= bit_od;
          end
          if (cnt >= {8'd0, t_low(bit_od)} - 16'd1 && want_stall && stall_cnt < cfg_t_stall) begin
            // Controller Clock Stalling (§5.1.2.5): the next host command is not here yet, so hold SCL
            // Low in the ACK/NACK, write parity or read T-bit phase, never with SCL High.
            stall_cnt <= stall_cnt + 16'd1;
            cnt <= cnt;
          end else if (cnt >= {8'd0, t_low(bit_od)} - 16'd1) begin
            // sample on the SCL rising edge
            logic s; logic [1:0] dh, de;
            if (want_stall) stall_to <= 1'b1;   // limit reached: finish the bit, then abort with P
            stall_cnt <= '0;
            s = sda_s;
            samp <= s;
            rx <= {rx[7:0], s};
            dh = lo_drive(bi); de = lo_drive(bi);
            unique case (op)
              OP_ADDR: begin
                if (bi < 4'd8) begin
                  if (fl[F_ARB] && dat[7 - bi] && !s) arb_lost <= 1'b1;   // §5.1.2.2.1
                end else begin
                  // ACK slot (9th bit)
                  if (arb_lost) begin
                    // header won by a Target. rx[0] is the RnW the Target sent.
                    if (rx[0] && ibi_ack_now()) begin dh = D_0; de = D_REL; end // §5.1.2.3.2
                    else begin dh = D_REL; de = D_REL; end
                  end else if (!s && !dat[0]) begin
                    dh = D_0; de = D_0;      // Target ACKed a write: Controller takes SDA (§5.1.2.3.1)
                  end else begin
                    dh = D_REL; de = D_REL;  // ACK of a read (Target drives next) or NACK
                  end
                end
              end
              OP_READ: if (bi == 4'd8) begin
                // T-bit: T=0 End-of-Data, or Controller abort (LAST): drive low after SCL rises
                if (!s || fl[F_LAST]) begin dh = D_0; de = D_0; end
                else begin dh = D_REL; de = D_REL; end
              end
              default: ;
            endcase
            d_hi <= dh; d_end <= de;
            drive(dh);
            scl_q <= 1'b1;
            st <= S_BIT_HI; cnt <= '0;
          end
        end

        S_BIT_HI: begin
          cnt <= cnt + 16'd1;
          if (cnt == {8'd0, t_high(bit_od, b_init)} - 16'd2) drive(d_end);
          if (cnt >= {8'd0, t_high(bit_od, b_init)} - 16'd1) begin
            if (bi + 4'd1 >= blen) begin
              st <= S_BYTE_DONE;
            end else begin
              bi <= bi + 4'd1; fall_to(bi + 4'd1);
            end
          end
        end

        // ---------------------------------------------------------------
        S_BYTE_DONE: begin
          st <= S_FETCH;
          if (stall_to) begin
            stall_to <= 1'b0;
            push_rsp(RSP_ERR, 8'h04, {12'd0, op});
            sub <= X_ABORT_STOP; flush <= 1'b1;
          end else
          unique case (op)
            OP_ADDR: begin
              logic won_by_tgt, ack;
              won_by_tgt = arb_lost;
              ack = !samp;
              if (sub == X_DAA_HDR) begin
                if (ack) begin sub <= X_DAA_PID; subn <= '0; end
                else begin push_rsp(RSP_DAA, 8'h00, 16'd0); daa_end <= 1'b1; sub <= X_NONE; end
              end else if (won_by_tgt) begin
                // rx[8:1] = header bits A6..A0,RnW; rx[0] = ACK slot
                push_rsp(RSP_IBI, {7'd0, rx[1] && ibi_ack_now()}, {8'd0, rx[8:1]});
                if (rx[1] && ibi_ack_now() && cfg_ibi_bytes != 4'd0) begin
                  sub <= X_IBI_READ; subn <= cfg_ibi_bytes;
                end else sub <= X_IBI_STOP;
                if (host_op) flush <= 1'b1;
              end else begin
                // Arbitrable header: report what was on the wire. Push-Pull header: our own bits.
                push_rsp(RSP_ADDR, {6'd0, 1'b0, ack}, {8'd0, fl[F_ARB] ? rx[8:1] : dat[7:0]});
                if (sub == X_IBI_HDR) sub <= X_IBI_STOP;   // nobody won: spurious START
                else if (!ack && fl[F_NACKSTOP]) begin sub <= X_ABORT_STOP; flush <= 1'b1; end
              end
            end
            OP_WRITE: ;
            OP_READ: begin
              push_rsp(RSP_READ, {7'd0, samp}, {8'd0, rx[8:1]});
              if (samp && fl[F_LAST]) in_start <= 1'b1;   // Controller abort produced an Sr
              // Target ended the read (T=0) before the host's last READ: drop the host's remaining
              // READs up to its STOP, which is then executed (§5.1.2.3.4).
              if (!samp && !fl[F_LAST] && host_op) flush <= 1'b1;
              if (sub == X_IBI_READ) begin
                if (!samp || subn == 4'd1) sub <= X_IBI_STOP;
                else subn <= subn - 4'd1;
              end
            end
            OP_RAWRD: begin
              if (sub == X_DAA_ACK) begin
                push_rsp(RSP_DAA, {6'd0, !samp, 1'b1}, {9'd0, daa_da});
                sub <= X_NONE;
              end else begin
                push_rsp(RSP_RAWRD, 8'd0, {8'd0, rx[7:0]});
                if (sub == X_DAA_PID) begin
                  if (subn == 4'd7) sub <= X_DAA_DA; else subn <= subn + 4'd1;
                end
              end
            end
            OP_RAWWR: if (sub == X_DAA_DA) sub <= X_DAA_ACK;
            default: ;
          endcase
        end
        default: st <= S_FREE;
      endcase
    end
  end
endmodule
