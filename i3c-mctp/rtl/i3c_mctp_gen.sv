// I3C + MCTP signal generator core: I3C SDR engine + MCTP-over-I3C packet sequencer.
//
// Two command sources feed the engine:
//   * host commands (raw engine commands, see i3c_sdr_engine.sv): CCCs, ENTDAA, private reads, IBI
//     handling, patterns, deliberately broken frames;
//   * the MCTP sequencer, which turns one MCTP message into one I3C private write per packet
//     (DSP0233 v1.0.1 §5.2.1, Figure 4):
//        S, [7'h7E/W, ACK, Sr,]  DA/W, ACK, {Hdr ver, Dest EID, Src EID, flags, payload..., PEC}+T, P
//     The optional 7'h7E broadcast header before the packet is allowed by DSP0233 §5.2.1.1 (l.405).
//     Without it, the Target address follows S and is an arbitrable (Open-Drain) header
//     (I3C Basic §5.1.2.2). With it, the address follows Sr and is Push-Pull (§5.1.2.2.4).
//     A NACK of the Target address (DSP0233 §5.2.1.2: buffer full / absent) ends the transfer with P
//     and the rest of that packet is discarded by the engine; mctp_nack_o counts it.
// Host commands are accepted only while no MCTP message is in flight.
`timescale 1ns/1ps
module i3c_mctp_gen (
  input  logic        clk_i,
  input  logic        rst_ni,
  // engine timing/config
  input  logic [7:0]  cfg_t_low_pp, cfg_t_high_pp, cfg_t_low_od, cfg_t_high_od, cfg_t_high_init,
  input  logic [7:0]  cfg_t_cas, cfg_t_cbp, cfg_t_pat,
  input  logic [15:0] cfg_t_buf,
  input  logic [15:0] cfg_t_stall,
  input  logic        cfg_ibi_en, cfg_ibi_ack,
  input  logic [3:0]  cfg_ibi_bytes,
  // host command / response streams
  input  logic        hcmd_valid_i,
  input  logic [31:0] hcmd_i,
  output logic        hcmd_ready_o,
  output logic        rsp_valid_o,
  output logic [31:0] rsp_o,
  // MCTP job
  input  logic        mctp_start_i,
  output logic        mctp_busy_o,
  input  logic [6:0]  mctp_da_i,
  input  logic [7:0]  mctp_dst_eid_i, mctp_src_eid_i,
  input  logic        mctp_to_i,
  input  logic [2:0]  mctp_tag_i,
  input  logic [1:0]  mctp_seq_i,
  input  logic [7:0]  mctp_tu_i,
  input  logic [15:0] mctp_len_i,
  input  logic [3:0]  mctp_hdr_ver_i,
  input  logic        mctp_pec_corrupt_i,
  input  logic        mctp_use_7e_i,
  output logic [1:0]  mctp_seq_next_o,
  output logic [15:0] mctp_pkts_o,       // packets sent (address ACKed) in the last job
  output logic [15:0] mctp_nack_o,       // packets whose address was NACKed in the last job
  input  logic        msg_valid_i,
  input  logic [7:0]  msg_data_i,
  output logic        msg_ready_o,
  // status
  output logic        bus_free_o,
  // pins
  output logic        scl_o, sda_oe_o, sda_o, sda_pu_en_o,
  input  logic        sda_i
);
  localparam logic [3:0] OP_START = 4'd1, OP_STOP = 4'd2, OP_ADDR = 4'd3, OP_WRITE = 4'd4;
  localparam logic [7:0] FL_ARB = 8'h01, FL_NACKSTOP = 8'h04;

  // ---------------- MCTP packetizer ----------------
  logic       pk_valid, pk_first, pk_last, pk_ready, pk_busy;
  logic [7:0] pk_data;
  mctp_i3c_tx u_tx (
    .clk_i, .rst_ni,
    .start_i(mctp_start_i && !job && e_ready), .busy_o(pk_busy),
    .dyn_addr_i(mctp_da_i), .dst_eid_i(mctp_dst_eid_i), .src_eid_i(mctp_src_eid_i),
    .to_i(mctp_to_i), .tag_i(mctp_tag_i), .seq_i(mctp_seq_i), .tu_i(mctp_tu_i),
    .msg_len_i(mctp_len_i), .hdr_ver_i(mctp_hdr_ver_i), .pec_corrupt_i(mctp_pec_corrupt_i),
    .seq_next_o(mctp_seq_next_o),
    .msg_valid_i, .msg_data_i, .msg_ready_o,
    .out_valid_o(pk_valid), .out_data_o(pk_data), .out_first_o(pk_first), .out_last_o(pk_last),
    .out_ready_i(pk_ready));

  // ---------------- packet sequencer ----------------
  typedef enum logic [2:0] {Q_IDLE, Q_S, Q_7E, Q_SR, Q_DA, Q_DATA, Q_STOP, Q_WAIT} q_e;
  q_e q;
  logic [6:0] da_q; logic use7e_q; logic job;
  logic [31:0] q_cmd; logic q_valid;
  logic        e_ready;

  always_comb begin
    q_valid = 1'b0; q_cmd = '0; pk_ready = 1'b0;
    unique case (q)
      Q_S:    begin q_valid = 1'b1; q_cmd = {OP_START, 8'h00, 4'd0, 16'd0}; end
      Q_7E:   begin q_valid = 1'b1; q_cmd = {OP_ADDR, FL_ARB | FL_NACKSTOP, 4'd0, 8'd0, 7'h7E, 1'b0}; end
      Q_SR:   begin q_valid = 1'b1; q_cmd = {OP_START, 8'h00, 4'd0, 16'd0}; end
      Q_DA:   begin q_valid = 1'b1;
                    q_cmd = {OP_ADDR, (use7e_q ? 8'h00 : FL_ARB) | FL_NACKSTOP, 4'd0, 8'd0, da_q, 1'b0}; end
      Q_DATA: begin q_valid = pk_valid; q_cmd = {OP_WRITE, 8'h00, 4'd0, 8'd0, pk_data}; pk_ready = e_ready; end
      Q_STOP: begin q_valid = 1'b1; q_cmd = {OP_STOP, 8'h00, 4'd0, 16'd0}; end
      default: ;
    endcase
  end

  // engine command mux
  logic        e_valid; logic [31:0] e_cmd;
  assign e_valid      = job ? q_valid : hcmd_valid_i;
  assign e_cmd        = job ? q_cmd   : hcmd_i;
  assign hcmd_ready_o = !job && e_ready && !mctp_start_i;

  logic        rsp_v; logic [31:0] rsp_d;
  assign rsp_valid_o = rsp_v;
  assign rsp_o       = rsp_d;
  assign mctp_busy_o = job;

  wire e_fire = e_valid && e_ready;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      q <= Q_IDLE; da_q <= '0; use7e_q <= 1'b0; job <= 1'b0; mctp_pkts_o <= '0; mctp_nack_o <= '0;
    end else begin
      if (mctp_start_i && !job && e_ready) begin
        job <= 1'b1; da_q <= mctp_da_i; use7e_q <= mctp_use_7e_i; q <= Q_WAIT;
        mctp_pkts_o <= '0; mctp_nack_o <= '0;
      end
      // count ACK/NACK of the Target address (responses tagged RSP_ADDR with header = {da,0})
      if (job && rsp_v && rsp_d[31:28] == 4'd3 && rsp_d[7:0] == {da_q, 1'b0}) begin
        if (rsp_d[20]) mctp_pkts_o <= mctp_pkts_o + 16'd1;
        else           mctp_nack_o <= mctp_nack_o + 16'd1;
      end
      unique case (q)
        Q_WAIT: begin
          if (pk_valid && pk_first) q <= Q_S;
          else if (!pk_busy && job && !mctp_start_i) begin job <= 1'b0; q <= Q_IDLE; end
        end
        Q_S:    if (e_fire) q <= use7e_q ? Q_7E : Q_DA;
        Q_7E:   if (e_fire) q <= Q_SR;
        Q_SR:   if (e_fire) q <= Q_DA;
        Q_DA:   if (e_fire) q <= Q_DATA;
        Q_DATA: if (e_fire && pk_last) q <= Q_STOP;
        Q_STOP: if (e_fire) q <= Q_WAIT;
        default: ;
      endcase
    end
  end

  logic e_rsp_v; logic [31:0] e_rsp;
  always_comb begin rsp_v = e_rsp_v; rsp_d = e_rsp; end

  i3c_sdr_engine u_eng (
    .clk_i, .rst_ni,
    .cfg_t_low_pp, .cfg_t_high_pp, .cfg_t_low_od, .cfg_t_high_od, .cfg_t_high_init,
    .cfg_t_cas, .cfg_t_cbp, .cfg_t_buf, .cfg_t_stall, .cfg_t_pat, .cfg_ibi_en, .cfg_ibi_ack, .cfg_ibi_bytes,
    .cmd_valid_i(e_valid), .cmd_i(e_cmd), .cmd_ready_o(e_ready),
    .rsp_valid_o(e_rsp_v), .rsp_o(e_rsp),
    .bus_free_o, .busy_o(),
    .scl_o, .sda_oe_o, .sda_o, .sda_i, .sda_pu_en_o);
endmodule
