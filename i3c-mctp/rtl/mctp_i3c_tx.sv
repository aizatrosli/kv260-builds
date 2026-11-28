// MCTP-over-I3C packetizer (Controller -> Target private write payload).
//
// Splits one MCTP message into packets and emits, per packet, the byte stream that follows the
// I3C address phase:  [Hdr ver] [Dest EID] [Src EID] [SOM|EOM|Seq|TO|Tag] [payload...] [PEC]
//   Byte layout: DSP0233 v1.0.1 §5.2.1.1 Table 1; field definitions: DSP0236 v1.4.0 §8.2 Table 1.
//   Header version = 0001b, reserved [7:4] = 0 (DSP0236 §8.2 Table 1, §5.3).
//   Message header (IC | message type) is simply the first message byte, so it lands in the first
//   packet only (DSP0236 §8.2 Table 1). The caller supplies it as msg byte 0.
//   Pkt Seq # starts at seq_i and increments mod 4 (DSP0236 §8.2 Table 1).
//   TO and Tag are constant from SOM to EOM (DSP0236 §8.2 Table 1).
//   All packets but the EOM one carry exactly tu_i payload bytes (DSP0236 §8.4.2). The spec
//   requires tu_i >= 64 (baseline transmission unit); smaller values are allowed here on purpose
//   so the generator can produce non-compliant traffic for negative testing.
//   PEC: CRC-8/0x07 over the address byte {dyn_addr, RnW=0} and all following bytes (DSP0233 §5.3.1).
//
// out_first_o marks the first byte of a packet (the transfer engine opens a new private write),
// out_last_o marks the PEC byte (the engine closes the transfer).
`timescale 1ns/1ps
module mctp_i3c_tx import crc8_pec_pkg::*; (
  input  logic        clk_i,
  input  logic        rst_ni,
  // job
  input  logic        start_i,
  output logic        busy_o,
  input  logic [6:0]  dyn_addr_i,
  input  logic [7:0]  dst_eid_i,
  input  logic [7:0]  src_eid_i,
  input  logic        to_i,
  input  logic [2:0]  tag_i,
  input  logic [1:0]  seq_i,
  input  logic [7:0]  tu_i,        // payload bytes per non-EOM packet (spec: >= 64)
  input  logic [15:0] msg_len_i,   // message bytes including the IC|type byte, >= 1
  input  logic [3:0]  hdr_ver_i,   // 4'b0001 for compliant traffic
  input  logic        pec_corrupt_i, // invert PEC bit 0 on every packet (negative testing)
  output logic [1:0]  seq_next_o,  // seq of the last EOM packet + 1, for the next message
  // message bytes in
  input  logic        msg_valid_i,
  input  logic [7:0]  msg_data_i,
  output logic        msg_ready_o,
  // packet bytes out
  output logic        out_valid_o,
  output logic [7:0]  out_data_o,
  output logic        out_first_o,
  output logic        out_last_o,
  input  logic        out_ready_i
);
  typedef enum logic [2:0] {IDLE, H1, H2, H3, H4, PAY, PEC} st_e;
  st_e st;
  logic [6:0]  addr; logic [7:0] dst, src, tu; logic to; logic [2:0] tag; logic [1:0] seq;
  logic [3:0]  hver; logic corrupt;
  logic [15:0] rem;      // message bytes not yet sent
  logic [7:0]  cnt;      // payload bytes left in this packet
  logic        som, eom;
  logic [7:0]  crc;
  logic [7:0]  crc_addr;

  assign busy_o   = (st != IDLE);
  assign crc_addr = crc8_pec(8'h00, {addr, 1'b0});

  always_comb begin
    out_valid_o = 1'b0; out_data_o = 8'h00; out_first_o = 1'b0; out_last_o = 1'b0;
    msg_ready_o = 1'b0;
    unique case (st)
      H1:  begin out_valid_o = 1'b1; out_data_o = {4'b0000, hver}; out_first_o = 1'b1; end
      H2:  begin out_valid_o = 1'b1; out_data_o = dst; end
      H3:  begin out_valid_o = 1'b1; out_data_o = src; end
      H4:  begin out_valid_o = 1'b1; out_data_o = {som, eom, seq, to, tag}; end
      PAY: begin out_valid_o = msg_valid_i; out_data_o = msg_data_i; msg_ready_o = out_ready_i; end
      PEC: begin out_valid_o = 1'b1; out_data_o = crc ^ {7'd0, corrupt}; out_last_o = 1'b1; end
      default: ;
    endcase
  end

  wire fire = out_valid_o & out_ready_i;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      st <= IDLE; addr <= '0; dst <= '0; src <= '0; tu <= '0; to <= 1'b0; tag <= '0; seq <= '0;
      hver <= '0; corrupt <= 1'b0; rem <= '0; cnt <= '0; som <= 1'b0; eom <= 1'b0; crc <= '0;
      seq_next_o <= '0;
    end else begin
      unique case (st)
        IDLE: if (start_i) begin
          addr <= dyn_addr_i; dst <= dst_eid_i; src <= src_eid_i; to <= to_i; tag <= tag_i;
          seq <= seq_i; tu <= tu_i; hver <= hdr_ver_i; corrupt <= pec_corrupt_i;
          rem <= msg_len_i; som <= 1'b1;
          st <= H1;
        end
        H1: if (fire) begin
          // Decide EOM and this packet's payload size before byte 4 goes out.
          eom <= (rem <= {8'd0, tu});
          cnt <= (rem <= {8'd0, tu}) ? rem[7:0] : tu;
          crc <= crc8_pec(crc_addr, out_data_o);
          st  <= H2;
        end
        H2:  if (fire) begin crc <= crc8_pec(crc, out_data_o); st <= H3; end
        H3:  if (fire) begin crc <= crc8_pec(crc, out_data_o); st <= H4; end
        H4:  if (fire) begin crc <= crc8_pec(crc, out_data_o); st <= PAY; end
        PAY: if (fire) begin
          crc <= crc8_pec(crc, out_data_o);
          rem <= rem - 16'd1;
          cnt <= cnt - 8'd1;
          if (cnt == 8'd1) st <= PEC;
        end
        PEC: if (fire) begin
          seq <= seq + 2'd1;
          som <= 1'b0;
          if (eom) begin seq_next_o <= seq + 2'd1; st <= IDLE; end
          else st <= H1;
        end
        default: st <= IDLE;
      endcase
    end
  end
endmodule
