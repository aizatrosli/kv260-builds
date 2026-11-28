// feed_rx: AXI4-Stream slave for the 16-byte fpga-trading wire format (SPEC.md).
//
// 32-bit tdata, 4 beats per message, tlast on beat 3:
//   beat 0: [7:0] type ('C' 0x43 / 'L' 0x4C / 'T' 0x54), [15:8] side (0 bid/buy, else ask/sell)
//   beat 1: price   beat 2: volume   beat 3: ts low 32 (not used by the book)
// The assembled message is presented on a valid/ready port held in one output register.
//
// Back-pressure is honest: beats 0..2 go into staging registers and are always accepted; beat 3
// (the one that completes a message) is only accepted when the output register is empty or is
// being drained in the same cycle, so tready never drops data.
//
// Framing errors (counted in err_cnt_o, message discarded, no feature vector emitted):
//   * tlast on beat 0..2                  -> runt, realign on the next beat
//   * no tlast on beat 3                  -> drop beats until the next tlast, then realign
//   * unknown type byte                   -> discard (the golden model raises on it)
// Side decoding matches ftrade.wire.decode: any nonzero side byte means ask/sell.
`timescale 1ns/1ps

module feed_rx (
  input  wire        clk_i,
  input  wire        rst_ni,          // synchronous, active low (AXI aresetn convention)
  input  wire [31:0] cycle_i,         // free-running cycle counter, stamped on the last beat
  // AXI4-Stream slave
  input  wire [31:0] s_axis_tdata,
  input  wire        s_axis_tvalid,
  output wire        s_axis_tready,
  input  wire        s_axis_tlast,
  // decoded message
  output reg         msg_valid_o,
  input  wire        msg_ready_i,
  output reg  [1:0]  msg_kind_o,      // 0 clear, 1 level, 2 trade
  output reg         msg_side_o,      // 0 bid/buy, 1 ask/sell
  output reg  [31:0] msg_price_o,
  output reg  [31:0] msg_vol_o,
  output reg  [31:0] msg_cycle_o,     // cycle_i when the last beat was accepted
  output reg  [15:0] err_cnt_o
);
  localparam [1:0] K_CLEAR = 2'd0, K_LEVEL = 2'd1, K_TRADE = 2'd2;

  reg  [1:0]  beat;                   // next beat index 0..3
  reg         drop;                   // resync: discard until tlast
  reg  [1:0]  st_kind;
  reg         st_ok, st_side;
  reg  [31:0] st_price, st_vol;

  wire last_beat = (beat == 2'd3);
  assign s_axis_tready = drop | ~last_beat | ~msg_valid_o | msg_ready_i;
  wire fire = s_axis_tvalid & s_axis_tready;

  // beat-0 decode
  wire [7:0] t_byte = s_axis_tdata[7:0];
  wire       t_ok   = (t_byte == 8'h43) | (t_byte == 8'h4C) | (t_byte == 8'h54);
  wire [1:0] t_kind = (t_byte == 8'h43) ? K_CLEAR : (t_byte == 8'h4C) ? K_LEVEL : K_TRADE;

  always @(posedge clk_i) begin
    if (!rst_ni) begin
      beat <= 2'd0; drop <= 1'b0; msg_valid_o <= 1'b0; err_cnt_o <= 16'd0;
      st_kind <= 2'd0; st_ok <= 1'b0; st_side <= 1'b0; st_price <= 32'd0; st_vol <= 32'd0;
      msg_kind_o <= 2'd0; msg_side_o <= 1'b0; msg_price_o <= 32'd0; msg_vol_o <= 32'd0;
      msg_cycle_o <= 32'd0;
    end else begin
      if (msg_valid_o && msg_ready_i) msg_valid_o <= 1'b0;
      if (fire) begin
        if (drop) begin
          if (s_axis_tlast) begin drop <= 1'b0; beat <= 2'd0; end
        end else if (!last_beat) begin
          case (beat)
            2'd0: begin st_kind <= t_kind; st_ok <= t_ok; st_side <= |s_axis_tdata[15:8]; end
            2'd1: st_price <= s_axis_tdata;
            default: st_vol <= s_axis_tdata;
          endcase
          if (s_axis_tlast) begin beat <= 2'd0; err_cnt_o <= err_cnt_o + 16'd1; end
          else beat <= beat + 2'd1;
        end else begin
          beat <= 2'd0;
          if (!s_axis_tlast) begin
            drop <= 1'b1; err_cnt_o <= err_cnt_o + 16'd1;
          end else if (!st_ok) begin
            err_cnt_o <= err_cnt_o + 16'd1;
          end else begin
            msg_valid_o <= 1'b1;
            msg_kind_o  <= st_kind;  msg_side_o <= st_side;
            msg_price_o <= st_price; msg_vol_o  <= st_vol;
            msg_cycle_o <= cycle_i;
          end
        end
      end
    end
  end
endmodule
