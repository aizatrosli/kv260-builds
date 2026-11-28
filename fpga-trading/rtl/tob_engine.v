// tob_engine: top-of-book + feature engine (fpga-trading SPEC.md), AXIS in -> AXIS out.
//
//   s_axis (32b, 4 beats/msg) -> feed_rx -> book control (bid/ask book_side + flow register)
//     -> features (21-cycle pipeline) -> output FIFO -> m_axis (OUT_W = 32: 8 beats/msg,
//        tlast on the 8th, word k = F_k; OUT_W = 256: one beat/msg, F_k at [32k +: 32])
//
// Flow control is credit based: a message is admitted to the book only while fewer than
// FIFO_DEPTH messages are between the book and the m_axis port, so the book and feature
// pipeline never stall and the output FIFO can never overflow. Back-pressure on m_axis
// propagates to s_axis_tready through feed_rx.
//
// Status (directly usable as AXI-Lite / AXI GPIO readback):
//   cycle_o      free-running cycle counter
//   lat_last_o   per-message latency: cycles from the cycle the last input beat was accepted
//                to the first cycle the first output beat of that message is valid
//   lat_min_o / lat_max_o, msg_in_o (accepted msgs), msg_out_o (vectors started), rx_err_o
//
// Book update: one message in flight in the book, cycle A = compare, cycle B = commit, so the
// book accepts a message every 2 cycles (the 32-bit input delivers at most one every 4).
`timescale 1ns/1ps

module tob_engine #(
  parameter OUT_W   = 32,             // 32 or 256
  parameter FIFO_AW = 4               // output FIFO depth = 2^FIFO_AW messages
) (
  input  wire             clk_i,
  input  wire             rst_ni,     // synchronous, active low
  // AXI4-Stream slave: wire-format messages
  input  wire [31:0]      s_axis_tdata,
  input  wire             s_axis_tvalid,
  output wire             s_axis_tready,
  input  wire             s_axis_tlast,
  // AXI4-Stream master: feature vectors
  output reg  [OUT_W-1:0] m_axis_tdata,
  output reg              m_axis_tvalid,
  input  wire             m_axis_tready,
  output reg              m_axis_tlast,
  // status
  output reg  [31:0]      cycle_o,
  output reg  [31:0]      lat_last_o,
  output reg  [31:0]      lat_min_o,
  output reg  [31:0]      lat_max_o,
  output reg  [31:0]      msg_in_o,
  output reg  [31:0]      msg_out_o,
  output wire [15:0]      rx_err_o
);
  localparam BEATS = 256 / OUT_W;     // output beats per message
  localparam DEPTH = 1 << FIFO_AW;
  localparam [1:0] K_CLEAR = 2'd0, K_LEVEL = 2'd1, K_TRADE = 2'd2;

  always @(posedge clk_i) begin
    if (!rst_ni) cycle_o <= 32'd0;
    else         cycle_o <= cycle_o + 32'd1;
  end

  // ---------------- receive ----------------
  wire        m_valid, m_side;
  wire [1:0]  m_kind;
  wire [31:0] m_price, m_vol, m_cyc;
  wire        m_ready;
  feed_rx u_rx (
    .clk_i(clk_i), .rst_ni(rst_ni), .cycle_i(cycle_o),
    .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid),
    .s_axis_tready(s_axis_tready), .s_axis_tlast(s_axis_tlast),
    .msg_valid_o(m_valid), .msg_ready_i(m_ready), .msg_kind_o(m_kind), .msg_side_o(m_side),
    .msg_price_o(m_price), .msg_vol_o(m_vol), .msg_cycle_o(m_cyc), .err_cnt_o(rx_err_o));

  // ---------------- book control ----------------
  reg  [FIFO_AW:0] inflight;          // admitted, not yet popped from the output FIFO
  reg              st_a, st_b;        // book cycle A / B in progress
  wire             go = m_valid & ~st_a & (inflight != DEPTH[FIFO_AW:0]);
  assign m_ready = ~st_a & (inflight != DEPTH[FIFO_AW:0]);
  wire             fifo_pop;

  wire is_clr = (m_kind == K_CLEAR), is_lvl = (m_kind == K_LEVEL), is_trd = (m_kind == K_TRADE);

  // Full 8-level books are exported by book_side; the features only need the best price, the
  // top-5 volumes and the non-empty flag, the rest is synthesis-pruned (handy for an ILA/debug).
  /* verilator lint_off UNUSEDSIGNAL */
  wire [255:0] bid_p, bid_v, ask_p, ask_v;
  wire [7:0]   bid_ok, ask_ok;
  /* verilator lint_on UNUSEDSIGNAL */
  book_side #(.N(8), .DESC(1)) u_bid (
    .clk_i(clk_i), .rst_ni(rst_ni), .req_i(go & is_lvl & ~m_side), .clr_i(go & is_clr),
    .price_i(m_price), .vol_i(m_vol), .price_o(bid_p), .vol_o(bid_v), .valid_o(bid_ok));
  book_side #(.N(8), .DESC(0)) u_ask (
    .clk_i(clk_i), .rst_ni(rst_ni), .req_i(go & is_lvl & m_side), .clr_i(go & is_clr),
    .price_i(m_price), .vol_i(m_vol), .price_o(ask_p), .vol_o(ask_v), .valid_o(ask_ok));

  // trade-flow register: flow += (s*vol - flow) >>> 4, wrapped to int32.
  // cycle A: dlt = s*vol - flow (35-bit signed); cycle B: flow += dlt >>> 4.
  reg  [31:0] flow;
  /* verilator lint_off UNUSEDSIGNAL */
  reg  [34:0] dlt;                                  // [3:0] are shifted out by >>> 4
  /* verilator lint_on UNUSEDSIGNAL */
  reg         a_trd, a_clr;
  reg  [31:0] a_cyc, b_cyc;
  wire [34:0] sx   = m_side ? (35'd0 - {3'b000, m_vol}) : {3'b000, m_vol};
  wire [34:0] fl35 = {{3{flow[31]}}, flow};
  wire [31:0] dsh  = {dlt[34], dlt[34:4]};          // arithmetic >>> 4 (floor), mod 2^32
  always @(posedge clk_i) begin
    if (!rst_ni) begin
      st_a <= 1'b0; st_b <= 1'b0; a_trd <= 1'b0; a_clr <= 1'b0; flow <= 32'd0; dlt <= 35'd0;
      a_cyc <= 32'd0; b_cyc <= 32'd0;
    end else begin
      st_a  <= go;
      st_b  <= st_a;
      a_trd <= go & is_trd;
      a_clr <= go & is_clr;
      dlt   <= sx - fl35;
      if (go) a_cyc <= m_cyc;
      if (st_a) b_cyc <= a_cyc;
      if (st_a && a_clr)      flow <= 32'd0;
      else if (st_a && a_trd) flow <= flow + dsh;
    end
  end

  // ---------------- features ----------------
  wire         f_valid;
  wire [255:0] f_vec;
  wire [31:0]  f_tag;
  features u_feat (
    .clk_i(clk_i), .rst_ni(rst_ni), .in_valid_i(st_b), .tag_i(b_cyc),
    .bid_ok_i(bid_ok[0]), .ask_ok_i(ask_ok[0]), .bid_p0_i(bid_p[31:0]), .ask_p0_i(ask_p[31:0]),
    .bid_v_i(bid_v[159:0]), .ask_v_i(ask_v[159:0]), .flow_i(flow),
    .out_valid_o(f_valid), .out_feat_o(f_vec), .out_tag_o(f_tag));

  // ---------------- output FIFO (LUTRAM, async read) with empty bypass ----------------
  reg  [287:0]       mem [0:DEPTH-1];
  reg  [FIFO_AW-1:0] wr, rd;
  reg  [FIFO_AW:0]   cnt;
  always @(posedge clk_i) if (f_valid) mem[wr] <= {f_tag, f_vec};
  wire               empty = (cnt == {(FIFO_AW+1){1'b0}});
  wire               head_ok = ~empty | f_valid;          // bypass: write-through when empty
  wire [287:0]       head = empty ? {f_tag, f_vec} : mem[rd];

  reg  [2:0]   w;                                         // next output word of the head
  wire         load = head_ok & (~m_axis_tvalid | m_axis_tready);
  wire         last_w = (w == BEATS[2:0] - 3'd1);
  assign       fifo_pop = load & last_w;
  /* verilator lint_off UNUSEDSIGNAL */
  wire [255:0] head_sh = head[255:0] >> {w, 5'd0};          // only [OUT_W-1:0] is used
  /* verilator lint_on UNUSEDSIGNAL */
  wire [31:0]  lat_now = cycle_o + 32'd1 - head[287:256];

  always @(posedge clk_i) begin
    if (!rst_ni) begin
      wr <= {FIFO_AW{1'b0}}; rd <= {FIFO_AW{1'b0}}; cnt <= {(FIFO_AW+1){1'b0}};
      inflight <= {(FIFO_AW+1){1'b0}}; w <= 3'd0;
      m_axis_tvalid <= 1'b0; m_axis_tlast <= 1'b0; m_axis_tdata <= {OUT_W{1'b0}};
      lat_last_o <= 32'd0; lat_min_o <= 32'hFFFF_FFFF; lat_max_o <= 32'd0;
      msg_in_o <= 32'd0; msg_out_o <= 32'd0;
    end else begin
      if (f_valid) wr <= wr + {{(FIFO_AW-1){1'b0}}, 1'b1};
      if (fifo_pop) rd <= rd + {{(FIFO_AW-1){1'b0}}, 1'b1};
      cnt      <= cnt + {{FIFO_AW{1'b0}}, f_valid} - {{FIFO_AW{1'b0}}, fifo_pop};
      inflight <= inflight + {{FIFO_AW{1'b0}}, go} - {{FIFO_AW{1'b0}}, fifo_pop};
      if (go) msg_in_o <= msg_in_o + 32'd1;
      if (load) begin
        m_axis_tvalid <= 1'b1;
        m_axis_tdata  <= head_sh[OUT_W-1:0];
        m_axis_tlast  <= last_w;
        w             <= last_w ? 3'd0 : w + 3'd1;
        if (w == 3'd0) begin
          lat_last_o <= lat_now;
          if (lat_now < lat_min_o) lat_min_o <= lat_now;
          if (lat_now > lat_max_o) lat_max_o <= lat_now;
          msg_out_o  <= msg_out_o + 32'd1;
        end
      end else if (m_axis_tready) begin
        m_axis_tvalid <= 1'b0;
      end
    end
  end
endmodule
