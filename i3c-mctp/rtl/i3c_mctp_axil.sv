// AXI4-Lite register front end for the I3C/MCTP generator (for the KV260 PS via an M_AXI_HPM port).
//
// Register map (32-bit, byte offsets):
//   0x00 ID          RO  0x13C3_0100 (I3C/MCTP generator, v1.0)
//   0x04 CTRL        RW  [0] IBI detect on free bus  [1] ACK IBIs  [7:4] IBI bytes to read (MDB+payload)
//                        [8] W1: clear FIFOs and sticky flags
//   0x08 STATUS      RO  [0] bus free  [1] MCTP busy  [2] CMD FIFO full  [3] RSP FIFO empty
//                        [4] RSP overflow (sticky)  [5] MSG FIFO full  [31:16] RSP FIFO count
//   0x10 TIM0        RW  [7:0] t_low_pp [15:8] t_high_pp [23:16] t_low_od [31:24] t_high_od   (clk cycles)
//   0x14 TIM1        RW  [7:0] t_high_init [15:8] t_cas [23:16] t_cbp [31:24] t_pat
//   0x18 TIM2        RW  [15:0] t_buf [31:16] t_stall (max SCL-low clock stall, 0 = never stall)
//   0x20 CMD         WO  push one engine command (format: i3c_sdr_engine.sv)
//   0x24 RSP         RO  pop one response (0 when empty; check STATUS[3] first)
//   0x30 MCTP_CFG0   RW  [6:0] DA [8] TO [11:9] tag [13:12] seq [19:16] hdr ver [20] corrupt PEC
//                        [21] 7'h7E before the address [31:24] transmission unit
//   0x34 MCTP_CFG1   RW  [7:0] dest EID [15:8] src EID [31:16] message length (bytes, incl. IC|type)
//   0x38 MCTP_CTRL   W   [0] start (message bytes must already be in MSG FIFO)
//                    R   [0] busy [5:4] next seq number
//   0x3C MCTP_STAT   RO  [15:0] packets ACKed [31:16] packets NACKed (last job)
//   0x40 MSG         WO  push one message byte [7:0]
// Reset values of TIM0..TIM2 meet MIPI I3C Basic v1.1.1 Tables 86/87 (pure bus, 12.5 MHz) at
// CLK_MHZ = 100: tLOW=tHIGH=40 ns, tLOW_OD=250 ns, tHIGH_OD=40 ns, tHIGH_INIT=250 ns, tCAS=50 ns,
// tCBP=30 ns, bus free 1 us.
`timescale 1ns/1ps
module i3c_mctp_axil #(
  parameter int CMD_DEPTH = 512,
  parameter int RSP_DEPTH = 512,
  parameter int MSG_DEPTH = 4096
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  // AXI4-Lite slave
  input  logic [7:0]  s_axi_awaddr,
  input  logic        s_axi_awvalid,
  output logic        s_axi_awready,
  input  logic [31:0] s_axi_wdata,
  input  logic [3:0]  s_axi_wstrb,
  input  logic        s_axi_wvalid,
  output logic        s_axi_wready,
  output logic [1:0]  s_axi_bresp,
  output logic        s_axi_bvalid,
  input  logic        s_axi_bready,
  input  logic [7:0]  s_axi_araddr,
  input  logic        s_axi_arvalid,
  output logic        s_axi_arready,
  output logic [31:0] s_axi_rdata,
  output logic [1:0]  s_axi_rresp,
  output logic        s_axi_rvalid,
  input  logic        s_axi_rready,
  // pins (to pads)
  output logic        scl_o,
  output logic        sda_oe_o,
  output logic        sda_o,
  input  logic        sda_i,
  output logic        sda_pu_en_o,
  output logic        frame_o        // high while the bus is not free (scope trigger)
);
  // ---------------- registers ----------------
  logic [31:0] ctrl, tim0, tim1, tim2, mcfg0, mcfg1;
  logic        rsp_ovf;
  logic        mstart_q;
  logic [31:0] wdata_q;

  // ---------------- FIFOs ----------------
  logic cmd_wr, cmd_full, cmd_empty, cmd_rd; logic [31:0] cmd_q;
  logic rsp_wr, rsp_full, rsp_empty, rsp_rd; logic [31:0] rsp_q, rsp_w;
  logic msg_wr, msg_full, msg_empty, msg_rd; logic [7:0] msg_q;
  logic [$clog2(RSP_DEPTH):0] rsp_cnt;
  logic clr;

  sync_fifo #(.W(32), .DEPTH(CMD_DEPTH)) u_cmd (.clk_i, .rst_ni, .clr_i(clr), .wr_i(cmd_wr), .wdata_i(wdata_q),
    .full_o(cmd_full), .rd_i(cmd_rd), .rdata_o(cmd_q), .empty_o(cmd_empty), .count_o());
  sync_fifo #(.W(32), .DEPTH(RSP_DEPTH)) u_rsp (.clk_i, .rst_ni, .clr_i(clr), .wr_i(rsp_wr), .wdata_i(rsp_w),
    .full_o(rsp_full), .rd_i(rsp_rd), .rdata_o(rsp_q), .empty_o(rsp_empty), .count_o(rsp_cnt));
  sync_fifo #(.W(8), .DEPTH(MSG_DEPTH)) u_msg (.clk_i, .rst_ni, .clr_i(clr), .wr_i(msg_wr), .wdata_i(wdata_q[7:0]),
    .full_o(msg_full), .rd_i(msg_rd), .rdata_o(msg_q), .empty_o(msg_empty), .count_o());

  // ---------------- core ----------------
  logic hcmd_ready, rsp_v, mbusy, mstart, bus_free, msg_ready;
  logic [1:0] seq_next; logic [15:0] mpkts, mnack;
  // Frame gate: hand commands to the engine only once a whole frame (up to a STOP) is queued, or the
  // FIFO is full, so the engine never runs dry mid-frame. If it still does, it stalls SCL Low in the
  // 9th bit (I3C Basic §5.1.2.5) for at most TIM2.t_stall, then aborts the frame with P.
  logic [$clog2(CMD_DEPTH):0] stops_q;
  wire  cmd_go = !cmd_empty && (stops_q != '0 || cmd_full);
  assign cmd_rd = hcmd_ready && cmd_go;
  always_ff @(posedge clk_i or negedge rst_ni)
    if (!rst_ni) stops_q <= '0;
    else if (clr) stops_q <= '0;
    else begin
      logic inc, dec;
      inc = cmd_wr && !cmd_full && wdata_q[31:28] == 4'd2;
      dec = cmd_rd && cmd_q[31:28] == 4'd2;
      if (inc && !dec) stops_q <= stops_q + 1'b1;
      else if (dec && !inc) stops_q <= stops_q - 1'b1;
    end
  assign msg_rd = msg_ready && !msg_empty;
  assign rsp_wr = rsp_v;
  assign frame_o = !bus_free;

  i3c_mctp_gen u_gen (
    .clk_i, .rst_ni,
    .cfg_t_low_pp(tim0[7:0]), .cfg_t_high_pp(tim0[15:8]), .cfg_t_low_od(tim0[23:16]), .cfg_t_high_od(tim0[31:24]),
    .cfg_t_high_init(tim1[7:0]), .cfg_t_cas(tim1[15:8]), .cfg_t_cbp(tim1[23:16]), .cfg_t_pat(tim1[31:24]),
    .cfg_t_buf(tim2[15:0]), .cfg_t_stall(tim2[31:16]),
    .cfg_ibi_en(ctrl[0]), .cfg_ibi_ack(ctrl[1]), .cfg_ibi_bytes(ctrl[7:4]),
    .hcmd_valid_i(cmd_go), .hcmd_i(cmd_q), .hcmd_ready_o(hcmd_ready),
    .rsp_valid_o(rsp_v), .rsp_o(rsp_w),
    .mctp_start_i(mstart), .mctp_busy_o(mbusy),
    .mctp_da_i(mcfg0[6:0]), .mctp_dst_eid_i(mcfg1[7:0]), .mctp_src_eid_i(mcfg1[15:8]),
    .mctp_to_i(mcfg0[8]), .mctp_tag_i(mcfg0[11:9]), .mctp_seq_i(mcfg0[13:12]), .mctp_tu_i(mcfg0[31:24]),
    .mctp_len_i(mcfg1[31:16]), .mctp_hdr_ver_i(mcfg0[19:16]), .mctp_pec_corrupt_i(mcfg0[20]),
    .mctp_use_7e_i(mcfg0[21]), .mctp_seq_next_o(seq_next), .mctp_pkts_o(mpkts), .mctp_nack_o(mnack),
    .msg_valid_i(!msg_empty), .msg_data_i(msg_q), .msg_ready_o(msg_ready),
    .bus_free_o(bus_free),
    .scl_o, .sda_oe_o, .sda_o, .sda_pu_en_o, .sda_i);

  // ---------------- AXI4-Lite write ----------------
  logic       aw_hs, w_hs;
  logic [7:0] awaddr_q;  logic aw_have;
  logic        w_have;
  assign s_axi_awready = !aw_have && !s_axi_bvalid;
  assign s_axi_wready  = !w_have && !s_axi_bvalid;
  assign s_axi_bresp   = 2'b00;
  wire do_write = aw_have && w_have && !s_axi_bvalid;

  assign cmd_wr = do_write && awaddr_q == 8'h20;
  assign msg_wr = do_write && awaddr_q == 8'h40;
  assign mstart = mstart_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      aw_have <= 1'b0; w_have <= 1'b0; awaddr_q <= '0; wdata_q <= '0; s_axi_bvalid <= 1'b0;
      ctrl <= 32'h0000_0012;                       // IBI ACK, read 1 byte (MDB) after an IBI
      tim0 <= {8'd4, 8'd25, 8'd4, 8'd4};           // t_high_od, t_low_od, t_high_pp, t_low_pp
      tim1 <= {8'd4, 8'd3, 8'd5, 8'd25};           // t_pat, t_cbp, t_cas, t_high_init
      tim2 <= {16'd9000, 16'd100};                  // stall <= 90 us (Table 10: 100 us), bus free 1 us
      mcfg0 <= {8'd64, 2'b00, 1'b1, 1'b0, 4'h1, 2'b00, 2'b00, 3'd0, 1'b1, 1'b0, 7'h08};
      mcfg1 <= '0; mstart_q <= 1'b0; clr <= 1'b0; rsp_ovf <= 1'b0;
    end else begin
      clr <= 1'b0; mstart_q <= 1'b0;
      if (s_axi_awvalid && s_axi_awready) begin aw_have <= 1'b1; awaddr_q <= {s_axi_awaddr[7:2], 2'b00}; end
      if (s_axi_wvalid && s_axi_wready)   begin w_have  <= 1'b1; wdata_q  <= s_axi_wdata; end
      if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;
      if (do_write) begin
        aw_have <= 1'b0; w_have <= 1'b0; s_axi_bvalid <= 1'b1;
        unique case (awaddr_q)
          8'h04: begin ctrl <= {23'd0, 1'b0, wdata_q[7:0]}; if (wdata_q[8]) begin clr <= 1'b1; rsp_ovf <= 1'b0; end end
          8'h10: tim0 <= wdata_q;
          8'h14: tim1 <= wdata_q;
          8'h18: tim2 <= wdata_q;
          8'h30: mcfg0 <= wdata_q;
          8'h34: mcfg1 <= wdata_q;
          8'h38: mstart_q <= wdata_q[0];
          default: ;
        endcase
      end
      if (rsp_v && rsp_full) rsp_ovf <= 1'b1;
    end
  end
  // ---------------- AXI4-Lite read ----------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin s_axi_rvalid <= 1'b0; s_axi_rdata <= '0; end
    else begin
      if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 1'b0;
      if (s_axi_arvalid && s_axi_arready) begin
        s_axi_rvalid <= 1'b1;
        unique case ({s_axi_araddr[7:2], 2'b00})
          8'h00: s_axi_rdata <= 32'h13C3_0100;
          8'h04: s_axi_rdata <= ctrl;
          8'h08: s_axi_rdata <= {16'(rsp_cnt), 10'd0, msg_full, rsp_ovf, rsp_empty, cmd_full, mbusy, bus_free};
          8'h10: s_axi_rdata <= tim0;
          8'h14: s_axi_rdata <= tim1;
          8'h18: s_axi_rdata <= tim2;
          8'h24: s_axi_rdata <= rsp_empty ? 32'd0 : rsp_q;
          8'h30: s_axi_rdata <= mcfg0;
          8'h34: s_axi_rdata <= mcfg1;
          8'h38: s_axi_rdata <= {26'd0, seq_next, 3'd0, mbusy};
          8'h3C: s_axi_rdata <= {mnack, mpkts};
          default: s_axi_rdata <= 32'hDEAD_BEEF;
        endcase
      end
    end
  end
  assign s_axi_arready = !s_axi_rvalid;
  assign s_axi_rresp   = 2'b00;
  assign rsp_rd = s_axi_arvalid && s_axi_arready && ({s_axi_araddr[7:2], 2'b00} == 8'h24) && !rsp_empty;
endmodule
