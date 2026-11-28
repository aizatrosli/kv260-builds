// jepa_top: the unmodified tpuv1 tpu_core + a sequencer + the VPU, with the whole JEPA encoder
// hard-coded in ROMs. After reset and run_i, it executes the program ROM to the end: no host
// program load, no weight DMA. The only input is the image in the Unified Buffer.
//
//   program ROM --> sequencer --+--> tpu_core instruction port (tpuv1 ISA, incl. expanded X_LOADQ)
//                               +--> Weight FIFO <-- weight ROM (X_WROM) | UB tiles (X_WUB, transpose)
//                               +--> VPU <-> UB  (X_VEC)        LUT ROM -> VPU (X_LUT)
//                               +--> UB <-- weight ROM (X_UBROM: position embedding)
// Extension ops that touch the UB wait until the core has drained, then own the UB ports.
// Encodings: jepa/compile_jepa.py. Bit-exact spec: compile_jepa.Machine.
`timescale 1ns/1ps

module jepa_top #(
  parameter int unsigned N        = 16,
  parameter int unsigned PROG_D   = 16384,
  parameter int unsigned WROM_D   = 131072,
  parameter int unsigned PROM_D   = 16384,
  parameter int unsigned LROM_D   = 4096,
  parameter int unsigned WF_D     = 512,
  parameter bit          BOOT_LOAD = 1'b1,     // 1: weights arrive through the wload port
  parameter string PROG_INIT = "", WROM_INIT = "", PROM_INIT = "", LROM_INIT = ""
) (
  input  logic               clk_i,
  input  logic               rst_ni,
  input  logic               run_i,
  // Boot load of the weight/data memory (UltraRAM: it cannot be initialised by the bitstream).
  // At power-up a loader (on the board: iDMA reading the weight image the PS put in DDR) streams
  // the words in order; wload_last_i marks the final word. run_i is held off until then.
  input  logic               wload_valid_i,
  output logic               wload_ready_o,
  input  logic [127:0]       wload_data_i,
  input  logic               wload_last_i,
  output logic               weights_ready_o,
  // UB native ports (to tpu_fabric / unified_buffer)
  output logic               ub_rd_en_o,
  output logic [15:0]        ub_rd_addr_o,
  input  logic [N*8-1:0]     ub_rd_data_i,
  output logic               ub_wr_en_o,
  output logic [15:0]        ub_wr_addr_o,
  output logic [N*8-1:0]     ub_wr_data_o,
  output logic [N-1:0]       ub_wr_be_o,
  // ACC native ports
  output logic               acc_rd_en_o,
  output logic [10:0]        acc_rd_addr_o,
  input  logic [N*32-1:0]    acc_rd_data_i,
  output logic               acc_wr_en_o,
  output logic [10:0]        acc_wr_addr_o,
  output logic [N*32-1:0]    acc_wr_data_o,
  output logic [3:0]         acc_wr_be_o,
  // status
  output logic               busy_o,
  output logic               halted_o,
  output logic               cfg_err_o,
  output logic [7:0]         cfg_err_op_o,
  output logic               irq_o,
  output logic [31:0]        irq_tag_o,
  output logic [31:0]        perf_o [8],
  output logic [31:0]        seq_perf_o [4]    // {vec, wub, drain, ubrom} busy cycles
);
  localparam int unsigned UBLAT = 3;
  localparam logic [7:0] X_WROM = 8'h40, X_WUB = 8'h41, X_LOADQ = 8'h42, X_VEC = 8'h43,
                         X_LUT = 8'h44, X_VCFG = 8'h45, X_UBROM = 8'h46, X_MM = 8'h47;

  // ------------------------------------------------------------------ ROMs
  (* rom_style = "block" *) logic [127:0] prog_rom [PROG_D];
  (* ram_style = "ultra" *) logic [127:0] wrom     [WROM_D];     // URAM, filled at boot
  (* rom_style = "block" *) logic [71:0]  prom     [PROM_D];
  (* rom_style = "block" *) logic [15:0]  lrom     [LROM_D];
  initial begin
    if (PROG_INIT != "") $readmemh(PROG_INIT, prog_rom);
    if (WROM_INIT != "") $readmemh(WROM_INIT, wrom);
    if (PROM_INIT != "") $readmemh(PROM_INIT, prom);
    if (LROM_INIT != "") $readmemh(LROM_INIT, lrom);
  end

  // ------------------------------------------------------------------ core
  logic         c_if_valid, c_if_ready;
  logic [127:0] c_if_data;
  logic         c_wf_valid, c_wf_ready;
  logic [127:0] c_wf_data;
  logic         c_ub_rd_en, c_ub_wr_en, core_busy;
  logic [15:0]  c_ub_rd_addr, c_ub_wr_addr;
  logic [N*8-1:0] c_ub_wr_data;
  logic [N-1:0] c_ub_wr_be;

  tpu_core #(.N(N)) i_core (
    .clk_i, .rst_ni,
    .if_valid_i(c_if_valid), .if_ready_o(c_if_ready), .if_data_i(c_if_data),
    .wf_valid_i(c_wf_valid), .wf_ready_o(c_wf_ready), .wf_data_i(c_wf_data),
    .ub_rd_en_o(c_ub_rd_en), .ub_rd_addr_o(c_ub_rd_addr), .ub_rd_data_i(ub_rd_data_i),
    .ub_wr_en_o(c_ub_wr_en), .ub_wr_addr_o(c_ub_wr_addr), .ub_wr_data_o(c_ub_wr_data), .ub_wr_be_o(c_ub_wr_be),
    .acc_rd_en_o, .acc_rd_addr_o, .acc_rd_data_i, .acc_wr_en_o, .acc_wr_addr_o, .acc_wr_data_o, .acc_wr_be_o,
    .run_i, .err_clr_i(1'b0), .perf_clr_i(1'b0),
    .dma_req_valid_o(), .dma_req_ready_i(1'b0), .dma_src_addr_o(), .dma_dst_addr_o(),
    .dma_length_o(), .dma_reps_o(), .dma_src_stride_o(), .dma_dst_stride_o(),
    .dma_rsp_valid_i(1'b0), .dma_rsp_ready_o(), .dma_rsp_error_i(1'b0),
    .dma_bus_error_i(1'b0), .dma_error_o(),
    .busy_o(core_busy), .halted_o, .cfg_err_o, .cfg_err_op_o, .irq_o, .irq_tag_o,
    .debug_tag_o(), .perf_o
  );

  // ------------------------------------------------------------------ Weight FIFO
  logic [127:0] wf_mem [WF_D];
  logic [$clog2(WF_D):0] wf_cnt;
  logic [$clog2(WF_D)-1:0] wf_wp, wf_rp;
  logic wf_push; logic [127:0] wf_pdata;
  wire  wf_pop = c_wf_valid && c_wf_ready;
  assign c_wf_valid = (wf_cnt != 0);
  assign c_wf_data  = wf_mem[wf_rp];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin wf_cnt <= '0; wf_wp <= '0; wf_rp <= '0; end
    else begin
      if (wf_push) begin wf_mem[wf_wp] <= wf_pdata; wf_wp <= wf_wp + 1'b1; end
      if (wf_pop) wf_rp <= wf_rp + 1'b1;
      wf_cnt <= wf_cnt + (wf_push ? 1 : 0) - (wf_pop ? 1 : 0);
    end
  end

  // ------------------------------------------------------------------ VPU
  logic        v_cmd_valid, v_cmd_ready, v_in_valid, v_in_ready, v_out_valid, v_busy;
  logic [1:0]  v_op;
  logic [5:0]  v_rows;
  logic signed [17:0] v_ma, v_mb;
  logic        v_lut_we, v_lut_sel;
  logic [7:0]  v_lut_addr;
  logic [15:0] v_lut_wdata;
  logic [N*8-1:0] v_in_a, v_in_b, v_out;
  vpu #(.N(N), .MAXR(32)) i_vpu (
    .clk_i, .rst_ni,
    .cmd_valid_i(v_cmd_valid), .cmd_ready_o(v_cmd_ready), .cmd_op_i(v_op), .cmd_rows_i(v_rows),
    .cmd_ma_i(v_ma), .cmd_mb_i(v_mb),
    .lut_we_i(v_lut_we), .lut_sel_i(v_lut_sel), .lut_addr_i(v_lut_addr), .lut_wdata_i(v_lut_wdata),
    .in_valid_i(v_in_valid), .in_ready_o(v_in_ready), .in_a_i(v_in_a), .in_b_i(v_in_b),
    .out_valid_o(v_out_valid), .out_ready_i(1'b1), .out_data_o(v_out), .busy_o(v_busy));

  // ------------------------------------------------------------------ sequencer
  typedef enum logic [4:0] {S_IDLE, S_FETCH, S_DISP, S_LQ_RD, S_LQ_M, S_LQ_B, S_LQ_Z, S_DRAIN,
                            S_UBROM, S_LUT, S_WUB_RD, S_WUB_PUSH, S_VEC_CMD, S_VEC_RD, S_VEC_WAIT,
                            S_VEC_OUT, S_MM_SYNC, S_MM_MAT, S_MM_SYNCM, S_MM_ACT, S_DONE} st_e;
  st_e st;
  logic [$clog2(PROG_D)-1:0] pc;
  logic [127:0] pw;                         // current program word
  wire  [7:0]   pop_ = pw[7:0];

  // weight ROM pointer + background stream
  logic [$clog2(WROM_D)-1:0] rp;
  logic [23:0]  wpend;
  logic         ws_v;                       // a ROM word read last cycle, push it now
  logic [127:0] wrom_q;
  logic [$clog2(WROM_D)-1:0] wrom_a;
  logic         wrom_re;
  always_ff @(posedge clk_i) if (wrom_re) wrom_q <= wrom[wrom_a];

  // boot loader write port
  logic [$clog2(WROM_D)-1:0] la;
  logic wloaded;
  assign wload_ready_o   = !wloaded;
  assign weights_ready_o = wloaded;
  always_ff @(posedge clk_i) if (wload_valid_i && !wloaded) wrom[la] <= wload_data_i;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin la <= '0; wloaded <= (WROM_INIT != "") || !BOOT_LOAD; end
    else if (wload_valid_i && !wloaded) begin la <= la + 1'b1; if (wload_last_i) wloaded <= 1'b1; end
  end

  // param / LUT ROM reads
  logic [$clog2(PROM_D)-1:0] prom_a;
  logic [71:0]  prom_q;
  always_ff @(posedge clk_i) prom_q <= prom[prom_a];
  logic [$clog2(LROM_D)-1:0] lrom_a;
  logic [15:0]  lrom_q;
  always_ff @(posedge clk_i) lrom_q <= lrom[lrom_a];

  // fields
  logic [4:0]   c;                          // LOADQ channel
  logic [1:0]   lq_q;
  logic [15:0]  f_base, f_b, f_o, f_stride, f_cnt, f_rstride;
  logic [7:0]   f_kt, f_nt, f_vstride;
  logic         f_tr;
  logic [1:0]   f_op;
  logic [5:0]   f_rows;
  logic [7:0]   kt, nt;
  logic [15:0]  i, v, vo;                   // generic counters (vo: vector being written back)
  logic [5:0]   r, ro;
  logic [2:0]   drain_ok;
  logic [N*8-1:0] tbuf [16];
  logic [UBLAT:0] rdv;                       // read-valid pipeline
  logic [4:0]   rdi [UBLAT+1];               // index pipeline (tile row / b-flag)
  logic         seq_ub;                     // sequencer owns the UB ports
  logic         s_rd_en, s_wr_en;
  logic [15:0]  s_rd_addr, s_wr_addr;
  logic [N*8-1:0] s_wr_data;
  logic         got_a, got_b;
  // X_MM: one output tile = [SYNC act] + KT x MATMUL + LOADQ + SYNC matrix + ACTIVATE
  logic         mm_mode;
  logic [1:0]   mm_q;
  logic [5:0]   mm_kt, kcnt;
  logic [15:0]  mm_in, mm_out, mm_B;
  logic [10:0]  mm_acc;

  // UB port mux
  assign ub_rd_en_o   = seq_ub ? s_rd_en   : c_ub_rd_en;
  assign ub_rd_addr_o = seq_ub ? s_rd_addr : c_ub_rd_addr;
  assign ub_wr_en_o   = seq_ub ? s_wr_en   : c_ub_wr_en;
  assign ub_wr_addr_o = seq_ub ? s_wr_addr : c_ub_wr_addr;
  assign ub_wr_data_o = seq_ub ? s_wr_data : c_ub_wr_data;
  assign ub_wr_be_o   = seq_ub ? '1        : c_ub_wr_be;
  assign busy_o       = (st != S_IDLE) && (st != S_DONE) || core_busy;

  // core instruction port
  logic [127:0] lq_word;
  always_comb begin
    c_if_valid = 1'b0; c_if_data = pw;
    case (st)
      S_DISP:  begin c_if_valid = (pop_ < 8'h40); c_if_data = pw; end
      S_LQ_M:  begin c_if_valid = 1'b1; c_if_data = {32'd0, 64'(prom_q[39:0]),  4'h1, 6'd0, lq_q, c[3:0], 8'h00, 8'h30}; end
      S_LQ_B:  begin c_if_valid = 1'b1; c_if_data = {32'd0, 64'(prom_q[71:40]), 4'h2, 6'd0, lq_q, c[3:0], 8'h00, 8'h30}; end
      S_MM_SYNC:  begin c_if_valid = 1'b1; c_if_data = 128'h831; end                       // SYNC U_ACT
      S_MM_SYNCM: begin c_if_valid = 1'b1; c_if_data = 128'h431; end                       // SYNC U_MATRIX
      S_MM_MAT:   begin c_if_valid = 1'b1;
                        c_if_data = {64'd0, mm_B, 5'd0, mm_acc, mm_in + 16'(kcnt) * mm_B, 6'd0, 1'b1, (kcnt != 0), 8'h10}; end
      S_MM_ACT:   begin c_if_valid = 1'b1;
                        c_if_data = {60'd0, 2'd0, mm_q, mm_B, mm_out, 5'd0, mm_acc, 8'h00, 8'h20}; end
      S_LQ_Z:  begin c_if_valid = 1'b1; c_if_data = {32'd0, 64'(prom_q[23:0]),  4'h3, 10'd0, lq_q, 8'h00, 8'h30}; end
      default: ;
    endcase
  end

  // weight ROM port: the background stream uses it unless an X_UBROM copy runs
  wire ws_can = (wpend != 0) && (wf_cnt < WF_D - 4) && (st != S_UBROM);
  always_comb begin
    wrom_re = 1'b0; wrom_a = rp;
    if (st == S_UBROM && i < f_cnt) wrom_re = 1'b1;
    else if (ws_can)                wrom_re = 1'b1;
  end

  // Weight FIFO push mux: ROM stream (registered read) or UB tile words
  logic        wub_push;
  logic [127:0] wub_word;
  always_comb begin
    wf_push = ws_v || wub_push; wf_pdata = ws_v ? wrom_q : wub_word;
  end
  // UB tile -> word (t = 15 - row); transpose: byte j of row i = tbuf[j][i]
  always_comb begin
    logic [3:0] row;
    row = 4'(15 - i);
    wub_push = (st == S_WUB_PUSH) && (wf_cnt < WF_D - 4) && !ws_v;
    for (int j = 0; j < 16; j++)
      wub_word[8*j +: 8] = f_tr ? tbuf[j][8*row +: 8] : tbuf[row][8*j +: 8];
  end

  // VPU wiring
  assign v_op = f_op; assign v_rows = f_rows;
  logic signed [17:0] ma_q, mb_q;
  logic signed [17:0] sq_q;                  // softmax numerator (X_VCFG sq)
  assign v_ma = (f_op == 2'd3) ? sq_q : ma_q; assign v_mb = mb_q;

  // perf
  logic [31:0] pf [4];
  assign seq_perf_o = pf;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      st <= S_IDLE; pc <= '0; pw <= '0; rp <= '0; wpend <= '0; ws_v <= 1'b0;
      seq_ub <= 1'b0; s_rd_en <= 1'b0; s_wr_en <= 1'b0; s_rd_addr <= '0; s_wr_addr <= '0; s_wr_data <= '0;
      prom_a <= '0; lrom_a <= '0; c <= '0; lq_q <= '0; drain_ok <= '0; rdv <= '0;
      v_cmd_valid <= 1'b0; v_in_valid <= 1'b0; v_lut_we <= 1'b0; v_lut_sel <= 1'b0; v_lut_addr <= '0; v_lut_wdata <= '0;
      ma_q <= '0; mb_q <= '0; sq_q <= 18'sd127; i <= '0; v <= '0; vo <= '0; r <= '0; ro <= '0; kt <= '0; nt <= '0;
      f_base <= '0; f_b <= '0; f_o <= '0; f_stride <= '0; f_cnt <= '0; f_rstride <= '0;
      f_kt <= '0; f_nt <= '0; f_vstride <= '0; f_tr <= 1'b0; f_op <= '0; f_rows <= '0;
      got_a <= 1'b0; got_b <= 1'b0; mm_mode <= 1'b0; mm_q <= '0; mm_kt <= '0; kcnt <= '0;
      mm_in <= '0; mm_out <= '0; mm_B <= '0; mm_acc <= '0;
      foreach (pf[k]) pf[k] <= '0;
    end else begin
      // background weight stream (registered ROM read -> FIFO push next cycle)
      ws_v <= 1'b0;
      if (ws_can && !(st == S_UBROM && i < f_cnt)) begin ws_v <= 1'b1; rp <= rp + 1'b1; wpend <= wpend - 1'b1; end
      s_rd_en <= 1'b0; s_wr_en <= 1'b0; v_lut_we <= 1'b0;
      rdv <= {rdv[UBLAT-1:0], s_rd_en};
      for (int k = UBLAT; k > 0; k--) rdi[k] <= rdi[k-1];
      if (st == S_VEC_CMD || st == S_VEC_RD || st == S_VEC_WAIT || st == S_VEC_OUT) pf[0] <= pf[0] + 1;
      if (st == S_WUB_RD || st == S_WUB_PUSH) pf[1] <= pf[1] + 1;
      if (st == S_DRAIN) pf[2] <= pf[2] + 1;
      if (st == S_UBROM || st == S_LUT) pf[3] <= pf[3] + 1;

      // VPU output rows -> UB (any VEC state)
      if (v_out_valid) begin
        s_wr_en <= 1'b1; s_wr_addr <= f_o + vo * 16'(f_vstride) + 16'(ro) * f_rstride; s_wr_data <= v_out;
        if (ro + 1'b1 == f_rows) begin ro <= '0; vo <= vo + 1'b1; end else ro <= ro + 1'b1;
      end

      case (st)
        S_IDLE: if (run_i && wloaded) st <= S_FETCH;
        S_FETCH: begin seq_ub <= 1'b0; pw <= prog_rom[pc]; pc <= pc + 1'b1; st <= S_DISP; end
        S_DISP: begin
          if (pop_ < 8'h40) begin
            if (c_if_ready) st <= (pop_ == 8'h3F) ? S_DONE : S_FETCH;
          end else case (pop_)
            X_WROM:  begin wpend <= wpend + pw[31:8] - ((ws_can && !(st == S_UBROM)) ? 24'd1 : 24'd0); st <= S_FETCH; end
            X_VCFG:  begin ma_q <= pw[25:8]; mb_q <= pw[49:32]; sq_q <= {4'd0, pw[63:50]}; st <= S_FETCH; end
            X_LOADQ: begin lq_q <= pw[9:8]; c <= '0; prom_a <= 14'(pw[31:16] * 17); st <= S_LQ_RD; end
            X_MM:    begin
              lq_q <= pw[9:8]; mm_q <= pw[9:8]; c <= '0; prom_a <= 14'(pw[75:60] * 17); mm_mode <= 1'b1;
              mm_kt <= pw[16:11]; mm_in <= pw[32:17]; mm_acc <= pw[43:33]; mm_out <= pw[59:44]; mm_B <= pw[91:76];
              kcnt <= '0;
              st <= pw[10] ? S_MM_SYNC : S_MM_MAT;
            end
            default: begin                 // UB / VPU ops: drain first
              f_tr <= pw[8]; f_base <= pw[31:16]; f_stride <= pw[47:32]; f_kt <= pw[55:48]; f_nt <= pw[63:56];
              f_op <= pw[9:8]; f_rows <= pw[15:10]; f_b <= pw[47:32]; f_o <= pw[63:48];
              f_cnt <= (pop_ == X_UBROM) ? pw[47:32] : pw[79:64];
              f_rstride <= pw[95:80]; f_vstride <= pw[103:96];
              drain_ok <= '0; st <= S_DRAIN;
            end
          endcase
        end
        S_LQ_RD: st <= (c == 5'd16) ? S_LQ_Z : S_LQ_M;          // prom_q valid next cycle
        S_LQ_M:  if (c_if_ready) st <= S_LQ_B;
        S_LQ_B:  if (c_if_ready) begin c <= c + 1'b1; prom_a <= prom_a + 1'b1; st <= S_LQ_RD; end
        S_LQ_Z:  if (c_if_ready) st <= mm_mode ? S_MM_SYNCM : S_FETCH;
        S_MM_SYNC:  if (c_if_ready) st <= S_MM_MAT;
        S_MM_MAT:   if (c_if_ready) begin
          kcnt <= kcnt + 1'b1;
          if (kcnt + 1'b1 == mm_kt) st <= S_LQ_RD;          // LOADQ after the matmuls (overlaps)
        end
        S_MM_SYNCM: if (c_if_ready) st <= S_MM_ACT;
        S_MM_ACT:   if (c_if_ready) begin mm_mode <= 1'b0; st <= S_FETCH; end
        S_DRAIN: begin
          // wait: background stream done, FIFO push settled, core idle for 4 cycles
          if (wpend == 0 && !ws_v && !core_busy) drain_ok <= drain_ok + 1'b1; else drain_ok <= '0;
          if (drain_ok == 3'd4) begin
            i <= '0; v <= '0; vo <= '0; r <= '0; ro <= '0; kt <= '0; nt <= '0; got_a <= 1'b0; got_b <= 1'b0;
            case (pop_)
              X_UBROM: begin seq_ub <= 1'b1; st <= S_UBROM; end
              X_LUT:   begin lrom_a <= 12'(pw[23:16] * 256); v_lut_sel <= pw[8]; st <= S_LUT; end
              X_WUB:   begin seq_ub <= 1'b1; st <= S_WUB_RD; end
              default: begin seq_ub <= 1'b1; st <= S_VEC_CMD; end
            endcase
          end
        end
        S_UBROM: begin
          // read wrom[rp] in cycle i, write UB base+i-1 in cycle i+1
          if (i != 0) begin s_wr_en <= 1'b1; s_wr_addr <= f_base + i - 1'b1; s_wr_data <= wrom_q; end
          if (i < f_cnt) begin rp <= rp + 1'b1; i <= i + 1'b1; end
          else st <= S_FETCH;
        end
        S_LUT: begin
          // lrom_q (address i) is valid one cycle after lrom_a
          if (i != 0) begin v_lut_we <= 1'b1; v_lut_addr <= 8'(i - 1); v_lut_wdata <= lrom_q; end
          if (i < 256) begin lrom_a <= lrom_a + 1'b1; i <= i + 1'b1; end
          else st <= S_FETCH;
        end
        // ---------------- X_WUB: per tile (nt outer, kt inner) read 16 rows, push 16 words
        S_WUB_RD: begin
          if (i < 16) begin
            s_rd_en <= 1'b1;
            s_rd_addr <= f_base + (f_tr ? 16'(kt) : 16'(nt)) * f_stride + 16'((f_tr ? 16'(nt) : 16'(kt)) * 16) + i;
            rdi[0] <= 5'(i);
            i <= i + 1'b1;
          end
          if (rdv[UBLAT-1]) tbuf[rdi[UBLAT][3:0]] <= ub_rd_data_i;
          if (rdv[UBLAT-1] && rdi[UBLAT] == 5'd15) begin i <= '0; st <= S_WUB_PUSH; end
        end
        S_WUB_PUSH: if (wub_push) begin
          if (i == 15) begin
            i <= '0;
            if (kt + 1'b1 < f_kt) begin kt <= kt + 1'b1; st <= S_WUB_RD; end
            else begin
              kt <= '0;
              if (nt + 1'b1 < f_nt) begin nt <= nt + 1'b1; st <= S_WUB_RD; end
              else begin seq_ub <= 1'b0; st <= S_FETCH; end
            end
          end else i <= i + 1'b1;
        end
        // ---------------- X_VEC: per vector: command, rows in (a [, b]), rows out
        S_VEC_CMD: begin
          if (v == f_cnt) begin
            if (vo == f_cnt && !v_busy && !v_out_valid && !s_wr_en) st <= S_FETCH;
          end else if (!v_cmd_valid && v_cmd_ready) begin
            v_cmd_valid <= 1'b1;
          end else if (v_cmd_valid) begin             // accepted at this edge (ready was high)
            v_cmd_valid <= 1'b0; r <= '0; i <= '0; st <= S_VEC_RD;
          end
        end
        S_VEC_RD: begin                                // stream: one UB read per cycle (a,b alternate for ADD)
          logic [15:0] nrd, rrow;
          nrd  = (f_op == 2'd0) ? 16'(f_rows) * 2 : 16'(f_rows);
          rrow = (f_op == 2'd0) ? (i >> 1) : i;
          if (i < nrd) begin
            s_rd_en <= 1'b1;
            s_rd_addr <= ((f_op == 2'd0 && i[0]) ? f_b : f_base) + v * 16'(f_vstride) + rrow * f_rstride;
            rdi[0] <= {4'd0, (f_op == 2'd0) && i[0]};
            i <= i + 1'b1;
          end
          v_in_valid <= 1'b0;
          if (rdv[UBLAT-1]) begin
            if (f_op == 2'd0 && !rdi[UBLAT][0]) v_in_a <= ub_rd_data_i;           // ADD: hold a
            else begin
              if (f_op == 2'd0) v_in_b <= ub_rd_data_i; else v_in_a <= ub_rd_data_i;
              v_in_valid <= 1'b1;                                                  // VPU LOAD never stalls
              r <= r + 1'b1;
            end
          end
          if (v_in_valid && r == 6'(f_rows)) begin v_in_valid <= 1'b0; v <= v + 1'b1; st <= S_VEC_CMD; end
        end
        S_VEC_OUT: if (ro == f_rows || (v_out_valid && ro + 1'b1 == f_rows)) begin
          v <= v + 1'b1; st <= S_VEC_CMD;
        end
        S_DONE: ;
        default: st <= S_IDLE;
      endcase
    end
  end
endmodule
