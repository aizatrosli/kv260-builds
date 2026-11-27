// tb_vpu: replays every VPU job captured from one JEPA encoder inference (vpu_vectors.py) and
// compares each output row bit-exactly. Random input/output stalls exercise the handshakes.
//   +VEC=<dir>  +STALL_PCT=<p>  +SEED=<s>
`timescale 1ns/1ps

module tb_vpu;
  localparam int unsigned N = 16, MAXR = 32;
  localparam int unsigned MAXJ = 1 << 16, MAXI = 1 << 18;

  logic clk = 0, rst_n = 0;
  always #5 clk = ~clk;

  logic cmd_valid = 0, cmd_ready;
  logic [1:0] cmd_op;
  logic [5:0] cmd_rows;
  logic signed [17:0] cmd_ma, cmd_mb;
  logic lut_we = 0, lut_sel = 0;
  logic [7:0] lut_addr;
  logic [15:0] lut_wdata;
  logic in_valid = 0, in_ready, out_valid, out_ready = 0, busy;
  logic [N*8-1:0] in_a, in_b, out_data;

  vpu #(.N(N), .MAXR(MAXR)) dut (
    .clk_i(clk), .rst_ni(rst_n),
    .cmd_valid_i(cmd_valid), .cmd_ready_o(cmd_ready), .cmd_op_i(cmd_op), .cmd_rows_i(cmd_rows),
    .cmd_ma_i(cmd_ma), .cmd_mb_i(cmd_mb),
    .lut_we_i(lut_we), .lut_sel_i(lut_sel), .lut_addr_i(lut_addr), .lut_wdata_i(lut_wdata),
    .in_valid_i(in_valid), .in_ready_o(in_ready), .in_a_i(in_a), .in_b_i(in_b),
    .out_valid_o(out_valid), .out_ready_i(out_ready), .out_data_o(out_data), .busy_o(busy));

  logic [63:0]  cmdm [MAXJ];
  logic [255:0] inm  [MAXI];
  logic [127:0] expm [MAXI];
  int jobs, in_rows, exp_rows, stall_pct = 25, seed = 1;
  int errors = 0, checked = 0, ip = 0, ep = 0;
  int cnt_op [4];
  longint cycles = 0, busy_cycles = 0;
  string vec;

  always @(posedge clk) begin cycles++; if (busy) busy_cycles++; end

  initial begin
    int fd, v; string key; int r;
    if (!$value$plusargs("VEC=%s", vec)) vec = "build/vpu";
    void'($value$plusargs("STALL_PCT=%d", stall_pct));
    void'($value$plusargs("SEED=%d", seed));
    void'($urandom(seed));
    fd = $fopen({vec, "/meta.txt"}, "r");
    if (fd == 0) $fatal(1, "no %s/meta.txt", vec);
    while ($fscanf(fd, "%s %d", key, v) == 2)
      case (key) "jobs": jobs = v; "in_rows": in_rows = v; "exp_rows": exp_rows = v; default: ; endcase
    $fclose(fd);
    $readmemh({vec, "/cmd.hex"}, cmdm, 0, jobs - 1);
    $readmemh({vec, "/in.hex"},  inm,  0, in_rows - 1);
    $readmemh({vec, "/exp.hex"}, expm, 0, exp_rows - 1);
    foreach (cnt_op[i]) cnt_op[i] = 0;
    repeat (3) @(posedge clk); rst_n = 1; @(posedge clk);

    for (int j = 0; j < jobs; j++) begin
      logic [63:0] c; c = cmdm[j];
      if (c[63]) begin                                  // LUT load
        for (int k = 0; k < 256; k++) begin
          @(negedge clk); lut_we = 1; lut_sel = c[62]; lut_addr = 8'(k); lut_wdata = inm[ip][15:0]; ip++;
        end
        @(negedge clk); lut_we = 0;
        continue;
      end
      @(negedge clk);
      cmd_valid = 1; cmd_op = c[1:0]; cmd_rows = c[13:8]; cmd_ma = c[35:18]; cmd_mb = c[53:36];
      #1 while (!cmd_ready) begin @(negedge clk); #1; end
      @(negedge clk); cmd_valid = 0;
      cnt_op[c[1:0]]++;
      fork
        begin                                           // input rows
          for (r = 0; r < c[13:8]; r++) begin
            @(negedge clk);
            while ($urandom_range(99) < stall_pct) begin in_valid = 0; @(negedge clk); end
            in_valid = 1; in_a = inm[ip][127:0]; in_b = inm[ip][255:128]; ip++;
            #1 while (!in_ready) begin @(negedge clk); #1; end
          end
          @(negedge clk); in_valid = 0;
        end
        begin                                           // output rows
          int got; got = 0;
          while (got < c[13:8]) begin
            @(negedge clk); out_ready = ($urandom_range(99) >= stall_pct);
            #1;
            if (out_valid && out_ready) begin
              if (out_data !== expm[ep]) begin
                errors++;
                if (errors <= 10) $display("[ERROR] job %0d op %0d row %0d: got %h exp %h", j, c[1:0], got, out_data, expm[ep]);
              end
              ep++; got++; checked++;
            end
          end
          @(negedge clk); out_ready = 0;
        end
      join
    end
    $display("tb_vpu: %0d jobs (add %0d, gelu %0d, layernorm %0d, softmax %0d), %0d rows checked, errors=%0d, cycles=%0d busy=%0d",
             jobs, cnt_op[0], cnt_op[1], cnt_op[2], cnt_op[3], checked, errors, cycles, busy_cycles);
    if (errors == 0 && checked == exp_rows) $display("VPU_PASS"); else $display("VPU_FAIL");
    $finish;
  end
endmodule
