// Bus timing monitor for a pure I3C bus (no legacy I2C devices), simulation only.
// Limits from MIPI I3C Basic v1.1.1, Table 86 (Open Drain) and Table 87 (Push-Pull).
// Lines are ideal in simulation (zero rise/fall), so measured times equal digital times (tDIG_*).
//   od_i: 1 while the Controller marks the phase as Open Drain (sda_pu_en).
`timescale 1ns/1ps
module i3c_timing_mon (
  input logic scl,
  input logic sda,
  input logic od_i,
  input logic ctrl_drv_i      // Controller is driving SDA
);
  // Table 86
  localparam real T_LOW_OD_MIN   = 200.0;
  localparam real T_HIGH_INIT_MIN= 200.0;
  localparam real T_CAS_MIN      = 38.4;
  localparam real T_CBP_MIN      = T_CAS_MIN / 2;   // tCBP >= tCAS(min)/2
  // Table 87
  localparam real T_LOW_MIN      = 24.0;
  localparam real T_HIGH_MIN     = 24.0;
  localparam real T_DIG_L_MIN    = 32.0;
  localparam real T_DIG_H_MIN    = 32.0;
  localparam real F_SCL_MAX_MHZ  = 12.9;
  localparam real T_CASR_MIN     = T_CAS_MIN / 2;   // tCASr
  localparam real T_CBSR_MIN     = T_CAS_MIN / 2;   // tCBSr
  localparam real T_SU_PP_MIN    = 3.0;
  // Clock stalling is only allowed with SCL Low, at most 100 us (§5.1.2.5, Table 10). tCAS max is 1 us
  // in activity state ENTAS0 (Table 86). SCL High inside a frame longer than this is flagged.
  localparam real T_HIGH_STALL_MAX = 1000.0;
  localparam real T_LOW_STALL_MAX  = 100000.0;
  localparam real T_CAS_MAX_AS0    = 1000.0;
  real max_low = 0, max_high_in_frame = 0;

  int errors = 0;
  // SDA toggles during one SCL low period (> 2 means a pattern: HDR Exit / Target Reset)
  int      tog = 0;
  int      pat_counts[$];
  realtime t_tog = 0;
  logic    tog_lvl0 = 1'b1;   // SDA level at the SCL fall; a pattern starts from SDA High
  localparam real T_PAT_MIN = 32.0;   // tDIG_H min (Table 87), HDR Exit edge spacing §5.2.1.1.1
  int n_start = 0, n_rstart = 0, n_stop = 0, n_bits = 0;
  real min_low_od = 1e9, min_low_pp = 1e9, min_high = 1e9, min_period_pp = 1e9;
  real min_cas = 1e9, min_casr = 1e9, min_cbp = 1e9, min_cbsr = 1e9, min_buf = 1e9;

  realtime t_scl_fall = 0, t_scl_rise = 0, t_sda_chg = 0, t_start = 0, t_stop = -1;
  logic    low_od = 1'b0;
  logic    pending_start = 1'b0, pending_rstart = 1'b0;
  logic    in_frame = 1'b0;
  logic    first_hdr = 1'b1;          // tHIGH_INIT applies to the first broadcast after init
  logic    first_hdr_active = 1'b0;
  int      first_hdr_bits = 0;
  real     last_low = 0;

  task automatic fail(input string m);
    errors++;
    $display("[%0t] TIMING VIOLATION: %s", $time, m);
  endtask

  always @(negedge scl) begin
    realtime t; t = $realtime;
    if (scl === 1'b0 && t_scl_rise > 0) begin
      real h; h = t - t_scl_rise;
      if (in_frame && !pending_start && !pending_rstart && h > max_high_in_frame) max_high_in_frame = h;
      if (in_frame && !pending_start && !pending_rstart && h > T_HIGH_STALL_MAX)
        fail($sformatf("SCL held High %.1f ns inside a frame (stalls must be SCL Low, §5.1.2.5)", h));
      if (!pending_start && !pending_rstart) begin
        if (h < min_high) min_high = h;
        if (h < T_HIGH_MIN || h < T_DIG_H_MIN) fail($sformatf("tHIGH/tDIG_H %.1f ns", h));
        if (first_hdr_active && h < T_HIGH_INIT_MIN) fail($sformatf("tHIGH_INIT %.1f ns", h));
        if (!low_od) begin
          real p; p = last_low + h;
          if (p < min_period_pp) min_period_pp = p;
          if (p < 1000.0 / F_SCL_MAX_MHZ) fail($sformatf("PP SCL period %.1f ns (> %.1f MHz)", p, F_SCL_MAX_MHZ));
        end
        if (first_hdr_active) begin
          first_hdr_bits = first_hdr_bits + 1;
          if (first_hdr_bits >= 8) begin first_hdr_active = 1'b0; first_hdr = 1'b0; end
        end
      end
    end
    if (pending_start) begin
      real c; c = t - t_start;
      if (c < min_cas) min_cas = c;
      if (c < T_CAS_MIN) fail($sformatf("tCAS %.1f ns", c));
      if (c > T_CAS_MAX_AS0) fail($sformatf("tCAS %.1f ns > 1 us (ENTAS0 max)", c));
      if (first_hdr) begin first_hdr_active = 1'b1; first_hdr_bits = 0; end
    end
    if (pending_rstart) begin
      real c; c = t - t_start;
      if (c < min_casr) min_casr = c;
      if (c < T_CASR_MIN) fail($sformatf("tCASr %.1f ns", c));
    end
    pending_start = 1'b0; pending_rstart = 1'b0;
    t_scl_fall = t;
    low_od = od_i;
    tog = 0; tog_lvl0 = sda;
  end

  always @(posedge scl) begin
    realtime t; real l;
    t = $realtime; l = t - t_scl_fall;
    low_od = low_od | od_i;
    last_low = l;
    if (in_frame && l > max_low) max_low = l;
    if (in_frame && l > T_LOW_STALL_MAX) fail($sformatf("SCL Low stall %.1f ns > 100 us", l));
    if (in_frame) begin
      n_bits++;
      if (low_od) begin
        if (l < min_low_od) min_low_od = l;
        if (l < T_LOW_OD_MIN) fail($sformatf("tLOW_OD %.1f ns", l));
      end else begin
        if (l < min_low_pp) min_low_pp = l;
        if (l < T_LOW_MIN || l < T_DIG_L_MIN) fail($sformatf("tLOW/tDIG_L %.1f ns", l));
      end
      if (t - t_sda_chg < T_SU_PP_MIN && t_sda_chg > t_scl_fall)
        fail($sformatf("tSU %.1f ns", t - t_sda_chg));
    end
    if (tog > 2) pat_counts.push_back(tog - (tog_lvl0 ? 0 : 1));   // exclude the set-up edge to High
    tog = 0;
    t_scl_rise = t;
  end

  always @(sda) begin
    realtime t; t = $realtime;
    if (scl === 1'b1) begin
      if (sda === 1'b0) begin
        // START or Repeated START
        if (!in_frame) begin
          n_start++; pending_start = 1'b1;
          if (t_stop >= 0) begin
            real b; b = t - t_stop;
            if (b < min_buf) min_buf = b;
            if (b < T_CAS_MIN) fail($sformatf("bus free %.1f ns", b));
          end
        end else begin
          real c; c = t - t_scl_rise;
          n_rstart++; pending_rstart = 1'b1;
          if (c < min_cbsr) min_cbsr = c;
          if (c < T_CBSR_MIN) fail($sformatf("tCBSr %.1f ns", c));
        end
        in_frame = 1'b1; t_start = t;
      end else if (sda === 1'b1 && in_frame) begin
        real c; c = t - t_scl_rise;
        n_stop++;
        if (c < min_cbp) min_cbp = c;
        if (c < T_CBP_MIN) fail($sformatf("tCBP %.1f ns", c));
        in_frame = 1'b0; t_stop = t;
      end
    end else begin
      // SDA change while SCL low: Controller hold after SCL fall must be > 0 (tHD_PP, Table 87)
      if (ctrl_drv_i && (t - t_scl_fall) < 3.0) fail($sformatf("Controller SDA hold %.1f ns", t - t_scl_fall));
      if (ctrl_drv_i) begin
        if (tog > 0 && (t - t_tog) < T_PAT_MIN) fail($sformatf("pattern edge spacing %.1f ns", t - t_tog));
        tog++; t_tog = t;
      end
    end
    t_sda_chg = t;
  end

  function automatic void report();
    $display("timing: S=%0d Sr=%0d P=%0d bits=%0d", n_start, n_rstart, n_stop, n_bits);
    $display("timing: min tLOW_OD=%.1f (>=%.1f)  tLOW=%.1f (>=%.1f)  tHIGH=%.1f (>=%.1f)  PP period=%.1f (>=%.1f)",
             min_low_od, T_LOW_OD_MIN, min_low_pp, T_DIG_L_MIN, min_high, T_DIG_H_MIN, min_period_pp, 1000.0 / F_SCL_MAX_MHZ);
    $display("timing: min tCAS=%.1f (>=%.1f) tCASr=%.1f (>=%.1f) tCBP=%.1f (>=%.1f) tCBSr=%.1f (>=%.1f) busfree=%.1f",
             min_cas, T_CAS_MIN, min_casr, T_CASR_MIN, min_cbp, T_CBP_MIN, min_cbsr, T_CBSR_MIN, min_buf);
    $display("timing: longest SCL Low in frame=%.1f ns (<= 100 us), longest SCL High in frame=%.1f ns", max_low, max_high_in_frame);
    $display("timing: violations=%0d", errors);
  endfunction
endmodule
