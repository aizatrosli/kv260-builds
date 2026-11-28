// Verilator C++ testbench for rtl/tob_engine.v.
//
// Replays a stimulus file (one 32-bit AXIS beat per line: "<hex word> <tlast> <expect>", where
// expect=1 marks the last beat of a message that must produce a feature vector), with random
// tvalid gaps on the input and random tready stalls on the output, checks AXI4-Stream rules on
// the output (tvalid/tdata stable while stalled, tlast on the last beat, no extra vectors), and
// writes every received vector as "<F0..F7 hex> <lat_rtl> <lat_tb>" for tb/check.py.
//
//   Vtob_engine stim=<file> out=<file> seed=<n> in_stall=<pct> out_stall=<pct> gap=<cycles>
// gap idles the input for that many cycles after every tlast (unloaded-latency measurement).
#include "Vtob_engine.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#ifndef OUT_W
#define OUT_W 32
#endif
static const int BEATS = 256 / OUT_W;

struct Beat { uint32_t w; int last, expect; };

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)rs; }
static bool chance(int pct) { return pct > 0 && (int)(rnd() % 100) < pct; }

static void get_out(Vtob_engine* t, uint32_t* w) {
#if OUT_W == 256
  for (int k = 0; k < 8; k++) w[k] = t->m_axis_tdata[k];
#else
  w[0] = t->m_axis_tdata;
#endif
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  std::string stim, out;
  long seed = 1; int in_stall = 0, out_stall = 0, gap = 0, gap_left = 0;
  for (int i = 1; i < argc; i++) {
    const char* a = argv[i];
    if (!strncmp(a, "stim=", 5)) stim = a + 5;
    else if (!strncmp(a, "out=", 4)) out = a + 4;
    else if (!strncmp(a, "seed=", 5)) seed = atol(a + 5);
    else if (!strncmp(a, "in_stall=", 9)) in_stall = atoi(a + 9);
    else if (!strncmp(a, "out_stall=", 10)) out_stall = atoi(a + 10);
    else if (!strncmp(a, "gap=", 4)) gap = atoi(a + 4);
  }
  rs ^= (uint64_t)seed * 0x9E3779B97F4A7C15ull; for (int i = 0; i < 8; i++) rnd();
  std::vector<Beat> beats;
  FILE* f = fopen(stim.c_str(), "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", stim.c_str()); return 2; }
  { unsigned w; int l, e; while (fscanf(f, "%x %d %d", &w, &l, &e) == 3) beats.push_back({w, l, e}); }
  fclose(f);
  long expected = 0; for (auto& b : beats) expected += b.expect;
  FILE* fo = fopen(out.c_str(), "w");
  if (!fo) { fprintf(stderr, "cannot open %s\n", out.c_str()); return 2; }

  Vtob_engine* t = new Vtob_engine;
  t->clk_i = 0; t->rst_ni = 0; t->s_axis_tvalid = 0; t->m_axis_tready = 0;
  for (int i = 0; i < 6; i++) { t->clk_i = 0; t->eval(); t->clk_i = 1; t->eval(); }
  t->rst_ni = 1;

  size_t ip = 0; bool in_v = false;
  long got = 0, errors = 0, n = 0, first_in = -1, last_out = -1, idle_after = 0;
  int obeat = 0; bool started = false; uint32_t vec[8]; long lat_rtl = 0, lat_tb = 0;
  std::deque<long> pend;              // input cycle of each expected message
  bool prev_stall = false; uint32_t prev_w[8] = {0};
  const long MAXC = (long)beats.size() * (40 + gap) + 100000;

  for (n = 0; n < MAXC; n++) {
    if (!in_v && ip < beats.size()) { if (gap_left > 0) gap_left--; else in_v = !chance(in_stall); }
    t->s_axis_tvalid = in_v;
    t->s_axis_tdata = in_v ? beats[ip].w : 0xDEADBEEF;
    t->s_axis_tlast = in_v ? beats[ip].last : 0;
    t->m_axis_tready = !chance(out_stall);
    t->clk_i = 0; t->eval();

    // ---- sample this cycle ----
    uint32_t cw[8]; get_out(t, cw);
    if (prev_stall) {
      if (!t->m_axis_tvalid || memcmp(cw, prev_w, sizeof(uint32_t) * (OUT_W / 32))) {
        fprintf(stderr, "AXIS violation: output changed while stalled at cycle %ld\n", n); errors++;
      }
    }
    if (t->m_axis_tvalid && obeat == 0 && !started) {
      started = true;
      if (pend.empty()) { fprintf(stderr, "unexpected output vector at cycle %ld\n", n); errors++; lat_tb = -1; }
      else { lat_tb = n - pend.front(); pend.pop_front(); }
      lat_rtl = t->lat_last_o;
    }
    bool in_fire = t->s_axis_tvalid && t->s_axis_tready;
    bool out_fire = t->m_axis_tvalid && t->m_axis_tready;
    if (in_fire) {
      if (first_in < 0) first_in = n;
      if (beats[ip].expect) pend.push_back(n);
    }
    if (out_fire) {
      int expl = (obeat == BEATS - 1);
      if ((int)t->m_axis_tlast != expl) { fprintf(stderr, "bad tlast at cycle %ld beat %d\n", n, obeat); errors++; }
      for (int k = 0; k < OUT_W / 32; k++) vec[obeat * (OUT_W / 32) + k] = cw[k];
      if (++obeat == BEATS) {
        for (int k = 0; k < 8; k++) fprintf(fo, "%08x ", vec[k]);
        fprintf(fo, "%ld %ld\n", lat_rtl, lat_tb);
        obeat = 0; started = false; got++; last_out = n;
      }
    }
    prev_stall = t->m_axis_tvalid && !t->m_axis_tready;
    memcpy(prev_w, cw, sizeof(prev_w));

    t->clk_i = 1; t->eval();
    if (in_fire) { if (beats[ip].last) gap_left = gap; ip++; in_v = false; }
    if (ip >= beats.size() && got >= expected && ++idle_after > 300) break;
  }
  fclose(fo);
  if (got != expected) { fprintf(stderr, "got %ld vectors, expected %ld\n", got, expected); errors++; }
  if (!pend.empty()) { fprintf(stderr, "%zu messages without output\n", pend.size()); errors++; }
  if ((long)t->msg_out_o != got) { fprintf(stderr, "msg_out_o %u != %ld\n", t->msg_out_o, got); errors++; }
  long span = last_out - first_in + 1;
  printf("TB: OUT_W=%d beats=%zu vectors=%ld cycles=%ld span=%ld msgs/cycle=%.4f rx_err=%u "
         "lat_min=%u lat_max=%u in_stall=%d out_stall=%d gap=%d errors=%ld\n",
         OUT_W, beats.size(), got, n, span, span > 0 ? (double)got / span : 0.0,
         (unsigned)t->rx_err_o, t->lat_min_o, t->lat_max_o, in_stall, out_stall, gap, errors);
  t->final(); delete t;
  return errors ? 1 : 0;
}
