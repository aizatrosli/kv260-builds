// book_side: one side of a price-level book, N levels held entirely in flip-flops.
//
// Bids (DESC=1) are kept in descending price, asks (DESC=0) in ascending price; slot 0 is the
// best level. Slots are a dense prefix: valid_o = 0..01..1 (LSB first), and every invalid slot
// holds price 0 / volume 0, so downstream sums need no masking.
//
// Why registers + parallel compare + shift (the classic HFT book-builder structure):
//   A software book walks the level list (O(N) dependent steps, or a tree/hash). In hardware we
//   instead compare the incoming price against ALL N slots at once (N equality and N magnitude
//   comparators, one LUT/carry-chain level each). Because the slots are sorted, "new price is
//   better than slot i" is a thermometer code 0..01..1 whose first 1 is the insert position,
//   and "slot i is at/after the matching slot" is the thermometer for a delete. Each slot then
//   picks its next value from a 4:1 mux: keep / load new / take neighbour i-1 (insert shifts the
//   tail down by one and pushes slot N-1 out) / take neighbour i+1 (delete closes the gap).
//   That is a fixed 2-cycle update regardless of where the level is or how full the book is:
//   no loops, no memory ports, deterministic latency - which is what matters for a feed handler.
//   It scales to N ~ 16-32 per side before the comparator fan-in and mux fan-out hurt Fmax;
//   deeper books are kept in BRAM off the critical path and only the top N live in registers.
//
// Timing (one update in flight; the caller must not issue req/clr on two consecutive cycles):
//   cycle A (req_i or clr_i high): compare price_i against all slots, register per-slot select
//   cycle B:                      apply the selects; new book visible on the outputs after B.
// Semantics match ftrade.features.TopBook._update exactly:
//   price present, vol>0 -> overwrite;  price present, vol==0 -> delete (shift up)
//   price absent,  vol>0 -> insert at sorted position if better than some kept level or the
//                           side is not full; level pushed past N is dropped; else ignored
//   price absent,  vol==0 -> ignored
`timescale 1ns/1ps

module book_side #(
  parameter N    = 8,
  parameter DESC = 1                  // 1: bids (descending), 0: asks (ascending)
) (
  input  wire            clk_i,
  input  wire            rst_ni,      // synchronous, active low
  input  wire            req_i,       // level update for this side (cycle A)
  input  wire            clr_i,       // empty the side (cycle A, applied in cycle B)
  input  wire [31:0]     price_i,
  input  wire [31:0]     vol_i,
  output wire [N*32-1:0] price_o,     // slot i at [32*i +: 32]
  output wire [N*32-1:0] vol_o,
  output wire [N-1:0]    valid_o
);
  localparam [1:0] SEL_KEEP = 2'd0, SEL_NEW = 2'd1, SEL_PREV = 2'd2, SEL_NEXT = 2'd3;

  reg [N*32-1:0] prc_q, vol_q;
  reg [N-1:0]    val_q;
  assign price_o = prc_q;
  assign vol_o   = vol_q;
  assign valid_o = val_q;

  // ---------------- cycle A: parallel compare ----------------
  reg [N-1:0] match, better, mt_th, in_th;
  reg [2*N-1:0] sel_d;
  reg any_match, can_ins, vzero;
  integer i;
  always @* begin
    for (i = 0; i < N; i = i + 1) begin
      match[i]  = val_q[i] & (prc_q[32*i +: 32] == price_i);
      better[i] = ~val_q[i] | ((DESC != 0) ? (price_i > prc_q[32*i +: 32])
                                           : (price_i < prc_q[32*i +: 32]));
    end
    // prefix-OR thermometers (robust even if the compare vector were not monotone)
    mt_th[0] = match[0]; in_th[0] = better[0];
    for (i = 1; i < N; i = i + 1) begin
      mt_th[i] = mt_th[i-1] | match[i];
      in_th[i] = in_th[i-1] | better[i];
    end
    any_match = mt_th[N-1];
    can_ins   = in_th[N-1];
    vzero     = (vol_i == 32'd0);
    for (i = 0; i < N; i = i + 1) begin
      if (any_match && !vzero)                       // overwrite volume in place
        sel_d[2*i +: 2] = match[i] ? SEL_NEW : SEL_KEEP;
      else if (any_match)                            // delete: slots at/after the match move up
        sel_d[2*i +: 2] = mt_th[i] ? SEL_NEXT : SEL_KEEP;
      else if (!vzero && can_ins)                    // insert: first 1 loads, the rest move down
        sel_d[2*i +: 2] = !in_th[i] ? SEL_KEEP :
                          ((i == 0) || !in_th[(i == 0) ? 0 : i-1]) ? SEL_NEW : SEL_PREV;
      else
        sel_d[2*i +: 2] = SEL_KEEP;
    end
  end

  reg [2*N-1:0] sel_q;
  reg [31:0]    p_q, v_q;
  reg           go_q, clr_q;
  always @(posedge clk_i) begin
    if (!rst_ni) begin
      go_q <= 1'b0; clr_q <= 1'b0; sel_q <= {2*N{1'b0}}; p_q <= 32'd0; v_q <= 32'd0;
    end else begin
      go_q  <= req_i | clr_i;
      clr_q <= clr_i;
      sel_q <= req_i ? sel_d : {2*N{1'b0}};
      p_q   <= price_i;
      v_q   <= vol_i;
    end
  end

  // ---------------- cycle B: one-cycle shift / insert / delete ----------------
  // Each slot is a 4:1 mux (keep / new / slot i-1 / slot i+1); slot N is a constant empty slot.
  reg [N*32-1:0] prc_d, vol_d;
  reg [N-1:0]    val_d;
  always @* begin
    for (i = 0; i < N; i = i + 1) begin
      case (sel_q[2*i +: 2])
        SEL_NEW: begin
          prc_d[32*i +: 32] = p_q; vol_d[32*i +: 32] = v_q; val_d[i] = 1'b1;
        end
        SEL_PREV: begin        // never selected for slot 0
          prc_d[32*i +: 32] = (i == 0) ? 32'd0 : prc_q[32*((i == 0) ? 0 : i-1) +: 32];
          vol_d[32*i +: 32] = (i == 0) ? 32'd0 : vol_q[32*((i == 0) ? 0 : i-1) +: 32];
          val_d[i]          = (i == 0) ? 1'b0  : val_q[(i == 0) ? 0 : i-1];
        end
        SEL_NEXT: begin
          prc_d[32*i +: 32] = (i == N-1) ? 32'd0 : prc_q[32*((i == N-1) ? i : i+1) +: 32];
          vol_d[32*i +: 32] = (i == N-1) ? 32'd0 : vol_q[32*((i == N-1) ? i : i+1) +: 32];
          val_d[i]          = (i == N-1) ? 1'b0  : val_q[(i == N-1) ? i : i+1];
        end
        default: begin
          prc_d[32*i +: 32] = prc_q[32*i +: 32]; vol_d[32*i +: 32] = vol_q[32*i +: 32];
          val_d[i] = val_q[i];
        end
      endcase
    end
  end

  always @(posedge clk_i) begin
    if (!rst_ni || (go_q && clr_q)) begin
      prc_q <= {N*32{1'b0}}; vol_q <= {N*32{1'b0}}; val_q <= {N{1'b0}};
    end else if (go_q) begin
      prc_q <= prc_d; vol_q <= vol_d; val_q <= val_d;
    end
  end
endmodule
