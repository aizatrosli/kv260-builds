`timescale 1ns/1ps
// CRC-8 PEC for the MCTP I3C binding.
// DSP0233 v1.0.1 §5.3.1: polynomial x^8+x^2+x+1 (0x07), initial value 0x00, no final XOR,
// computed like the SMBus PEC (SMBus 2.0 §5.4.1.3: MSB first, no reflection).
// Covers the I3C address byte (with RnW) and every MCTP byte up to the PEC. It does not
// cover the 7'h7E broadcast header, START/STOP, T-bits or ACK/NACK.
// Check value over ASCII "123456789" = 0xF4 (standard CRC-8/SMBUS); tested in tb_mctp_i3c_tx.
package crc8_pec_pkg;
  function automatic logic [7:0] crc8_pec(input logic [7:0] crc, input logic [7:0] data);
    logic [7:0] c;
    c = crc ^ data;
    for (int i = 0; i < 8; i++) c = c[7] ? ((c << 1) ^ 8'h07) : (c << 1);
    return c;
  endfunction
endpackage
