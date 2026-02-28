// inst_mem.v
// 128 x 32-bit dual-port instruction memory.
// Port A = host write  (loads program while GPU is held in reset)
// Port B = GPU read    (instruction fetch, 1-cycle registered latency)
// No IP core needed - plain flip-flop/LUT RAM, works in sim and synthesis.
`timescale 1ns/1ps
`default_nettype none

module inst_mem (
    input  wire        clk,

    // Port A: host write
    input  wire        host_we,
    input  wire [6:0]  host_addr,
    input  wire [31:0] host_data,

    // Port B: GPU fetch
    input  wire [6:0]  pc,
    output reg  [31:0] instruction
);

reg [31:0] mem [0:127];

integer i;
initial begin
    for (i = 0; i < 128; i = i + 1)
        mem[i] = 32'hF0000000;   // HALT everywhere by default
end

// Port A: synchronous write
always @(posedge clk)
    if (host_we) mem[host_addr] <= host_data;

// Port B: synchronous read (1-cycle latency matches BRAM behaviour)
always @(posedge clk)
    instruction <= mem[pc];

endmodule
`default_nettype wire
