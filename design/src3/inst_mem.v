// inst_mem.v
// 128 x 32-bit dual-port instruction memory.
// Port A = host write  (loads program while GPU is held in reset)
// Port B = GPU fetch   (asynchronous read ? no latency)
//
// FIX: changed Port B from synchronous to asynchronous read.
// A synchronous read adds a 1-cycle fetch latency that the pipeline
// does not account for, causing every instruction to execute twice.
`timescale 1ns/1ps
`default_nettype none

module inst_mem (
    input  wire        clk,
    input  wire        host_we,
    input  wire [6:0]  host_addr,
    input  wire [31:0] host_data,
    input  wire [6:0]  pc,
    output wire [31:0] instruction
);

reg [31:0] mem [0:127];
integer i;

initial begin
    for (i = 0; i < 128; i = i + 1)
        mem[i] = 32'hF0000000;
end

// Port A: synchronous write
always @(posedge clk)
    if (host_we) mem[host_addr] <= host_data;

// Port B: asynchronous read ? instruction available same cycle as PC
assign instruction = mem[pc];

endmodule
`default_nettype wire