// data_mem.v
// 256 x 64-bit data memory.
// 1 synchronous write port, 2 synchronous read ports (1-cycle latency).
//   Write port  = GPU ST  OR  host pre-load  (muxed in datapath)
//   Read port A = GPU LD
//   Read port B = host readback after HALT
// Pre-loaded with test vectors matching the testbench address constants.
`timescale 1ns/1ps
`default_nettype none

module data_mem (
    input  wire        clk,

    // Write port
    input  wire        wr_en,
    input  wire [7:0]  wr_addr,
    input  wire [63:0] wr_data,

    // Read port A (GPU LD)
    input  wire        rd_en_a,
    input  wire [7:0]  rd_addr_a,
    output reg  [63:0] rd_data_a,

    // Read port B (host readback)
    input  wire        rd_en_b,
    input  wire [7:0]  rd_addr_b,
    output reg  [63:0] rd_data_b
);

reg [63:0] mem [0:255];

integer i;
initial begin
    for (i = 0; i < 256; i = i + 1)
        mem[i] = 64'd0;

    // byte 0x080 -> word 16 : int16x4 [1,2,3,4]
    mem[16] = {16'd4,    16'd3,    16'd2,    16'd1   };
    // byte 0x0A0 -> word 20 : int16x4 [10,20,30,40]
    mem[20] = {16'd40,   16'd30,   16'd20,   16'd10  };
    // byte 0x100 -> word 32 : bf16x4 [1.0,2.0,3.0,4.0]
    mem[32] = {16'h4080, 16'h4040, 16'h4000, 16'h3F80};
    // byte 0x120 -> word 36 : bf16x4 [2.0,2.0,2.0,2.0]
    mem[36] = {16'h4000, 16'h4000, 16'h4000, 16'h4000};
    // byte 0x140 -> word 40 : bf16x4 [1.0,1.0,1.0,1.0]
    mem[40] = {16'h3F80, 16'h3F80, 16'h3F80, 16'h3F80};
    // byte 0x180 -> word 48 : int16x4 [-1,-2,-3,-4]
    mem[48] = {-16'sd4,  -16'sd3,  -16'sd2,  -16'sd1 };
end

// Synchronous write
always @(posedge clk)
    if (wr_en) mem[wr_addr] <= wr_data;

// Synchronous read A
always @(posedge clk)
    if (rd_en_a) rd_data_a <= mem[rd_addr_a];

// Synchronous read B
always @(posedge clk)
    if (rd_en_b) rd_data_b <= mem[rd_addr_b];

endmodule
`default_nettype wire
