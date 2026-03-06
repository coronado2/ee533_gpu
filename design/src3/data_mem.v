// data_mem.v
// 256 x 64-bit data memory.
// 1 synchronous write port, 2 synchronous read ports (1-cycle latency).
`timescale 1ns/1ps
`default_nettype none

module data_mem (
    input  wire        clk,
    input  wire        wr_en,
    input  wire [7:0]  wr_addr,
    input  wire [63:0] wr_data,
    input  wire        rd_en_a,
    input  wire [7:0]  rd_addr_a,
    output reg  [63:0] rd_data_a,
    input  wire        rd_en_b,
    input  wire [7:0]  rd_addr_b,
    output reg  [63:0] rd_data_b
);

reg [63:0] mem [0:255];
integer i;

initial begin
    for (i = 0; i < 256; i = i + 1)
        mem[i] = 64'd0;
    mem[16] = {16'd4,    16'd3,    16'd2,    16'd1   };
    mem[20] = {16'd40,   16'd30,   16'd20,   16'd10  };
    mem[32] = {16'h4080, 16'h4040, 16'h4000, 16'h3F80};
    mem[36] = {16'h4000, 16'h4000, 16'h4000, 16'h4000};
    mem[40] = {16'h3F80, 16'h3F80, 16'h3F80, 16'h3F80};
    mem[48] = {-16'sd4,  -16'sd3,  -16'sd2,  -16'sd1 };
end

always @(posedge clk)
    if (wr_en) mem[wr_addr] <= wr_data;

// FIX: port A is gated so rd_data_a holds its value between loads.
always @(posedge clk)
    if (rd_en_a) rd_data_a <= mem[rd_addr_a];

always @(posedge clk)
    if (rd_en_b) rd_data_b <= mem[rd_addr_b];

endmodule
`default_nettype wire