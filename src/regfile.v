`timescale 1ns / 1ps
// Simple 2-read, 1-write register file

`include "defines.v"

module regfile (
    input  wire                     clk,
    input  wire                     rst_n,

    // Write port
    input  wire                     wena,
    input  wire [`REG_ADDR_WIDTH-1:0] waddr,
    input  wire [`DATA_WIDTH-1:0]     wdata,

    // Read ports
    input  wire [`REG_ADDR_WIDTH-1:0] rs1addr,
    input  wire [`REG_ADDR_WIDTH-1:0] rs2addr,
    output wire [`DATA_WIDTH-1:0]     rs1data,
    output wire [`DATA_WIDTH-1:0]     rs2data
);

    localparam NUMREGS = (1 << `REG_ADDR_WIDTH);

    // 2^REG_ADDR_WIDTH registers
    reg [`DATA_WIDTH-1:0] regs [0:(1<<`REG_ADDR_WIDTH)-1];

    // Read logic with write first bypass
    assign rs1data = (wena && (waddr == rs1addr)) ? wdata : regs[rs1addr];
    assign rs2data = (wena && (waddr == rs2addr)) ? wdata : regs[rs2addr];

    integer i;

    // Write logic
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            for (i=0; i<NUMREGS; i=i+1) begin
                regs[i] <= {`DATA_WIDTH{1'b0}};
            end
        end
        else if (wena) begin
            regs[waddr] <= wdata;
        end
    end

endmodule