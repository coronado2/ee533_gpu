// generic_regfile.v
`timescale 1ns/1ps
`default_nettype none
module generic_regfile #(
    parameter CORE_THREAD_BASE = 0
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [3:0]  rs1_addr,
    output wire [63:0] rs1_data,
    input  wire [3:0]  rs2_addr,
    output wire [63:0] rs2_data,
    // Third read port: accumulator pre-value (rd current value for FMAC)
    input  wire [3:0]  acc_addr,
    output wire [63:0] acc_data,
    input  wire        we,
    input  wire [3:0]  rd_addr,
    input  wire [63:0] rd_data
);
localparam [15:0] T0 = CORE_THREAD_BASE[15:0];
localparam [15:0] T1 = CORE_THREAD_BASE[15:0] + 16'd1;
localparam [15:0] T2 = CORE_THREAD_BASE[15:0] + 16'd2;
localparam [15:0] T3 = CORE_THREAD_BASE[15:0] + 16'd3;
localparam [63:0] R14 = {T3, T2, T1, T0};
reg [63:0] regs [0:15];
integer i;
initial begin
    for (i = 0; i < 16; i = i + 1)
        regs[i] = 64'd0;
    regs[14] = R14;
    regs[15]  = 64'd0;
end
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < 16; i = i + 1)
            regs[i] <= 64'd0;
        regs[14] <= R14;
    end else if (we) begin
        regs[rd_addr] <= rd_data;
    end
end
// Write-bypass on all three read ports
assign rs1_data = (we && rd_addr == rs1_addr) ? rd_data : regs[rs1_addr];
assign rs2_data = (we && rd_addr == rs2_addr) ? rd_data : regs[rs2_addr];
assign acc_data = (we && rd_addr == acc_addr) ? rd_data : regs[acc_addr];
endmodule
`default_nettype wire