// generic_regfile.v
// 16 x 64-bit register file, 2 async read ports, 1 synchronous write port.
// Synthesises to distributed LUT-RAM or flip-flops (appropriate for 16 regs).
// r14 pre-loaded with packed threadIdx.x vector.
// r15 is the zero register - write suppression is handled in gpu_core
// via the rf_we gate (rf_we = wb_we & (wb_rd != 4'hF)).
`timescale 1ns/1ps
`default_nettype none
module generic_regfile #(
    parameter CORE_THREAD_BASE = 0
) (
    input  wire        clk,
    input  wire        rst_n,
    // Read port A (rs1) - asynchronous
    input  wire [3:0]  rs1_addr,
    output wire [63:0] rs1_data,
    // Read port B (rs2) - asynchronous
    input  wire [3:0]  rs2_addr,
    output wire [63:0] rs2_data,
    // Write port - synchronous
    input  wire        we,
    input  wire [3:0]  rd_addr,
    input  wire [63:0] rd_data
);

// Pre-compute 16-bit lane values as parameters to avoid truncation warnings
localparam [15:0] LANE0 = CORE_THREAD_BASE[15:0] + 16'd0;
localparam [15:0] LANE1 = CORE_THREAD_BASE[15:0] + 16'd1;
localparam [15:0] LANE2 = CORE_THREAD_BASE[15:0] + 16'd2;
localparam [15:0] LANE3 = CORE_THREAD_BASE[15:0] + 16'd3;

localparam [63:0] R14_INIT = {LANE3, LANE2, LANE1, LANE0};

reg [63:0] regs [0:15];
integer i;

initial begin
    for (i = 0; i < 16; i = i + 1)
        regs[i] = 64'd0;
    // r14 = packed threadIdx.x: {lane3, lane2, lane1, lane0} each 16-bit
    regs[14] = R14_INIT;
    // r15 = zero register, stays 0
    regs[15] = 64'd0;
end

// Synchronous write
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < 16; i = i + 1)
            regs[i] <= 64'd0;
        regs[14] <= R14_INIT;
    end else if (we) begin
        regs[rd_addr] <= rd_data;
    end
end

// Asynchronous reads
assign rs1_data = regs[rs1_addr];
assign rs2_data = regs[rs2_addr];

endmodule
`default_nettype wire