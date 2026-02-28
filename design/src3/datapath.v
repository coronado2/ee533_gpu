// datapath.v
// =========================================================================
// Structural top of the GPU datapath.
// Instantiates all memories and gpu_core, wires them together.
// Exposes host-facing ports so top.v can load programs, load data,
// read results, and monitor GPU status via the register bus.
//
// Hierarchy:
//   datapath
//   ├── inst_mem          128 x 32-bit RAM  (dual-port: host wr / GPU rd)
//   ├── generic_regfile    16 x 64-bit RAM  (2R/1W for GPU)
//   ├── data_mem          256 x 64-bit RAM  (3-port: GPU ST, GPU LD, host rd)
//   └── gpu_core          pipeline, ALU, tensor unit
// =========================================================================
`timescale 1ns/1ps
`default_nettype none

module datapath #(
    parameter CORE_THREAD_BASE = 0
) (
    input  wire        clk,
    input  wire        rst_n,

    // Host instruction memory write port (use while rst_n=0 only)
    input  wire        imem_host_we,
    input  wire [6:0]  imem_host_addr,
    input  wire [31:0] imem_host_data,

    // Host data memory write port (use while rst_n=0 only)
    input  wire        dmem_host_we,
    input  wire [7:0]  dmem_host_wr_addr,
    input  wire [63:0] dmem_host_wr_data,

    // Host data memory read port (1-cycle latency)
    input  wire [7:0]  dmem_host_rd_addr,
    output wire [63:0] dmem_host_rd_data,

    // GPU status
    output wire        halted,
    output wire [31:0] pc_out,
    output wire [63:0] result_out
);

// -------------------------------------------------------------------------
// Internal wires
// -------------------------------------------------------------------------
wire [6:0]  imem_gpu_addr;
wire [31:0] imem_gpu_data;

wire [3:0]  rf_rs1_addr, rf_rs2_addr, rf_wr_addr;
wire [63:0] rf_rs1_data, rf_rs2_data, rf_wr_data;
wire        rf_we;

wire        dmem_ld_en,   dmem_st_en;
wire [7:0]  dmem_ld_addr, dmem_st_addr;
wire [63:0] dmem_ld_data, dmem_st_data;

// -------------------------------------------------------------------------
// Instruction Memory  -  128 x 32-bit dual-port RAM
// Port A = host write (load program while GPU in reset)
// Port B = GPU instruction fetch
// -------------------------------------------------------------------------
inst_mem u_imem (
    .clk         (clk),
    .host_we     (imem_host_we),
    .host_addr   (imem_host_addr),
    .host_data   (imem_host_data),
    .pc          (imem_gpu_addr),
    .instruction (imem_gpu_data)
);

// -------------------------------------------------------------------------
// Register File  -  16 x 64-bit generic flip-flop array
// r14 = packed threadIdx.x vector (CORE_THREAD_BASE)
// r15 = zero register (writes suppressed in gpu_core)
// -------------------------------------------------------------------------
generic_regfile #(
    .CORE_THREAD_BASE (CORE_THREAD_BASE)
) u_regfile (
    .clk      (clk),
    .rst_n    (rst_n),
    .rs1_addr (rf_rs1_addr),
    .rs1_data (rf_rs1_data),
    .rs2_addr (rf_rs2_addr),
    .rs2_data (rf_rs2_data),
    .we       (rf_we),
    .rd_addr  (rf_wr_addr),
    .rd_data  (rf_wr_data)
);

// -------------------------------------------------------------------------
// Data Memory  -  256 x 64-bit three-port RAM
// Write port muxed: host owns it while rst_n=0, GPU owns it while running
// GPU LD port    = rd_a
// Host readback  = rd_b (always enabled so host can read after HALT)
// -------------------------------------------------------------------------
wire        dmem_wr_en   = rst_n ? dmem_st_en         : dmem_host_we;
wire [7:0]  dmem_wr_addr = rst_n ? dmem_st_addr        : dmem_host_wr_addr;
wire [63:0] dmem_wr_data = rst_n ? dmem_st_data        : dmem_host_wr_data;

data_mem u_dmem (
    .clk       (clk),
    .wr_en     (dmem_wr_en),
    .wr_addr   (dmem_wr_addr),
    .wr_data   (dmem_wr_data),
    .rd_en_a   (dmem_ld_en),
    .rd_addr_a (dmem_ld_addr),
    .rd_data_a (dmem_ld_data),
    .rd_en_b   (1'b1),
    .rd_addr_b (dmem_host_rd_addr),
    .rd_data_b (dmem_host_rd_data)
);

// -------------------------------------------------------------------------
// GPU Core
// -------------------------------------------------------------------------
gpu_core #(
    .CORE_THREAD_BASE (CORE_THREAD_BASE)
) u_gpu (
    .clk          (clk),
    .rst_n        (rst_n),
    .imem_addr    (imem_gpu_addr),
    .imem_data    (imem_gpu_data),
    .rf_rs1_addr  (rf_rs1_addr),
    .rf_rs1_data  (rf_rs1_data),
    .rf_rs2_addr  (rf_rs2_addr),
    .rf_rs2_data  (rf_rs2_data),
    .rf_we        (rf_we),
    .rf_wr_addr   (rf_wr_addr),
    .rf_wr_data   (rf_wr_data),
    .dmem_ld_en   (dmem_ld_en),
    .dmem_ld_addr (dmem_ld_addr),
    .dmem_ld_data (dmem_ld_data),
    .dmem_st_en   (dmem_st_en),
    .dmem_st_addr (dmem_st_addr),
    .dmem_st_data (dmem_st_data)
);

// -------------------------------------------------------------------------
// Status outputs
// -------------------------------------------------------------------------
assign halted     = u_gpu.halted;
assign pc_out     = u_gpu.u_pc.pc_out;
assign result_out = u_gpu.wb_data;

endmodule
`default_nettype wire
