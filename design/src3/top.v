// top.v
// =========================================================================
// Top-level module.  Wraps datapath with a generic_regs register bus so a
// host CPU can:
//   1. Load a program into instruction memory word-by-word
//   2. Pre-load data memory (test vectors, input arrays)
//   3. Release GPU reset and start execution
//   4. Poll the halted flag to know when done
//   5. Read back result data from data memory
//
// Register map (all 32-bit, base = `TOP_BLOCK_ADDR):
//
//   SOFTWARE REGS  (host writes)          HARDWARE REGS  (host reads)
//   ─────────────────────────────────     ──────────────────────────────
//   offset 0  GPU_RST        [0]          offset 0  HW_HALTED     [0]
//   offset 1  GPU_START      [0]          offset 1  HW_PC         [6:0]
//   offset 2  IMEM_WR_EN     [0]          offset 2  HW_RESULT_LO  [31:0]
//   offset 3  IMEM_WR_ADDR   [6:0]        offset 3  HW_RESULT_HI  [31:0]
//   offset 4  IMEM_WR_DATA   [31:0]       offset 4  HW_DMEM_RD_LO [31:0]
//   offset 5  DMEM_WR_EN     [0]          offset 5  HW_DMEM_RD_HI [31:0]
//   offset 6  DMEM_WR_ADDR   [7:0]
//   offset 7  DMEM_WR_DATA_LO[31:0]
//   offset 8  DMEM_WR_DATA_HI[31:0]
//   offset 9  DMEM_RD_ADDR   [7:0]
//
// Typical host sequence to run a program:
//   1. Write GPU_RST=1                    -- hold GPU in reset
//   2. Write IMEM_WR_EN=1
//      For each instruction:
//        Write IMEM_WR_ADDR=n
//        Write IMEM_WR_DATA=instruction
//   3. Write IMEM_WR_EN=0
//   4. Write GPU_RST=0, GPU_START=1       -- release reset, start
//   5. Poll HW_HALTED until it reads 1
//   6. Write DMEM_RD_ADDR=n
//      Read HW_DMEM_RD_LO and HW_DMEM_RD_HI -- one cycle latency
// =========================================================================
`include "defines.v"
`timescale 1ns/1ps
`default_nettype none

module top #(
    parameter UDP_REG_SRC_WIDTH = 2
) (
    // Register bus
    input  wire                          reg_req_in,
    input  wire                          reg_ack_in,
    input  wire                          reg_rd_wr_L_in,
    input  wire [`UDP_REG_ADDR_WIDTH-1:0] reg_addr_in,
    input  wire [`CPCI_NF2_DATA_WIDTH-1:0] reg_data_in,
    input  wire [UDP_REG_SRC_WIDTH-1:0]  reg_src_in,

    output wire                          reg_req_out,
    output wire                          reg_ack_out,
    output wire                          reg_rd_wr_L_out,
    output wire [`UDP_REG_ADDR_WIDTH-1:0] reg_addr_out,
    output wire [`CPCI_NF2_DATA_WIDTH-1:0] reg_data_out,
    output wire [UDP_REG_SRC_WIDTH-1:0]  reg_src_out,

    // Misc
    input  wire                          clk,
    input  wire                          reset
);

// -------------------------------------------------------------------------
// Software registers (host → GPU)
// -------------------------------------------------------------------------
wire [31:0] sw_gpu_rst;          // [0] hold GPU in reset
wire [31:0] sw_gpu_start;        // [0] start execution
wire [31:0] sw_imem_wr_en;       // [0] enable imem write port
wire [31:0] sw_imem_wr_addr;     // [6:0] imem word address
wire [31:0] sw_imem_wr_data;     // [31:0] instruction word
wire [31:0] sw_dmem_wr_en;       // [0] enable dmem write port
wire [31:0] sw_dmem_wr_addr;     // [7:0] dmem word address
wire [31:0] sw_dmem_wr_data_lo;  // [31:0] dmem write data low
wire [31:0] sw_dmem_wr_data_hi;  // [31:0] dmem write data high
wire [31:0] sw_dmem_rd_addr;     // [7:0] dmem readback address

// -------------------------------------------------------------------------
// Hardware registers (GPU → host)
// -------------------------------------------------------------------------
reg [31:0] hw_halted;        // GPU halted flag
reg [31:0] hw_pc;            // current PC
reg [31:0] hw_result_lo;     // last writeback [31:0]
reg [31:0] hw_result_hi;     // last writeback [63:32]
reg [31:0] hw_dmem_rd_lo;    // dmem readback [31:0]
reg [31:0] hw_dmem_rd_hi;    // dmem readback [63:32]

// -------------------------------------------------------------------------
// GPU control signals derived from software registers
// -------------------------------------------------------------------------
wire gpu_rst_n  = ~sw_gpu_rst[0] & ~reset;  // active-low reset to GPU
wire gpu_start  =  sw_gpu_start[0];          // start pulse (unused pin for now,
                                             // kept for future gating logic)

// Instruction memory host write port
wire        imem_host_we   = sw_imem_wr_en[0];
wire [6:0]  imem_host_addr = sw_imem_wr_addr[6:0];
wire [31:0] imem_host_data = sw_imem_wr_data;

// Data memory host write port
wire        dmem_host_we      = sw_dmem_wr_en[0];
wire [7:0]  dmem_host_wr_addr = sw_dmem_wr_addr[7:0];
wire [63:0] dmem_host_wr_data = {sw_dmem_wr_data_hi, sw_dmem_wr_data_lo};

// Data memory host read port
wire [7:0]  dmem_host_rd_addr = sw_dmem_rd_addr[7:0];
wire [63:0] dmem_host_rd_data;   // driven by datapath

// GPU status outputs
wire        gpu_halted;
wire [31:0] gpu_pc;
wire [63:0] gpu_result;

// -------------------------------------------------------------------------
// Datapath instance
// -------------------------------------------------------------------------
datapath u_datapath (
    .clk               (clk),
    .rst_n             (gpu_rst_n),

    // Host instruction memory write port
    .imem_host_we      (imem_host_we),
    .imem_host_addr    (imem_host_addr),
    .imem_host_data    (imem_host_data),

    // Host data memory write port
    .dmem_host_we      (dmem_host_we),
    .dmem_host_wr_addr (dmem_host_wr_addr),
    .dmem_host_wr_data (dmem_host_wr_data),

    // Host data memory read port
    .dmem_host_rd_addr (dmem_host_rd_addr),
    .dmem_host_rd_data (dmem_host_rd_data),

    // Status outputs
    .halted            (gpu_halted),
    .pc_out            (gpu_pc),
    .result_out        (gpu_result)
);

// -------------------------------------------------------------------------
// Hardware register update
// -------------------------------------------------------------------------
always @(posedge clk) begin
    if (reset) begin
        hw_halted      <= 32'd0;
        hw_pc          <= 32'd0;
        hw_result_lo   <= 32'd0;
        hw_result_hi   <= 32'd0;
        hw_dmem_rd_lo  <= 32'd0;
        hw_dmem_rd_hi  <= 32'd0;
    end else begin
        hw_halted     <= {31'd0, gpu_halted};
        hw_pc         <= gpu_pc;
        hw_result_lo  <= gpu_result[31:0];
        hw_result_hi  <= gpu_result[63:32];
        // dmem readback has 1-cycle BRAM latency; registered here captures it
        hw_dmem_rd_lo <= dmem_host_rd_data[31:0];
        hw_dmem_rd_hi <= dmem_host_rd_data[63:32];
    end
end

// -------------------------------------------------------------------------
// Generic register block
// 10 software regs (host writes), 6 hardware regs (host reads)
// -------------------------------------------------------------------------
generic_regs #(
    .UDP_REG_SRC_WIDTH  (UDP_REG_SRC_WIDTH),
    .TAG                (`TOP_BLOCK_ADDR),
    .REG_ADDR_WIDTH     (`TOP_REG_ADDR_WIDTH),
    .NUM_COUNTERS       (0),
    .NUM_SOFTWARE_REGS  (10),
    .NUM_HARDWARE_REGS  (6)
) module_regs (
    .reg_req_in        (reg_req_in),
    .reg_ack_in        (reg_ack_in),
    .reg_rd_wr_L_in    (reg_rd_wr_L_in),
    .reg_addr_in       (reg_addr_in),
    .reg_data_in       (reg_data_in),
    .reg_src_in        (reg_src_in),

    .reg_req_out       (reg_req_out),
    .reg_ack_out       (reg_ack_out),
    .reg_rd_wr_L_out   (reg_rd_wr_L_out),
    .reg_addr_out      (reg_addr_out),
    .reg_data_out      (reg_data_out),
    .reg_src_out       (reg_src_out),

    .counter_updates   (),
    .counter_decrement (),

    // Software regs: index 0 = first in bus, packed LSB-first
    .software_regs ({
        sw_dmem_rd_addr,       // offset 9
        sw_dmem_wr_data_hi,    // offset 8
        sw_dmem_wr_data_lo,    // offset 7
        sw_dmem_wr_addr,       // offset 6
        sw_dmem_wr_en,         // offset 5
        sw_imem_wr_data,       // offset 4
        sw_imem_wr_addr,       // offset 3
        sw_imem_wr_en,         // offset 2
        sw_gpu_start,          // offset 1
        sw_gpu_rst             // offset 0
    }),

    // Hardware regs: index 0 = first in bus
    .hardware_regs ({
        hw_dmem_rd_hi,         // offset 5
        hw_dmem_rd_lo,         // offset 4
        hw_result_hi,          // offset 3
        hw_result_lo,          // offset 2
        hw_pc,                 // offset 1
        hw_halted              // offset 0
    }),

    .clk   (clk),
    .reset (reset)
);

endmodule
`default_nettype wire
