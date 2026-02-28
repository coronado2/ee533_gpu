// gpu_core.v
// =============================================================================
// GPU compute core.  All BRAM instantiation lives in datapath.v.
// This module exposes memory interface ports so the datapath can connect
// the BRAM IP cores directly.
//
// Pipeline: Fetch → Decode → Execute → Writeback
//   Integer ALU  : 1-cycle result  (VADD, VSUB, RELU)
//   Load         : 2-cycle result  (BRAM read registered in datapath)
//   Tensor VMUL  : 2-cycle result  (tensor unit)
//   Tensor FMAC  : 5-cycle result  (tensor unit)
//
// Stall sources:
//   tensor_busy  – multi-cycle tensor op in flight
//   halted       – HALT decoded, PC frozen, pipeline drains
//
// Thread ID:
//   r14 is pre-loaded by regfile_bram with the packed threadIdx.x vector.
//   r15 is the zero register (writes suppressed).
//
// Instruction encoding:
//   [31:28] opcode  [27:24] dtype  [23:20] rd
//   [19:16] rs1     [15:12] rs2    [11:0]  imm/unused
//
//   0=VADD  1=VSUB  2=VMUL(bf16)  3=FMAC(bf16)
//   4=RELU  5=LD    6=ST          F=HALT
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

module gpu_core #(
    parameter CORE_THREAD_BASE = 0
) (
    input  wire        clk,
    input  wire        rst_n,

    // ── Instruction memory port (to inst_mem in datapath) ──────────────────
    output wire [6:0]  imem_addr,       // PC → inst_mem address
    input  wire [31:0] imem_data,       // inst_mem → fetched instruction

    // ── Register file port (to regfile_bram in datapath) ───────────────────
    output wire [3:0]  rf_rs1_addr,
    input  wire [63:0] rf_rs1_data,
    output wire [3:0]  rf_rs2_addr,
    input  wire [63:0] rf_rs2_data,
    output wire        rf_we,
    output wire [3:0]  rf_wr_addr,
    output wire [63:0] rf_wr_data,

    // ── Data memory port (to data_mem_bram in datapath) ────────────────────
    output reg         dmem_ld_en,
    output reg  [7:0]  dmem_ld_addr,    // word address (byte >> 3)
    input  wire [63:0] dmem_ld_data,

    output reg         dmem_st_en,
    output reg  [7:0]  dmem_st_addr,    // word address (byte >> 3)
    output reg  [63:0] dmem_st_data
);

// ---------------------------------------------------------------------------
// Pipeline registers
// ---------------------------------------------------------------------------
reg [63:0] rs1_data_r, rs2_data_r;
reg [3:0]  opcode_r, dtype_r, rd_r;
reg        reg_write_r;
reg        is_ld_r, is_st_r;

// ---------------------------------------------------------------------------
// Program Counter  (7-bit word address for 128-entry inst_mem)
// ---------------------------------------------------------------------------
reg  [6:0] pc_reg;
wire       stall;
reg        halted;

assign imem_addr = pc_reg;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)       pc_reg <= 7'd0;
    else if (!stall)  pc_reg <= pc_reg + 7'd1;
end

// ---------------------------------------------------------------------------
// Decode  (combinational, operates on registered instruction from BRAM)
// BRAM has 1-cycle latency so imem_data is already the registered output.
// ---------------------------------------------------------------------------
wire [3:0] opcode  = imem_data[31:28];
wire [3:0] dtype   = imem_data[27:24];
wire [3:0] rd      = imem_data[23:20];
wire [3:0] rs1     = imem_data[19:16];
wire [3:0] rs2     = imem_data[15:12];

wire is_ld   = (opcode == 4'h5);
wire is_st   = (opcode == 4'h6);
wire is_halt = (opcode == 4'hF);
wire reg_write = ~is_st & ~is_halt;

// ---------------------------------------------------------------------------
// Register file – address/data wires routed to datapath
// ---------------------------------------------------------------------------
assign rf_rs1_addr = rs1;
assign rf_rs2_addr = rs2;

// Writeback (driven at bottom of file)
reg        wb_we;
reg [3:0]  wb_rd;
reg [63:0] wb_data;

// Suppress writes to r15 (zero register)
assign rf_we      = wb_we & (wb_rd != 4'hF);
assign rf_wr_addr = wb_rd;
assign rf_wr_data = wb_data;

// ---------------------------------------------------------------------------
// Integer SIMD ALU  (VADD, VSUB, RELU – combinational, int16×4)
// ---------------------------------------------------------------------------
wire [63:0] int_result;

exec_int16x4 u_exec_int (
    .opcode (opcode_r),
    .a      (rs1_data_r),
    .b      (rs2_data_r),
    .result (int_result)
);

// ---------------------------------------------------------------------------
// Tensor unit – BF16×4 lanes
// ---------------------------------------------------------------------------
wire tensor_op_r = (opcode_r == 4'h2) || (opcode_r == 4'h3);
reg  tensor_busy;

wire [2:0] tensor_op_sel = (opcode_r == 4'h2) ? 3'b010 :   // VMUL
                            (opcode_r == 4'h3) ? 3'b011 :   // FMAC
                                                 3'b000;

wire tensor_start_wire = tensor_op_r & ~tensor_busy;

wire [63:0] tensor_result;
wire        tensor_done;

// Accumulator for FMAC: capture rd's current value one cycle before launch
reg [63:0] acc_lat;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) acc_lat <= 64'd0;
    else        acc_lat <= rf_rs2_data;
end

tensor_bf16_4lane u_tensor (
    .clk     (clk),
    .rst_n   (rst_n),
    .start   (tensor_start_wire),
    .op      (tensor_op_sel),
    .a_reg   (rs1_data_r),
    .b_reg   (rs2_data_r),
    .acc_reg (acc_lat),
    .rd_reg  (tensor_result),
    .done    (tensor_done)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)                 tensor_busy <= 1'b0;
    else if (tensor_start_wire) tensor_busy <= 1'b1;
    else if (tensor_done)       tensor_busy <= 1'b0;
end

// ---------------------------------------------------------------------------
// Pipeline stage: latch decode outputs into execute stage
// ---------------------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rs1_data_r  <= 64'd0;  rs2_data_r  <= 64'd0;
        opcode_r    <= 4'd0;   dtype_r     <= 4'd0;
        rd_r        <= 4'd0;   reg_write_r <= 1'b0;
        is_ld_r     <= 1'b0;   is_st_r     <= 1'b0;
    end else if (!stall) begin
        rs1_data_r  <= rf_rs1_data;
        rs2_data_r  <= rf_rs2_data;
        opcode_r    <= opcode;
        dtype_r     <= dtype;
        rd_r        <= rd;
        reg_write_r <= reg_write & ~halted;
        is_ld_r     <= is_ld;
        is_st_r     <= is_st;
    end
end

// ---------------------------------------------------------------------------
// Data memory address/enable (combinational, pipelined operands)
// Addresses are byte-addressed in the program; we right-shift 3 for
// the 8-byte-aligned word address that the BRAM uses.
// ---------------------------------------------------------------------------
always @(*) begin
    dmem_ld_en   = 1'b0;
    dmem_ld_addr = 8'd0;
    dmem_st_en   = 1'b0;
    dmem_st_addr = 8'd0;
    dmem_st_data = 64'd0;

    case (opcode_r)
        4'h5: begin                                  // LD
            dmem_ld_en   = 1'b1;
            dmem_ld_addr = rs1_data_r[10:3];         // byte → word addr
        end
        4'h6: begin                                  // ST
            dmem_st_en   = 1'b1;
            dmem_st_addr = rs1_data_r[10:3];
            dmem_st_data = rs2_data_r;
        end
        default: ;
    endcase
end

// ---------------------------------------------------------------------------
// Load pipeline  (BRAM read has 1 registered cycle of latency)
// ---------------------------------------------------------------------------
reg        is_ld_rr;
reg [63:0] dmem_ld_data_r;
reg [3:0]  rd_rr;
reg        reg_write_rr;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        is_ld_rr       <= 1'b0;
        dmem_ld_data_r <= 64'd0;
        rd_rr          <= 4'd0;
        reg_write_rr   <= 1'b0;
    end else begin
        is_ld_rr       <= is_ld_r;
        dmem_ld_data_r <= dmem_ld_data;
        rd_rr          <= rd_r;
        reg_write_rr   <= reg_write_r & is_ld_r;
    end
end

// ---------------------------------------------------------------------------
// Integer ALU result register (align to writeback timing)
// ---------------------------------------------------------------------------
reg [63:0] wb_alu;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) wb_alu <= 64'd0;
    else        wb_alu <= int_result;
end

// ---------------------------------------------------------------------------
// HALT
// ---------------------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)       halted <= 1'b0;
    else if (is_halt) halted <= 1'b1;
end

assign stall = halted | tensor_busy;

// ---------------------------------------------------------------------------
// Writeback arbitration
// Priority: load (oldest, rd_rr) > tensor > integer ALU (rd_r)
// ---------------------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        wb_we   <= 1'b0;
        wb_rd   <= 4'd0;
        wb_data <= 64'd0;
    end else begin
        if (reg_write_rr) begin
            wb_we   <= 1'b1;
            wb_rd   <= rd_rr;
            wb_data <= dmem_ld_data_r;
        end else if (tensor_done) begin
            wb_we   <= 1'b1;
            wb_rd   <= rd_r;
            wb_data <= tensor_result;
        end else if (reg_write_r & ~is_ld_r) begin
            wb_we   <= 1'b1;
            wb_rd   <= rd_r;
            wb_data <= wb_alu;
        end else begin
            wb_we   <= 1'b0;
            wb_rd   <= 4'd0;
            wb_data <= 64'd0;
        end
    end
end

endmodule
`default_nettype wire