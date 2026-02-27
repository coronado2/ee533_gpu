// gpu_core.v
`timescale 1ns/1ps
`default_nettype none

module gpu_core (
    input  wire clk,
    input  wire rst_n
);

///////////////////////////////////////////////////////////////////////////////
// Pipeline & state registers (declare up front to avoid implicit-wire issues)
///////////////////////////////////////////////////////////////////////////////
reg [63:0] rs1_data_r, rs2_data_r;
reg [3:0]  opcode_r, dtype_r, rd_r;
reg        reg_write_r;
reg        is_ld_r, is_st_r;
reg [63:0] writeback_data;

///////////////////////////////////////////////////////////////////////////////
// ----- PC -----
///////////////////////////////////////////////////////////////////////////////
wire [31:0] pc;
reg  halted;
wire stall;                       // driven below
wire [31:0] next_pc = pc + 1;

// pc module must accept rst_n port; change pc.v if necessary to match this.
pc u_pc (
    .clk(clk),
    .rst_n(rst_n),
    .stall(stall),
    .next_pc(next_pc),
    .pc_out(pc)
);

///////////////////////////////////////////////////////////////////////////////
// ----- Instruction memory -----
// Connect block ROM (inst_mem) with .coe initialization
///////////////////////////////////////////////////////////////////////////////
/*wire [31:0] instruction;

inst_mem u_inst_mem (
    .clka(clk),
    .addra(pc[9:2]),   // word address (assuming 256 depth)
    .douta(instruction)
);
*/

///////////////////////////////////////////////////////////////////////////////
// Behavioral Instruction Memory (Simulation Only)
///////////////////////////////////////////////////////////////////////////////
reg [31:0] imem [0:255];   // 256 x 32-bit instructions
reg [31:0] instruction;

initial begin
    $readmemh("gpu_program.mem", imem);
end

always @(posedge clk) begin
    instruction <= imem[pc[9:2]];
end

///////////////////////////////////////////////////////////////////////////////
// ----- Decode -----
///////////////////////////////////////////////////////////////////////////////
wire [3:0] opcode, dtype, rd, rs1, rs2;
wire is_ld, is_st, is_halt;
wire reg_write;

control_unit u_ctrl (
    .instruction(instruction),
    .opcode(opcode), .dtype(dtype),
    .rd(rd), .rs1(rs1), .rs2(rs2),
    .is_ld(is_ld), .is_st(is_st), .is_halt(is_halt),
    .reg_write(reg_write)
);

//wire [63:0] rs1_data;
//wire [63:0] rs2_data;

///////////////////////////////////////////////////////////////////////////////
// ----- Register file ----- (mirrored BRAMs externally)

/*regfile_bram u_regfile (
    .clk(clk),
    .rs1_addr(rs1[3:0]),
    .rs1_data(rs1_data),
    .rs2_addr(rs2[3:0]),
    .rs2_data(rs2_data),
    .we(reg_write_r),        // write occurs in the pipeline stage (use reg_write_r)
    .rd_addr(rd_r[3:0]),
    .rd_data(writeback_data)
);
*/

///////////////////////////////////////////////////////////////////////////////
// Behavioral Register File (16 × 64-bit)
///////////////////////////////////////////////////////////////////////////////
reg [63:0] regfile [0:15];

wire [63:0] rs1_data;
wire [63:0] rs2_data;

assign rs1_data = regfile[rs1[3:0]];
assign rs2_data = regfile[rs2[3:0]];

always @(posedge clk) begin
    if (reg_write_r)
        regfile[rd_r[3:0]] <= writeback_data;
end

///////////////////////////////////////////////////////////////////////////////
// ----- LD/ST data memory -----
// Uses your behavioral data_mem_bram module (for simulation)
///////////////////////////////////////////////////////////////////////////////
wire [63:0] mem_load_data;
reg  ld_en, st_en;
reg  [31:0] mem_addr;
reg  [63:0] mem_write_data;

data_mem_bram u_datamem (
    .clk(clk),
    .ld_en(ld_en),
    .ld_addr(mem_addr),
    .ld_data(mem_load_data),
    .st_en(st_en),
    .st_addr(mem_addr),
    .st_data(mem_write_data)
);

///////////////////////////////////////////////////////////////////////////////
// ----- Execution path -----
// integer SIMD ALU uses pipelined/reg inputs
///////////////////////////////////////////////////////////////////////////////
wire [63:0] int_result;
exec_int16x4 u_exec_int (
    .opcode(opcode_r),
    .a(rs1_data_r),
    .b(rs2_data_r),
    .result(int_result)
);

///////////////////////////////////////////////////////////////////////////////
// BF16 tensor unit
// NOTE: to simulate integer-only path we keep a simple stub here.
// Replace this stub with your real tensor_bf16_4lane instantiation later.
// If you do instantiate a real tensor module, remove the assign stubs below.
///////////////////////////////////////////////////////////////////////////////
wire [63:0] tensor_result;
wire tensor_done;

// --- STUB: no tensor activity for now (prevents PC stalls and multiple-driver issues) ---
assign tensor_done   = 1'b0;
assign tensor_result = 64'd0;

// If/when you have the real tensor module, use something like:
// tensor_bf16_4lane u_tensor (
//     .clk(clk), .rst_n(rst_n),
//     .start(tensor_start),
//     .a_reg(rs1_data_r),
//     .b_reg(rs2_data_r),
//     .acc_reg(rs1_data_r),
//     .rd_reg(tensor_result),
//     .done(tensor_done)
// );

reg tensor_start;

///////////////////////////////////////////////////////////////////////////////
// Pipeline registers: capture read data and control signals (Stage1 -> Stage2)
///////////////////////////////////////////////////////////////////////////////
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rs1_data_r   <= 64'd0;
        rs2_data_r   <= 64'd0;
        opcode_r     <= 4'd0;
        dtype_r      <= 4'd0;
        rd_r         <= 4'd0;
        reg_write_r  <= 1'b0;
        is_ld_r      <= 1'b0;
        is_st_r      <= 1'b0;
    end else begin
        rs1_data_r   <= rs1_data;    // BRAM read data arrives 1 cycle after address
        rs2_data_r   <= rs2_data;
        opcode_r     <= opcode;
        dtype_r      <= dtype;
        rd_r         <= rd;
        reg_write_r  <= reg_write;
        is_ld_r      <= is_ld;
        is_st_r      <= is_st;
    end
end

///////////////////////////////////////////////////////////////////////////////
// Memory control and execution-start logic (use pipelined signals).
// This is combinational control for the next cycle's memory ops / tensor start.
///////////////////////////////////////////////////////////////////////////////
always @(*) begin
    // defaults
    ld_en = 1'b0;
    st_en = 1'b0;
    mem_addr = 32'd0;
    mem_write_data = 64'd0;
    tensor_start = 1'b0;

    case (opcode_r)
        4'h0, 4'h1, 4'h4: begin
            // integer ALU - nothing to start here
        end

        4'h2, 4'h3: begin
            // BF16 ops - would start tensor (tensor core must assert done later)
            tensor_start = 1'b1;
        end

        4'h5: begin // LD
            ld_en = 1'b1;
            mem_addr = rs1_data_r[31:0];   // address from rs1 (pipelined)
        end

        4'h6: begin // ST
            st_en = 1'b1;
            mem_addr = rs1_data_r[31:0];   // store address from rs1 (pipelined)
            mem_write_data = rs2_data_r;   // store data from rs2 (pipelined)
        end

        default: ;
    endcase
end

///////////////////////////////////////////////////////////////////////////////
// Load pipeline: mem_load_data is valid one cycle after ld_en asserted; capture it.
// is_ld_rr indicates that on this cycle we should commit mem_load_data_r as writeback.
///////////////////////////////////////////////////////////////////////////////
reg is_ld_rr;
reg [63:0] mem_load_data_r;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        is_ld_rr <= 1'b0;
        mem_load_data_r <= 64'd0;
    end else begin
        is_ld_rr <= is_ld_r;           // delayed indicator for a load that was started previous cycle
        mem_load_data_r <= mem_load_data; // capture data from data BRAM (synchronous read)
    end
end

///////////////////////////////////////////////////////////////////////////////
// ALU result hold register used for arbitration (holds int_result)
///////////////////////////////////////////////////////////////////////////////
reg [63:0] wb_reg;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        wb_reg <= 64'd0;
    end else begin
        wb_reg <= int_result;  // always update with the latest integer ALU result
    end
end

///////////////////////////////////////////////////////////////////////////////
// Writeback arbitration and commit to register-file write data (sequential).
// Priority: load (from memory) > tensor_result (when done) > integer result
///////////////////////////////////////////////////////////////////////////////
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        writeback_data <= 64'd0;
    end else begin
        if (is_ld_rr) begin
            writeback_data <= mem_load_data_r;
        end else if (tensor_done) begin
            writeback_data <= tensor_result;
        end else begin
            writeback_data <= wb_reg;
        end
    end
end

///////////////////////////////////////////////////////////////////////////////
// HALT handling: latch halted state when is_halt asserted by control unit.
// Stall PC while halted or while a tensor op is in progress.
///////////////////////////////////////////////////////////////////////////////
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) halted <= 1'b0;
    else if (is_halt) halted <= 1'b1;
end

assign stall = halted | ((dtype_r == 4'h1) & ~tensor_done);

endmodule

`default_nettype wire