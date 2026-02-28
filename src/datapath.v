`timescale 1ns / 1ps
`include "defines.v"

module datapath (
    input wire          clk,
    input wire          rst_n,
    
    // IMEM
    input wire  [`INSTR_WIDTH-1:0]           imem_data_in,
    output wire [`INSTR_WIDTH-1:0]           imem_addr_out,

    //DMEM
    input wire  [`DATA_WIDTH-1:0]            dmem_data_in,
    output wire [`D_MEM_ADDR_WIDTH-1:0]      dmem_addr_out,
    output wire [`DATA_WIDTH-1:0]            dmem_data_out,
    output wire                             dmem_wena    
);

//============================= Signals ===================================
// Contorl
wire [`OPCODE-1:0] opcode_ctrl;
wire [`DTYPE-1:0] dtype_ctrl;
wire [`REG_ADDR_WIDTH-1:0] rd_ctrl;
wire [`REG_ADDR_WIDTH-1:0] rs1_ctrl;
wire [`REG_ADDR_WIDTH-1:0] rs2_ctrl;
wire [`DATA_WIDTH-1:0] rs1_data_ctrl;
wire [`DATA_WIDTH-1:0] rs2_data_ctrl;
wire [`DATA_WIDTH-1:0] imm_ctrl;
wire is_ld_ctrl;
wire is_vfma;
wire is_st_ctrl;
wire is_halt_ctrl;
wire regwrite_ctrl;

// EX
wire [`OPCODE-1:0] opcode_ex;
wire [`DTYPE-1:0] dtype_ex;
wire [`REG_ADDR_WIDTH-1:0] rd_ex;
wire [`DATA_WIDTH-1:0] rs1_data_ex;
wire [`DATA_WIDTH-1:0] rs2_data_ex;
wire [`DATA_WIDTH-1:0] imm_ex;
wire is_ld_ex;
wire is_st_ex;
wire is_halt_ex;
wire regwrite_ex;
wire [`DATA_WIDTH-1:0] int_result_ex;
wire [`DATA_WIDTH-1:0] tensor_result_ex;
wire start;
reg [`DATA_WIDTH-1:0] acc_lat;
reg fma_acc_phase;
wire tensor_busy;
wire done = 1'b0;

// MEM

// WB
wire wregen_wb; // pending logic
wire [`REG_ADDR_WIDTH-1:0] rd_wb;
wire [`DATA_WIDTH-1:0] write_data;

//============================== Logic ====================================

// Contorl Unit
control_unit u_ctrl (
    .instruction(imem_data_in),
    .opcode(opcode_ctrl),
    .dtype(dtype_ctrl),
    .rd(rd_ctrl),
    .rs1(rs1_ctrl),
    .rs2(rs2_ctrl),
    .imm(imm_ctrl),
    .is_ld(is_ld_ctrl),
    .is_vfma(is_vfma),
    .is_st(is_st_ctrl),
    .is_halt(is_halt_ctrl),
    .reg_write(regwrite_ctrl)
);

wire [`REG_ADDR_WIDTH-1:0] rs1_sel = (fma_acc_phase) ? rd_ex : rs1_ctrl;

regfile u_regfile (
    .clk(clk),
    .rst_n(rst_n),
    .wena(wregen_wb),
    .waddr(rd_wb),
    .wdata(write_data),
    .rs1addr(rs1_sel),
    .rs2addr(rs2_ctrl),
    .rs1data(rs1_data_ctrl),
    .rs2data(rs2_data_ctrl)
);

pipeline_reg #(.REGS(4+`OPCODE+`DTYPE+`REG_ADDR_WIDTH+`DATA_WIDTH+`DATA_WIDTH+`DATA_WIDTH)) ctrl_ex_stage(
    .clk(clk),
    .rst_n(pipe_en),
    .en(1'b1),
    .D({is_ld_ctrl, is_st_ctrl, is_halt_ctrl, regwrite_ctrl, opcode_ctrl, 
        dtype_ctrl, rd_ctrl, rs1_data_ctrl, rs2_data_ctrl, imm_ctrl}),
    .Q({is_ld_ex, is_st_ex, is_halt_ex, regwrite_ex, opcode_ex,
        dtype_ex, rd_ex, rs1_data_ex, rs2_data_ex, imm_ex})
);

// EX
wire stall = fma_acc_phase || tensor_busy;
wire pipe_en = ~stall;

exec_int16x4 u_int(
    .opcode(opcode_ex),
    .a(rs1_data_ex),
    .b(rs2_data_ex), // temp, will use mux to select imm if needed
    .result(int_result_ex)
);

tensor u_tensor(
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .op(opcode_ex[2:0]),
    .a_reg(rs1_data_ex),
    .b_reg(rs2_data_ex),
    .acc_reg(),
    .rd_reg(tensor_result_ex),
    .done(done)
);

//MEM


//WB

endmodule