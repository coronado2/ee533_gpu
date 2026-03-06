// gpu_core.v
// =========================================================================
// FIX 1: rd_rr only latches rd_r when is_ld_r is high.
// FIX 2: Writeback for loads uses dmem_ld_data directly.
// FIX 3: tensor_rd_r latches rd_r at tensor_start, held until tensor_done.
// FIX 4: tensor_done_hold stalls one extra cycle after tensor_done.
// FIX 5: rf_acc_addr uses decode-stage rd (one cycle ahead of EX).
//         acc_lat_r registers rf_acc_data so it is stable at tensor_start.
//         The regfile write-bypass on the acc port ensures the value is
//         captured even when the LD that writes rd completes the same cycle
//         FMAC is being decoded.
// =========================================================================
`timescale 1ns/1ps
`default_nettype none

module gpu_core #(
    parameter CORE_THREAD_BASE = 0
) (
    input  wire        clk,
    input  wire        rst_n,
    output wire [6:0]  imem_addr,
    input  wire [31:0] imem_data,
    output wire [3:0]  rf_rs1_addr,
    input  wire [63:0] rf_rs1_data,
    output wire [3:0]  rf_rs2_addr,
    input  wire [63:0] rf_rs2_data,
    output wire        rf_we,
    output wire [3:0]  rf_wr_addr,
    output wire [63:0] rf_wr_data,
    output wire [3:0]  rf_acc_addr,
    input  wire [63:0] rf_acc_data,
    output reg         dmem_ld_en,
    output reg  [7:0]  dmem_ld_addr,
    input  wire [63:0] dmem_ld_data,
    output reg         dmem_st_en,
    output reg  [7:0]  dmem_st_addr,
    output reg  [63:0] dmem_st_data,
    output wire        halted,
    output wire [31:0] pc_out
);

reg [63:0] rs1_data_r, rs2_data_r;
reg [3:0]  opcode_r, dtype_r, rd_r;
reg        reg_write_r;
reg        is_ld_r, is_st_r;

wire stall;
reg  halted_r;
assign halted = halted_r;

wire [3:0] opcode, dtype, rd, rs1, rs2;
wire       is_ld, is_st, is_halt, reg_write;

control_unit u_ctrl (
    .instruction (imem_data),
    .opcode      (opcode),
    .dtype       (dtype),
    .rd          (rd),
    .rs1         (rs1),
    .rs2         (rs2),
    .is_ld       (is_ld),
    .is_st       (is_st),
    .is_halt     (is_halt),
    .reg_write   (reg_write)
);

wire [31:0] pc_next = halted_r ? pc_out : pc_out + 32'd1;

pc u_pc (
    .clk    (clk),
    .rst_n  (rst_n),
    .stall  (stall),
    .next_pc(pc_next),
    .pc_out (pc_out)
);

assign imem_addr   = pc_out[6:0];
assign rf_rs1_addr = rs1;
assign rf_rs2_addr = rs2;

// FIX 5: read rd one cycle early (decode stage) so the registered
//         value is ready when tensor_start fires in EX stage.
assign rf_acc_addr = rd;

reg        wb_we;
reg [3:0]  wb_rd;
reg [63:0] wb_data;

assign rf_we      = wb_we & (wb_rd != 4'hF);
assign rf_wr_addr = wb_rd;
assign rf_wr_data = wb_data;

wire [63:0] int_result;

exec_int16x4 u_exec_int (
    .opcode (opcode_r),
    .a      (rs1_data_r),
    .b      (rs2_data_r),
    .result (int_result)
);

wire tensor_op_r = (opcode_r == 4'h2) || (opcode_r == 4'h3);
reg  tensor_busy;
reg  tensor_done_hold;

wire [2:0] tensor_op_sel = (opcode_r == 4'h2) ? 3'b010 :
                            (opcode_r == 4'h3) ? 3'b011 :
                                                 3'b000;

wire tensor_start_wire = tensor_op_r & ~tensor_busy;

wire [63:0] tensor_result;
wire        tensor_done;

// FIX 5: register rf_acc_data so it is stable at tensor_start.
//         rf_acc_addr=rd (decode), so this captures the accumulator
//         the cycle before FMAC enters EX.
reg [63:0] acc_lat;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) acc_lat <= 64'd0;
    else        acc_lat <= rf_acc_data;
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

// FIX 3: preserve rd at tensor launch
reg [3:0] tensor_rd_r;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)                 tensor_rd_r <= 4'd0;
    else if (tensor_start_wire) tensor_rd_r <= rd_r;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        tensor_busy      <= 1'b0;
        tensor_done_hold <= 1'b0;
    end else begin
        if (tensor_start_wire) tensor_busy <= 1'b1;
        else if (tensor_done)  tensor_busy <= 1'b0;
        tensor_done_hold <= tensor_done;
    end
end

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
        reg_write_r <= reg_write & ~halted_r;
        is_ld_r     <= is_ld;
        is_st_r     <= is_st;
    end
end

always @(*) begin
    dmem_ld_en   = 1'b0;
    dmem_ld_addr = 8'd0;
    dmem_st_en   = 1'b0;
    dmem_st_addr = 8'd0;
    dmem_st_data = 64'd0;

    case (opcode_r)
        4'h5: begin
            dmem_ld_en   = 1'b1;
            dmem_ld_addr = rs1_data_r[10:3];
        end
        4'h6: begin
            dmem_st_en   = 1'b1;
            dmem_st_addr = rs1_data_r[10:3];
            dmem_st_data = rs2_data_r;
        end
        default: ;
    endcase
end

// FIX 1 + FIX 2: LD writeback path
reg        is_ld_rr;
reg [3:0]  rd_rr;
reg        reg_write_rr;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        is_ld_rr     <= 1'b0;
        rd_rr        <= 4'd0;
        reg_write_rr <= 1'b0;
    end else begin
        is_ld_rr     <= is_ld_r;
        if (is_ld_r) rd_rr <= rd_r;
        reg_write_rr <= reg_write_r & is_ld_r;
    end
end

reg [63:0] wb_alu;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) wb_alu <= 64'd0;
    else        wb_alu <= int_result;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) halted_r <= 1'b0;
    else if (is_halt) halted_r <= 1'b1;
end

assign stall = halted_r | tensor_busy | tensor_done_hold;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        wb_we   <= 1'b0;
        wb_rd   <= 4'd0;
        wb_data <= 64'd0;
    end else begin
        if (reg_write_rr) begin
            wb_we   <= 1'b1;
            wb_rd   <= rd_rr;
            wb_data <= dmem_ld_data;
        end else if (tensor_done) begin
            wb_we   <= 1'b1;
            wb_rd   <= tensor_rd_r;
            wb_data <= tensor_result;
        end else if (reg_write_r & ~is_ld_r) begin
            wb_we   <= 1'b1;
            wb_rd   <= rd_r;
            wb_data <= int_result;
        end else begin
            wb_we   <= 1'b0;
            wb_rd   <= 4'd0;
            wb_data <= 64'd0;
        end
    end
end

endmodule
`default_nettype wire