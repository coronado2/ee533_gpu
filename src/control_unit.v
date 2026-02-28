// control_unit.v
module control_unit (
    input  wire [31:0] instruction,

    output wire [3:0]  opcode,
    output wire [3:0]  dtype,
    output wire [3:0]  rd,
    output wire [3:0]  rs1,
    output wire [3:0]  rs2,
    output wire [63:0] imm,
    output wire        is_ld,
    output wire        is_vfma,
    output wire        is_st,
    output wire        is_halt,
    output wire        reg_write  // true when instruction writes register file
);
//Opcodes
localparam OP_VADD  = 4'h0;
localparam OP_VSUB  = 4'h1;
localparam OP_VMUL  = 4'h2;
localparam OP_VFMA  = 4'h3;
localparam OP_RELU  = 4'h4;
localparam OP_LD    = 4'h5;
localparam OP_ST    = 4'h6;
localparam OP_HALT  = 4'hF;

assign opcode = instruction[31:28];
assign dtype  = instruction[27:24];
assign rd     = instruction[23:20];
assign rs1    = instruction[19:16];
assign rs2    = instruction[15:12];

assign imm = {{52{instruction[11]}}, instruction[11:0]};

assign is_ld  =  (opcode == OP_LD);
assign is_vfma = (opcode == OP_VFMA);
assign is_st  =  (opcode == OP_ST);
assign is_halt=  (opcode == OP_HALT);

// Reg Write for EX and LD
assign reg_write = (opcode == OP_VADD) | (opcode == OP_VSUB) |
                   (opcode == OP_VMUL) | (opcode == OP_VFMA) |
                   (opcode == OP_RELU) | (opcode == OP_LD);

endmodule