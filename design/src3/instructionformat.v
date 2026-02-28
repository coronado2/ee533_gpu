localparam OP_VADD  = 4'h0;
localparam OP_VSUB  = 4'h1;
localparam OP_VMUL  = 4'h2;
localparam OP_VFMA  = 4'h3;
localparam OP_RELU  = 4'h4;
localparam OP_LD    = 4'h5;
localparam OP_ST    = 4'h6;
localparam OP_HALT  = 4'hF;

// Instruction Format

/*
[31:28] opcode
[27:24] dtype
[23:20] rd
[19:16] rs1
[15:12] rs2
[11:0]  immediate
*/
