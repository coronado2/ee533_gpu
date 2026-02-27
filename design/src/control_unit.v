// control_unit.v
module control_unit (
    input  wire [31:0] instruction,

    output wire [3:0]  opcode,
    output wire [3:0]  dtype,
    output wire [3:0]  rd,
    output wire [3:0]  rs1,
    output wire [3:0]  rs2,
    output wire        is_ld,
    output wire        is_st,
    output wire        is_halt,
    output wire        reg_write  // true when instruction writes register file
);

assign opcode = instruction[31:28];
assign dtype  = instruction[27:24];
assign rd     = instruction[23:20];
assign rs1    = instruction[19:16];
assign rs2    = instruction[15:12];

assign is_ld  = (opcode == 4'h5);
assign is_st  = (opcode == 4'h6);
assign is_halt= (opcode == 4'hF);

// For now: all non-store, non-halt instructions write result back
assign reg_write = ~is_st & ~is_halt;

endmodule