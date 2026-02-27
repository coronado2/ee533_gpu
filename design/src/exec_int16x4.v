// exec_int16x4.v
module exec_int16x4 (
    input  wire [3:0]  opcode,
    input  wire [63:0] a,
    input  wire [63:0] b,
    output reg  [63:0] result
);

wire signed [15:0] a0 = a[15:0];
wire signed [15:0] a1 = a[31:16];
wire signed [15:0] a2 = a[47:32];
wire signed [15:0] a3 = a[63:48];

wire signed [15:0] b0 = b[15:0];
wire signed [15:0] b1 = b[31:16];
wire signed [15:0] b2 = b[47:32];
wire signed [15:0] b3 = b[63:48];

reg signed [15:0] r0, r1, r2, r3;

always @(*) begin
    case (opcode)
        4'h0: begin // VADD
            r0 = a0 + b0;
            r1 = a1 + b1;
            r2 = a2 + b2;
            r3 = a3 + b3;
        end

        4'h1: begin // VSUB
            r0 = a0 - b0;
            r1 = a1 - b1;
            r2 = a2 - b2;
            r3 = a3 - b3;
        end

        4'h4: begin // RELU
            r0 = (a0 > 0) ? a0 : 0;
            r1 = (a1 > 0) ? a1 : 0;
            r2 = (a2 > 0) ? a2 : 0;
            r3 = (a3 > 0) ? a3 : 0;
        end

        default: begin
            r0 = 16'd0;
            r1 = 16'd0;
            r2 = 16'd0;
            r3 = 16'd0;
        end
    endcase

    result = {r3, r2, r1, r0};
end

endmodule