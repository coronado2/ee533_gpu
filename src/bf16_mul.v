`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    16:54:55 02/26/2026 
// Design Name: 
// Module Name:    bf16_mul 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
// bf16_mul.v
// BFloat16 multiplier: 1 sign bit, 8 exponent bits, 7 mantissa bits
// Format: [15] = sign, [14:7] = exponent (bias 127), [6:0] = mantissa
// Latency: 2 clock cycles, done pulses high for 1 cycle when result is valid.
// Special cases handled: zero, infinity, NaN propagation.

module bf16_mul (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,   // pulse high for 1 cycle to begin
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg  [15:0] y,
    output reg         done
);

// ?? field extraction ?????????????????????????????????????????????????????????
wire        sa = a[15];
wire [7:0]  ea = a[14:7];
wire [6:0]  ma = a[6:0];

wire        sb = b[15];
wire [7:0]  eb = b[14:7];
wire [6:0]  mb = b[6:0];

// ?? special-case flags ????????????????????????????????????????????????????????
wire a_zero  = (ea == 8'd0);
wire b_zero  = (eb == 8'd0);
wire a_inf   = (ea == 8'hFF) && (ma == 7'd0);
wire b_inf   = (eb == 8'hFF) && (mb == 7'd0);
wire a_nan   = (ea == 8'hFF) && (ma != 7'd0);
wire b_nan   = (eb == 8'hFF) && (mb != 7'd0);

// ?? stage 1 registers ?????????????????????????????????????????????????????????
// Compute sign, detect specials, begin mantissa multiply
reg        s1_sign;
reg [7:0]  s1_ea, s1_eb;
reg [7:0]  s1_ma_full;   // 1.mantissa as 8 bits (hidden bit prepended)
reg [7:0]  s1_mb_full;
reg        s1_zero, s1_inf, s1_nan;
reg        s1_valid;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        s1_valid <= 1'b0;
        s1_sign  <= 1'b0;
        s1_zero  <= 1'b0;
        s1_inf   <= 1'b0;
        s1_nan   <= 1'b0;
        s1_ea    <= 8'd0;
        s1_eb    <= 8'd0;
        s1_ma_full <= 8'd0;
        s1_mb_full <= 8'd0;
    end else begin
        s1_valid   <= start;
        s1_sign    <= sa ^ sb;
        s1_zero    <= a_zero | b_zero;
        s1_inf     <= (a_inf | b_inf) & ~a_nan & ~b_nan;
        s1_nan     <= a_nan | b_nan | (a_inf & b_zero) | (b_inf & a_zero);
        s1_ea      <= ea;
        s1_eb      <= eb;
        // Prepend hidden 1-bit for normal numbers (denormals treated as zero above)
        s1_ma_full <= {1'b1, ma};
        s1_mb_full <= {1'b1, mb};
    end
end

// ?? stage 2: multiply mantissas, compute exponent ????????????????????????????
// 8×8 unsigned product ? 16 bits
wire [15:0] mant_product = s1_ma_full * s1_mb_full;

// Exponent: e_a + e_b - 127 (remove one bias)
// Use 9-bit signed to catch under/overflow
wire [8:0] exp_sum = {1'b0, s1_ea} + {1'b0, s1_eb} - 9'd127;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        y    <= 16'd0;
        done <= 1'b0;
    end else begin
        done <= s1_valid;  // result is ready this cycle when s1_valid

        if (s1_valid) begin
            if (s1_nan) begin
                // Canonical quiet NaN
                y <= 16'hFF81;
            end else if (s1_inf) begin
                y <= {s1_sign, 8'hFF, 7'd0};
            end else if (s1_zero) begin
                y <= {s1_sign, 15'd0};
            end else begin
                // Normalise: if bit15 of product is set, shift right & inc exponent
                // mant_product[15] = 1  ? result mantissa is mant_product[14:8] (7 bits)
                // mant_product[14] = 1  ? result mantissa is mant_product[13:7]
                if (mant_product[15]) begin
                    // Overflow in mantissa multiply: shift right 1
                    // exp_sum + 1 because of the extra bit
                    if (exp_sum[8] || (exp_sum + 9'd1 >= 9'd255)) begin
                        // Overflow ? infinity
                        y <= {s1_sign, 8'hFF, 7'd0};
                    end else if (exp_sum == 9'd0 || exp_sum[8]) begin
                        // Underflow ? zero
                        y <= {s1_sign, 15'd0};
                    end else begin
                        y <= {s1_sign, exp_sum[7:0] + 8'd1, mant_product[14:8]};
                    end
                end else begin
                    if (exp_sum[8] || exp_sum == 9'd0) begin
                        y <= {s1_sign, 15'd0};
                    end else if (exp_sum >= 9'd255) begin
                        y <= {s1_sign, 8'hFF, 7'd0};
                    end else begin
                        y <= {s1_sign, exp_sum[7:0], mant_product[13:7]};
                    end
                end
            end
        end
    end
end

endmodule