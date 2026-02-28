// bf16_add.v  (fixed)
// BFloat16 adder/subtractor.
// Format: [15]=sign, [14:7]=exponent (bias 127), [6:0]=mantissa (7 bits)
// Latency: 3 clock cycles. done pulses high for 1 cycle when result is valid.
//
// Mantissa representation used internally:
//   8 bits = { hidden_1, mantissa[6:0] }  stored in 8-bit regs
//   bit7 = hidden 1 (implicit leading bit for normalised numbers)
//   bits[6:0] = stored mantissa field
//
// Accumulator (Stage 2):
//   9 bits = { carry_bit, 8-bit_mantissa }
//   bit8 = carry out from addition  (subtraction never produces borrow since big>=small)
//
// Normalisation (Stage 3):
//   Carry (bit8=1): right-shift 1, exp+1, mantissa = raw[7:1]
//   No carry (bit8=0): LZC over raw[7:0] to bring leading-1 to bit7,
//                      shift left by lzc, exp -= lzc, mantissa = norm[6:0]

module bf16_add (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg  [15:0] y,
    output reg         done
);

// ?? Field extraction ??????????????????????????????????????????????????????????
wire        sa = a[15];
wire [7:0]  ea = a[14:7];
wire [6:0]  ma = a[6:0];

wire        sb = b[15];
wire [7:0]  eb = b[14:7];
wire [6:0]  mb = b[6:0];

// ?? Special-case flags ????????????????????????????????????????????????????????
wire a_zero = (ea == 8'd0);
wire b_zero = (eb == 8'd0);
wire a_inf  = (ea == 8'hFF) && (ma == 7'd0);
wire b_inf  = (eb == 8'hFF) && (mb == 7'd0);
wire a_nan  = (ea == 8'hFF) && (ma != 7'd0);
wire b_nan  = (eb == 8'hFF) && (mb != 7'd0);

// ?????????????????????????????????????????????????????????????????????????????
// STAGE 1 ? align mantissas
// ?????????????????????????????????????????????????????????????????????????????
reg        s1_valid;
reg        s1_sign_big, s1_sign_sml;
reg [7:0]  s1_exp;
reg [7:0]  s1_big_mant;   // {hidden_1, mantissa[6:0]}
reg [7:0]  s1_sml_mant;   // aligned smaller mantissa
reg        s1_zero, s1_inf, s1_nan;
reg        s1_inf_sign;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        s1_valid    <= 1'b0;
        s1_sign_big <= 1'b0;
        s1_sign_sml <= 1'b0;
        s1_exp      <= 8'd0;
        s1_big_mant <= 8'd0;
        s1_sml_mant <= 8'd0;
        s1_zero     <= 1'b0;
        s1_inf      <= 1'b0;
        s1_nan      <= 1'b0;
        s1_inf_sign <= 1'b0;
    end else begin
        s1_valid    <= start;
        s1_nan      <= a_nan | b_nan | (a_inf & b_inf & (sa ^ sb));
        s1_zero     <= (a_zero & b_zero);
        s1_inf      <= (a_inf | b_inf) & ~(a_nan | b_nan);
        s1_inf_sign <= a_inf ? sa : sb;

        begin : align
            reg [7:0] diff;
            reg [7:0] big_m8, sml_m8;
            reg       sgn_big, sgn_sml;
            reg [7:0] exp_out;
            reg [7:0] shifted;

            if (ea >= eb) begin
                exp_out = ea;
                big_m8  = a_zero ? 8'd0 : {1'b1, ma};
                sml_m8  = b_zero ? 8'd0 : {1'b1, mb};
                sgn_big = sa;
                sgn_sml = sb;
                diff    = ea - eb;
            end else begin
                exp_out = eb;
                big_m8  = b_zero ? 8'd0 : {1'b1, mb};
                sml_m8  = a_zero ? 8'd0 : {1'b1, ma};
                sgn_big = sb;
                sgn_sml = sa;
                diff    = eb - ea;
            end

            // Right-shift smaller mantissa to align exponents; cap at 8
            shifted = (diff >= 8'd8) ? 8'd0 : (sml_m8 >> diff);

            s1_exp      <= exp_out;
            s1_big_mant <= big_m8;
            s1_sml_mant <= shifted;
            s1_sign_big <= sgn_big;
            s1_sign_sml <= sgn_sml;
        end
    end
end

// ?????????????????????????????????????????????????????????????????????????????
// STAGE 2 ? add or subtract aligned mantissas
// s2_mant_raw is 9 bits: bit8 = carry (add) or 0 (sub), bits[7:0] = 8-bit result
// ?????????????????????????????????????????????????????????????????????????????
reg        s2_valid;
reg [7:0]  s2_exp;
reg        s2_res_sign;
reg [8:0]  s2_mant_raw;
reg        s2_zero, s2_inf, s2_nan;
reg        s2_inf_sign;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        s2_valid    <= 1'b0;
        s2_exp      <= 8'd0;
        s2_res_sign <= 1'b0;
        s2_mant_raw <= 9'd0;
        s2_zero     <= 1'b0;
        s2_inf      <= 1'b0;
        s2_nan      <= 1'b0;
        s2_inf_sign <= 1'b0;
    end else begin
        s2_valid    <= s1_valid;
        s2_exp      <= s1_exp;
        s2_zero     <= s1_zero;
        s2_inf      <= s1_inf;
        s2_nan      <= s1_nan;
        s2_inf_sign <= s1_inf_sign;
        s2_res_sign <= s1_sign_big;

        if (s1_sign_big == s1_sign_sml) begin
            // Same sign ? add; carry lands at bit8
            s2_mant_raw <= {1'b0, s1_big_mant} + {1'b0, s1_sml_mant};
        end else begin
            // Opposite signs ? subtract (big - small, result ? 0)
            s2_mant_raw <= {1'b0, s1_big_mant} - {1'b0, s1_sml_mant};
        end
    end
end

// ?????????????????????????????????????????????????????????????????????????????
// STAGE 3 ? normalise and pack
// ?????????????????????????????????????????????????????????????????????????????

// LZC for 8-bit value: counts leading zeros from bit7 down.
// Returns how many left-shifts bring the leading '1' to bit7.
function [3:0] lzc8;
    input [7:0] v;
    begin
        if      (v[7]) lzc8 = 4'd0;
        else if (v[6]) lzc8 = 4'd1;
        else if (v[5]) lzc8 = 4'd2;
        else if (v[4]) lzc8 = 4'd3;
        else if (v[3]) lzc8 = 4'd4;
        else if (v[2]) lzc8 = 4'd5;
        else if (v[1]) lzc8 = 4'd6;
        else if (v[0]) lzc8 = 4'd7;
        else           lzc8 = 4'd8;  // zero
    end
endfunction

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        y    <= 16'd0;
        done <= 1'b0;
    end else begin
        done <= s2_valid;

        if (s2_valid) begin
            if (s2_nan) begin
                y <= 16'hFF81;
            end else if (s2_inf) begin
                y <= {s2_inf_sign, 8'hFF, 7'd0};
            end else if (s2_zero || (s2_mant_raw == 9'd0)) begin
                y <= {s2_res_sign, 15'd0};
            end else begin : normalise
                reg [3:0]  lzc;
                reg [7:0]  norm_m8;
                reg [8:0]  exp_adj;

                if (s2_mant_raw[8]) begin
                    // ?? Carry from addition ???????????????????????????????????
                    // raw = 1_hhhhhhhh (9 bits); true value = 1.hhhhhhh * 2^(exp+1)
                    // Shift right 1: hidden bit moves to bit7 of the 8-bit field
                    norm_m8 = s2_mant_raw[8:1];      // {carry=1, raw[7:1]}
                    exp_adj = {1'b0, s2_exp} + 9'd1;
                end else begin
                    // ?? No carry / subtraction ????????????????????????????????
                    // raw[7:0]: find leading 1, left-shift to bit7
                    lzc     = lzc8(s2_mant_raw[7:0]);
                    norm_m8 = s2_mant_raw[7:0] << lzc;
                    if ({5'd0, lzc} <= {1'b0, s2_exp})
                        exp_adj = {1'b0, s2_exp} - {5'd0, lzc};
                    else
                        exp_adj = 9'd0;  // underflow ? zero
                end

                // ?? Pack ??????????????????????????????????????????????????????
                if (exp_adj == 9'd0 || norm_m8 == 8'd0) begin
                    y <= {s2_res_sign, 15'd0};
                end else if (exp_adj >= 9'd255) begin
                    y <= {s2_res_sign, 8'hFF, 7'd0};
                end else begin
                    // norm_m8[7] = hidden 1 (drop it); norm_m8[6:0] = mantissa
                    y <= {s2_res_sign, exp_adj[7:0], norm_m8[6:0]};
                end
            end
        end
    end
end

endmodule