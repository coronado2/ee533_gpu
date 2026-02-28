// GPU Program Counter
`timescale 1ns / 1ps
`include "defines.v"

module prog_counter #(
    parameter PC_WIDTH = `PC_WIDTH
)(
    input                           clk,
    input                           rst_n,
    input                           en,
    output reg [`PC_WIDTH-1:0]      pc_out
);

always @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    pc_out <= {PC_WIDTH{1'b0}};
  end
  else if (en) begin
    pc_out <= pc_out + 1;
  end
end

endmodule
