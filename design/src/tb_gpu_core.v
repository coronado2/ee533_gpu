`timescale 1ns/1ps

module tb_gpu_core;

reg clk;
reg rst_n;

gpu_core uut (
    .clk(clk),
    .rst_n(rst_n)
);

// Clock generation (100 MHz)
always #5 clk = ~clk;

initial begin
    clk = 0;
    rst_n = 0;

    // Hold reset
    #20;
    rst_n = 1;

    // Run for some cycles
    #500;
end

endmodule