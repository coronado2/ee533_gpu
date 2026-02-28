// data_mem_wrap.v
// Read port muxed: GPU load takes priority when active, host reads otherwise.
// Port interface identical to data_mem.v ? only change module name in datapath.v.
`timescale 1ns/1ps
`default_nettype none

module data_mem_wrap (
    input  wire        clk,
    // Write port
    input  wire        wr_en,
    input  wire [7:0]  wr_addr,
    input  wire [63:0] wr_data,
    // Read port A ? GPU LD
    input  wire        rd_en_a,
    input  wire [7:0]  rd_addr_a,
    output wire [63:0] rd_data_a,
    // Read port B ? host readback
    input  wire        rd_en_b,
    input  wire [7:0]  rd_addr_b,
    output wire [63:0] rd_data_b
);

// Mux read address ? GPU load takes priority
wire [7:0]  rd_addr_mux = rd_en_a ? rd_addr_a : rd_addr_b;
wire [63:0] rd_data_mux;

// Both ports see the same data out ? each only uses it when it was the requester
assign rd_data_a = rd_data_mux;
assign rd_data_b = rd_data_mux;

data_mem_ip u_bram (
    // Port A ? write
    .clka  (clk),
    .ena   (1'b1),
    .wea   (wr_en),
    .addra (wr_addr),
    .dina  (wr_data),
    // Port B ? muxed read
    .clkb  (clk),
    .enb   (rd_en_a | rd_en_b),
    .addrb (rd_addr_mux),
    .doutb (rd_data_mux)
);

endmodule
`default_nettype wire