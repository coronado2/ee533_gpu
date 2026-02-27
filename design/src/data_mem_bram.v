// data_mem_bram.v
module data_mem_bram (
    input  wire        clk,

    // Load
    input  wire        ld_en,
    input  wire [31:0] ld_addr,   // byte address
    output reg  [63:0] ld_data,

    // Store
    input  wire        st_en,
    input  wire [31:0] st_addr,
    input  wire [63:0] st_data
);

// Instantiate a Block RAM for data memory named data_bram in Vivado
// For the skeleton, implement behavioral mem for simulation

localparam MEM_WORDS = 1024;
reg [63:0] mem [0:MEM_WORDS-1];
wire [31:0] word_addr_ld = ld_addr[11:3];  // assume 8-byte aligned
wire [31:0] word_addr_st = st_addr[11:3];

always @(posedge clk) begin
    if (st_en)
        mem[word_addr_st] <= st_data;
    if (ld_en)
        ld_data <= mem[word_addr_ld];
end

endmodule