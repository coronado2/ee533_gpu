// defines.v
// =========================================================================
// Address map for the GPU register bus.
// Follows the same convention as the NetFPGA generic_regs pattern.
// =========================================================================

// Register bus widths (match your platform)
`define UDP_REG_ADDR_WIDTH   23
`define CPCI_NF2_DATA_WIDTH  32

// Block tag: upper bits of the register address that select this module.
// Adjust to match whatever address space your platform assigns.
`define TOP_BLOCK_ADDR       20'hA0000     // base address of this block
`define TOP_REG_ADDR_WIDTH   8             // 256 word-addressed registers in block

// =========================================================================
// Software register word offsets  (host writes these)
// =========================================================================
`define GPU_RST          0   // [0]    write 1 to hold reset, 0 to release
`define GPU_START        1   // [0]    pulse 1 to start execution
`define IMEM_WR_EN       2   // [0]    1 = enable instruction memory write
`define IMEM_WR_ADDR     3   // [6:0]  word address to write (0-127)
`define IMEM_WR_DATA     4   // [31:0] 32-bit instruction word
`define DMEM_WR_EN       5   // [0]    1 = enable data memory write
`define DMEM_WR_ADDR     6   // [7:0]  word address to write (0-255)
`define DMEM_WR_DATA_LO  7   // [31:0] lower 32 bits of 64-bit write data
`define DMEM_WR_DATA_HI  8   // [31:0] upper 32 bits of 64-bit write data
`define DMEM_RD_ADDR     9   // [7:0]  word address to read back

// =========================================================================
// Hardware register word offsets  (hardware writes these, host reads)
// =========================================================================
`define HW_HALTED        0   // [0]    1 when GPU has executed HALT
`define HW_PC            1   // [6:0]  current program counter value
`define HW_RESULT_LO     2   // [31:0] last writeback value bits [31:0]
`define HW_RESULT_HI     3   // [31:0] last writeback value bits [63:32]
`define HW_DMEM_RD_LO    4   // [31:0] data memory readback bits [31:0]
`define HW_DMEM_RD_HI    5   // [31:0] data memory readback bits [63:32]
