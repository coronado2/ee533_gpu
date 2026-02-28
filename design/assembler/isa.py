"""
isa.py  –  Single source of truth for the GPU ISA.

Instruction encoding (32-bit), exactly as decoded by control_unit.v:

  Bits [31:28]  opcode   (4 bits)
  Bits [27:24]  dtype    (4 bits)   0 = INT16, 1 = BF16
  Bits [23:20]  rd       (4 bits)   destination register
  Bits [19:16]  rs1      (4 bits)   source register 1  (also LD/ST base address)
  Bits [15:12]  rs2      (4 bits)   source register 2  (also ST write-data)
  Bits [11: 0]  (reserved / zero)  12 bits

Opcodes  (control_unit.v + exec_int16x4.v + gpu_core.v dispatch):

  0x0  VADD  – packed int16 add          rd = rs1 + rs2        (dtype=INT16)
  0x1  VSUB  – packed int16 subtract     rd = rs1 - rs2        (dtype=INT16)
  0x2  VMUL  – packed BF16 multiply      rd = rs1 * rs2        (dtype=BF16, tensor unit)
  0x3  FMAC  – packed BF16 fused MAC     rd = rs1*rs2 + rd     (dtype=BF16, tensor unit)
  0x4  RELU  – packed int16 ReLU         rd = max(0, rs1)      (dtype=INT16)
  0x5  LD    – load 64-bit word          rd = mem[rs1]         (addr from rs1 register)
  0x6  ST    – store 64-bit word         mem[rs1] = rs2        (addr from rs1, data from rs2)
  0xF  HALT  – stop execution

Hardware notes (from gpu_core.v):

  - INT16 ops (0x0,0x1,0x4):  routed to exec_int16x4, 1-cycle result.
  - BF16 ops  (0x2,0x3):      routed to tensor_bf16_4lane (dtype_r==1 stalls PC
                               until tensor_done is asserted).
  - LD/ST address:            mem_addr = rs1_data_r[31:0]   (no immediate offset
                               added in hardware; set rs1 to desired byte address).
  - PC:                       increments by 1 each non-stall cycle (word addressing).
                               imem indexed as imem[pc[9:2]], meaning pc must be a
                               WORD address (0,1,2,...); the toolchain handles this.
  - Register file:            16 × 64-bit registers r0–r15.
  - Memory:                   64-bit words; byte address / 8 = word index.
"""

# ── Opcode table ──────────────────────────────────────────────────────────────
OPCODES = {
    #  name     code   dtype     description
    "VADD":  (0x0, "INT16"),
    "VSUB":  (0x1, "INT16"),
    "VMUL":  (0x2, "BF16"),
    "FMAC":  (0x3, "BF16"),
    "RELU":  (0x4, "INT16"),
    "LD":    (0x5, "INT16"),   # dtype field ignored by hardware for LD/ST
    "ST":    (0x6, "INT16"),
    "HALT":  (0xF, "INT16"),
}

OPCODE_TO_NAME = {v[0]: k for k, v in OPCODES.items()}

DTYPE_ENC = {"INT16": 0x0, "BF16": 0x1}
DTYPE_DEC = {0x0: "INT16", 0x1: "BF16"}

NUM_REGS   = 16      # r0 – r15
IMEM_DEPTH = 256     # instruction memory words
DMEM_WORDS = 512     # data BRAM words (64-bit each)
HALT_WORD  = 0xF0000000  # HALT instruction used to pad unused imem

# ── Encoding helpers ──────────────────────────────────────────────────────────

def encode(opcode: int, dtype: int, rd: int, rs1: int, rs2: int) -> int:
    """
    Pack one 32-bit instruction word.  Bits [11:0] are always zero (reserved).
    Mirrors the field layout in control_unit.v exactly.
    """
    assert 0 <= opcode <= 0xF, f"opcode out of range: {opcode:#x}"
    assert 0 <= dtype  <= 0xF, f"dtype out of range: {dtype:#x}"
    assert 0 <= rd     <= 0xF, f"rd out of range: {rd}"
    assert 0 <= rs1    <= 0xF, f"rs1 out of range: {rs1}"
    assert 0 <= rs2    <= 0xF, f"rs2 out of range: {rs2}"
    return (opcode << 28) | (dtype << 24) | (rd << 20) | (rs1 << 16) | (rs2 << 12)


def decode(word: int) -> dict:
    """Unpack a 32-bit instruction word into its fields."""
    return {
        "opcode": (word >> 28) & 0xF,
        "dtype":  (word >> 24) & 0xF,
        "rd":     (word >> 20) & 0xF,
        "rs1":    (word >> 16) & 0xF,
        "rs2":    (word >> 12) & 0xF,
    }


def disasm(word: int) -> str:
    """Human-readable disassembly of one instruction word."""
    d = decode(word)
    name  = OPCODE_TO_NAME.get(d["opcode"], f"OP{d['opcode']:X}")
    dtype = DTYPE_DEC.get(d["dtype"], f"T{d['dtype']:X}")
    oc    = d["opcode"]

    if oc == OPCODES["HALT"][0]:
        return "HALT"
    if oc == OPCODES["LD"][0]:
        return f"LD       r{d['rd']}, [r{d['rs1']}]"
    if oc == OPCODES["ST"][0]:
        return f"ST       [r{d['rs1']}], r{d['rs2']}"
    if oc == OPCODES["RELU"][0]:
        return f"RELU     r{d['rd']}, r{d['rs1']}"
    if oc == OPCODES["FMAC"][0]:
        return f"FMAC     r{d['rd']}, r{d['rs1']}, r{d['rs2']}   ; rd = rs1*rs2 + rd"
    return f"{name:<6}   r{d['rd']}, r{d['rs1']}, r{d['rs2']}"


def instruction(mnemonic: str, rd=0, rs1=0, rs2=0) -> int:
    """
    Convenience: build an instruction word from a mnemonic string.
    dtype is determined by the opcode's natural type (see OPCODES table).
    """
    mn = mnemonic.upper()
    if mn not in OPCODES:
        raise ValueError(f"Unknown mnemonic: {mnemonic!r}")
    opcode, dtype_str = OPCODES[mn]
    return encode(opcode, DTYPE_ENC[dtype_str], rd, rs1, rs2)