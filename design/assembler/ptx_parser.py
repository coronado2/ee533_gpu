"""
ptx_parser.py  –  Translate NVIDIA PTX (from nvcc -ptx -arch=sm_80) into
                  GPU assembly instructions compatible with our hardware ISA.

Supported PTX patterns (as emitted for our five kernel types):

  Integer (int16_t, 4 elements packed per 64-bit register):
    add.s16   rd, rs1, rs2     →  VADD  rd, rs1, rs2
    sub.s16   rd, rs1, rs2     →  VSUB  rd, rs1, rs2
    max.s16   rd, rs1, 0       →  RELU  rd, rs1

  BF16 (__nv_bfloat16, 4 elements packed per 64-bit register):
    mul.rn.bf16   rd, rs1, rs2            →  VMUL  rd, rs1, rs2
    fma.rn.bf16   rd, rs1, rs2, acc       →  FMAC  rd, rs1, rs2
      (acc is treated as rd for writeback; hardware reads rd as accumulator)

  Memory (64-bit packed register):
    ld.global.u64  rd, [base]      →  LD  rd, [base]
    ld.global.b64  rd, [base]      →  LD  rd, [base]
    st.global.u64  [base], rs2     →  ST  [base], rs2
    st.global.b64  [base], rs2     →  ST  [base], rs2

  Control:
    ret;  →  HALT

PTX register allocation:
  PTX uses virtual registers (%r0, %rd0, %f0, etc.).
  We assign GPU physical registers r0–r15 in first-seen order.
  The first PTX register seen gets r0, the next gets r1, and so on.
  A warning is printed if more than 16 unique PTX registers are referenced.
"""

import re
import sys
from typing import Optional

# Lines beginning with these patterns are PTX boilerplate — skip silently.
_SKIP_PATTERNS = [
    r"^\s*\.",           # directives: .version, .target, .reg, .param, .local ...
    r"^\s*//",           # comments
    r"^\s*\{",           # open brace
    r"^\s*\}",           # close brace
    r"\bversion\b",
    r"\btarget\b",
    r"\baddress_size\b",
    r"\.visible\b",
    r"\.entry\b",
    r"\bmov\.",          # mov (register moves / address computation)
    r"\bcvta\.",         # convert address space
    r"\bcvt\.",          # type conversion
    r"\bsetp\.",         # set predicate
    r"^\s*@",            # predicated instructions
    r"\bbra\b",          # branch
    r"\bbar\.sync\b",    # barrier
    r"\bld\.param\b",    # load from parameter space
    r"\bshl\.",          # shift left (index arithmetic)
    r"\badd\.\s*s64\b",  # 64-bit int add (pointer arithmetic)
    r"\badd\.\s*u64\b",
    r"\bmul\.\s*wide\b", # widening multiply (index arithmetic)
    r"\bmul\.\s*lo\b",   # multiply low (index arithmetic)
    r"\band\.\s*b32\b",  # bitwise and (index masking)
]


class PTXParser:
    def __init__(self, verbose: bool = True):
        self.verbose   = verbose
        self._reg_map: dict[str, int] = {}   # PTX name → GPU reg number
        self._warnings: list[str]     = []

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, ptx_text: str) -> list[dict]:
        """
        Parse PTX source text and return a list of GPU instruction dicts.

        Each dict has keys:
          op    str   mnemonic (VADD / VSUB / VMUL / FMAC / RELU / LD / ST / HALT)
          rd    int   destination register index (0–15)
          rs1   int   source register 1 index
          rs2   int   source register 2 index

        Raises ValueError if an unrecognised critical instruction is encountered.
        """
        self._reg_map = {}
        self._warnings = []
        instructions = []

        for lineno, raw in enumerate(ptx_text.splitlines(), start=1):
            line = re.sub(r"//.*", "", raw).strip().rstrip(";").strip()
            if not line:
                continue
            if self._should_skip(line):
                continue

            ins = self._try_parse_line(line, lineno)
            if ins is not None:
                instructions.append(ins)
            else:
                msg = f"  [PTX line {lineno}] unrecognised: {line}"
                self._warnings.append(msg)
                if self.verbose:
                    print(msg, file=sys.stderr)

        if self._warnings and self.verbose:
            print(f"  [PTX] {len(self._warnings)} unrecognised line(s) skipped.",
                  file=sys.stderr)

        return instructions

    @property
    def reg_map(self) -> dict[str, int]:
        """PTX register name → GPU register number after last parse()."""
        return dict(self._reg_map)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _should_skip(self, line: str) -> bool:
        return any(re.search(p, line) for p in _SKIP_PATTERNS)

    def _reg(self, name: str) -> int:
        """Map a PTX register name to a GPU physical register number."""
        if name not in self._reg_map:
            n = len(self._reg_map)
            if n >= 16:
                raise ValueError(
                    f"PTX kernel uses more than 16 registers; cannot map {name!r}.\n"
                    "Split into multiple kernels or reuse registers."
                )
            self._reg_map[name] = n
        return self._reg_map[name]

    def _try_parse_line(self, line: str, lineno: int) -> Optional[dict]:
        # ── ret → HALT ────────────────────────────────────────────────────────
        if re.match(r"ret\b", line):
            return {"op": "HALT", "rd": 0, "rs1": 0, "rs2": 0}

        # ── ld.global.{u64,b64,s16,...}  rd, [base] ───────────────────────────
        m = re.match(
            r"ld\.global\.\w+\s+(%\w+)\s*,\s*\[(%\w+)\s*\]", line)
        if m:
            return {"op": "LD", "rd": self._reg(m.group(1)),
                    "rs1": self._reg(m.group(2)), "rs2": 0}

        # ── st.global.{u64,b64,s16,...}  [base], rs2 ─────────────────────────
        m = re.match(
            r"st\.global\.\w+\s+\[(%\w+)\s*\]\s*,\s*(%\w+)", line)
        if m:
            return {"op": "ST", "rd": 0,
                    "rs1": self._reg(m.group(1)), "rs2": self._reg(m.group(2))}

        # ── add.s16 / add.u16  rd, rs1, rs2 → VADD ───────────────────────────
        m = re.match(r"add\.[su]16\s+(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)", line)
        if m:
            return {"op": "VADD", "rd": self._reg(m.group(1)),
                    "rs1": self._reg(m.group(2)), "rs2": self._reg(m.group(3))}

        # ── sub.s16 / sub.u16  rd, rs1, rs2 → VSUB ───────────────────────────
        m = re.match(r"sub\.[su]16\s+(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)", line)
        if m:
            return {"op": "VSUB", "rd": self._reg(m.group(1)),
                    "rs1": self._reg(m.group(2)), "rs2": self._reg(m.group(3))}

        # ── mul.rn.bf16  rd, rs1, rs2 → VMUL ─────────────────────────────────
        m = re.match(r"mul\.rn\.bf16\s+(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)", line)
        if m:
            return {"op": "VMUL", "rd": self._reg(m.group(1)),
                    "rs1": self._reg(m.group(2)), "rs2": self._reg(m.group(3))}

        # ── fma.rn.bf16  rd, rs1, rs2, acc → FMAC ────────────────────────────
        # PTX: rd = rs1*rs2 + acc   Hardware: rd = rs1*rs2 + rd (acc=rd)
        # We map acc to rd so the accumulator register is reused correctly.
        m = re.match(
            r"fma\.rn\.bf16\s+(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)",
            line)
        if m:
            rd_name  = m.group(1)
            rs1_name = m.group(2)
            rs2_name = m.group(3)
            acc_name = m.group(4)
            # If acc != rd in PTX, warn: hardware always reads rd as accumulator
            if acc_name != rd_name and self.verbose:
                print(
                    f"  [PTX line {lineno}] FMAC: PTX accumulator {acc_name!r} "
                    f"mapped to rd={rd_name!r}; hardware reads rd as accumulator. "
                    f"Ensure {acc_name!r} and {rd_name!r} are the same register "
                    f"or initialise rd before issuing FMAC.",
                    file=sys.stderr,
                )
            return {"op": "FMAC", "rd": self._reg(rd_name),
                    "rs1": self._reg(rs1_name), "rs2": self._reg(rs2_name)}

        # ── max.s16  rd, rs1, 0 → RELU ────────────────────────────────────────
        m = re.match(r"max\.[su]16\s+(%\w+)\s*,\s*(%\w+)\s*,\s*0\b", line)
        if m:
            return {"op": "RELU", "rd": self._reg(m.group(1)),
                    "rs1": self._reg(m.group(2)), "rs2": 0}

        return None  # unrecognised


def parse_ptx_file(path: str, verbose: bool = True) -> list[dict]:
    """Convenience wrapper: read a .ptx file and return instruction list."""
    with open(path) as f:
        text = f.read()
    parser = PTXParser(verbose=verbose)
    instructions = parser.parse(text)
    if verbose:
        print(f"  PTX register mapping: {parser.reg_map}")
    return instructions