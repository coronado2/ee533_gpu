#!/usr/bin/env python3
"""
compile_gpu.py  —  kernel.cu → gpu_program.mem

Usage:
    python compile_gpu.py kernel.cu
    python compile_gpu.py kernel.cu --kernel vadd_int16
    python compile_gpu.py kernel.cu --list        # show all kernels in the file

Requires nvcc on PATH (CUDA toolkit). Runs:
    nvcc -ptx -arch=sm_80 kernel.cu -o kernel.ptx

Then translates the PTX to your GPU's 32-bit opcodes and writes:
    gpu_program.mem   ← $readmemh("gpu_program.mem", imem) in gpu_core.v
    gpu_program.lst   ← annotated listing for debugging
"""

import re, sys, subprocess, argparse
from pathlib import Path

# ─── Opcode encoding — must match control_unit.v exactly ─────────────────────
#
#  Bit [31:28] opcode   Bit [27:24] dtype   Bit [23:20] rd
#  Bit [19:16] rs1      Bit [15:12] rs2     Bit [11:0]  zero
#
#  opcode 0x0  VADD  dtype=0  rd = rs1 + rs2          (int16, exec_int16x4)
#  opcode 0x1  VSUB  dtype=0  rd = rs1 - rs2          (int16, exec_int16x4)
#  opcode 0x2  VMUL  dtype=1  rd = rs1 * rs2          (bf16,  tensor unit)
#  opcode 0x3  FMAC  dtype=1  rd = rs1 * rs2 + rd     (bf16,  tensor unit, rd=acc)
#  opcode 0x4  RELU  dtype=0  rd = max(0, rs1)        (int16, exec_int16x4)
#  opcode 0x5  LD    dtype=0  rd = mem[rs1]           (64-bit load)
#  opcode 0x6  ST    dtype=0  mem[rs1] = rs2          (64-bit store)
#  opcode 0xF  HALT  dtype=0  stop

OPCODES = {
    #         opcode  dtype
    "VADD": (0x0,    0x0),
    "VSUB": (0x1,    0x0),
    "VMUL": (0x2,    0x1),
    "FMAC": (0x3,    0x1),
    "RELU": (0x4,    0x0),
    "LD":   (0x5,    0x0),
    "ST":   (0x6,    0x0),
    "HALT": (0xF,    0x0),
}

def encode(mnemonic, rd=0, rs1=0, rs2=0):
    op, dtype = OPCODES[mnemonic]
    return (op << 28) | (dtype << 24) | (rd << 20) | (rs1 << 16) | (rs2 << 12)

def disasm(word):
    op  = (word >> 28) & 0xF
    rd  = (word >> 20) & 0xF
    rs1 = (word >> 16) & 0xF
    rs2 = (word >> 12) & 0xF
    mn  = {0:"VADD",1:"VSUB",2:"VMUL",3:"FMAC",4:"RELU",5:"LD",6:"ST",0xF:"HALT"}.get(op,f"?{op:X}")
    if op == 0xF: return "HALT"
    if op == 0x5: return f"LD    r{rd}, [r{rs1}]"
    if op == 0x6: return f"ST    [r{rs1}], r{rs2}"
    if op == 0x4: return f"RELU  r{rd}, r{rs1}"
    if op == 0x3: return f"FMAC  r{rd}, r{rs1}, r{rs2}  ; acc=r{rd}"
    return f"{mn}  r{rd}, r{rs1}, r{rs2}"

# ─── Step 1: run nvcc ─────────────────────────────────────────────────────────

def run_nvcc(cu_path, ptx_path):
    cmd = ["nvcc", "-ptx", "-arch=sm_80", str(cu_path), "-o", str(ptx_path)]
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"nvcc failed:\n{r.stderr}")
    print(f"  → {ptx_path}")

# ─── Step 2: find kernel bodies in PTX ───────────────────────────────────────

def extract_kernels(ptx):
    """Return {name: body_text} for every .entry in the PTX."""
    kernels = {}
    for m in re.finditer(r'\.(?:visible\s+)?entry\s+(\w+)\s*\(', ptx):
        name  = m.group(1)
        start = ptx.index('{', m.start())
        depth = 0
        for i in range(start, len(ptx)):
            if ptx[i] == '{': depth += 1
            elif ptx[i] == '}':
                depth -= 1
                if depth == 0:
                    kernels[name] = ptx[start+1:i]
                    break
    return kernels

# ─── Step 3: translate PTX body → instruction list ────────────────────────────

# PTX lines we skip (pointer arithmetic, type conversions, boilerplate)
_SKIP = [
    r'^\s*[.{]',      # directives and braces
    r'^\s*}',
    r'\bcvta?\b',     # address conversion
    r'\bmov\b',       # register moves
    r'\bmul\.wide\b', r'\bmul\.lo\b',   # index multiply
    r'\badd\.[su]64\b', r'\badd\.u64\b',  # pointer add
    r'\bshl\b', r'\band\.b\b',
    r'\bsetp\b', r'^\s*@',
    r'\bbra\b', r'\bbar\.sync\b',
    r'\bld\.param\b',
    r'^\s*//',
]

def _skip(line):
    return any(re.search(p, line) for p in _SKIP)

class Regs:
    """Map PTX virtual registers to GPU r0–r15 in first-seen order."""
    def __init__(self):
        self._m = {}
    def __call__(self, name):
        if name not in self._m:
            n = len(self._m)
            if n >= 16:
                sys.exit(f"Kernel uses >16 registers (can't map {name!r}). Split the kernel.")
            self._m[name] = n
        return self._m[name]
    def mapping(self):
        return dict(self._m)

def translate(body, name):
    reg = Regs()
    insns = []
    R = r'(%\w+)'

    for line in body.splitlines():
        line = re.sub(r'//.*', '', line).strip().rstrip(';').strip()
        if not line or _skip(line):
            continue

        m = re.match(r'ret\b', line)
        if m:
            insns.append({"op":"HALT","rd":0,"rs1":0,"rs2":0}); continue

        m = re.match(rf'ld\.global\.\w+\s+{R}\s*,\s*\[{R}\]', line)
        if m:
            insns.append({"op":"LD","rd":reg(m[1]),"rs1":reg(m[2]),"rs2":0}); continue

        m = re.match(rf'st\.global\.\w+\s+\[{R}\]\s*,\s*{R}', line)
        if m:
            insns.append({"op":"ST","rd":0,"rs1":reg(m[1]),"rs2":reg(m[2])}); continue

        m = re.match(rf'add\.[su]16\s+{R}\s*,\s*{R}\s*,\s*{R}', line)
        if m:
            insns.append({"op":"VADD","rd":reg(m[1]),"rs1":reg(m[2]),"rs2":reg(m[3])}); continue

        m = re.match(rf'sub\.[su]16\s+{R}\s*,\s*{R}\s*,\s*{R}', line)
        if m:
            insns.append({"op":"VSUB","rd":reg(m[1]),"rs1":reg(m[2]),"rs2":reg(m[3])}); continue

        m = re.match(rf'mul\.rn\.bf16\s+{R}\s*,\s*{R}\s*,\s*{R}', line)
        if m:
            insns.append({"op":"VMUL","rd":reg(m[1]),"rs1":reg(m[2]),"rs2":reg(m[3])}); continue

        m = re.match(rf'fma\.rn\.bf16\s+{R}\s*,\s*{R}\s*,\s*{R}\s*,\s*{R}', line)
        if m:
            rd_n = m[1]
            if m[4] != rd_n:
                print(f"  [WARN] FMAC: acc={m[4]} != rd={rd_n}; hardware reads rd as acc")
            insns.append({"op":"FMAC","rd":reg(rd_n),"rs1":reg(m[2]),"rs2":reg(m[3])}); continue

        m = re.match(rf'max\.[su]16\s+{R}\s*,\s*{R}\s*,\s*0\b', line)
        if m:
            insns.append({"op":"RELU","rd":reg(m[1]),"rs1":reg(m[2]),"rs2":0}); continue

        # unrecognised — skip silently (it's likely index arithmetic nvcc emits)

    print(f"  registers: {reg.mapping()}")
    return insns

# ─── Step 4: assemble → words → files ────────────────────────────────────────

def assemble(insns):
    return [encode(i["op"], i.get("rd",0), i.get("rs1",0), i.get("rs2",0)) for i in insns]

def write_mem(words, path, depth=256):
    halt = encode("HALT")
    with open(path, "w") as f:
        for w in words:       f.write(f"{w:08X}\n")
        for _ in range(len(words), depth): f.write(f"{halt:08X}\n")

def write_lst(words, path, kernel):
    with open(path, "w") as f:
        f.write(f"// kernel: {kernel}\n// addr  word      disassembly\n")
        for i, w in enumerate(words):
            f.write(f"[{i:03d}]  {w:08X}  {disasm(w)}\n")

# ─── Write outputs ────────────────────────────────────────────────────────────

def write_lst_multi(all_words, path, sections):
    """Annotated listing for a multi-kernel program.
    sections = [(kernel_name, word_count), ...]
    """
    with open(path, "w") as f:
        f.write("// addr  word      disassembly\n")
        idx = 0
        for name, count in sections:
            f.write(f"\n// ── {name} ──\n")
            for w in all_words[idx:idx+count]:
                f.write(f"[{idx:03d}]  {w:08X}  {disasm(w)}\n")
                idx += count
                idx -= count - 1  # step one at a time
            idx = sum(c for _, c in sections[:sections.index((name,count))+1])

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="kernel.cu → gpu_program.mem  (your custom GPU opcodes)")
    ap.add_argument("cu", help=".cu source file")
    ap.add_argument("--kernel", help="specific kernel name (default: all kernels concatenated)")
    ap.add_argument("--list",   action="store_true", help="list kernels and exit")
    ap.add_argument("-o",       default="gpu_program", help="output base name")
    args = ap.parse_args()

    cu  = Path(args.cu)
    ptx = cu.with_suffix(".ptx")

    # 1. Compile
    print(f"\n[1] nvcc: {cu.name} → {ptx.name}")
    run_nvcc(cu, ptx)

    # 2. Find kernels
    text    = ptx.read_text()
    kernels = extract_kernels(text)
    print(f"\n[2] Kernels found: {', '.join(kernels)}")

    if args.list:
        for k in kernels: print(f"  {k}")
        return

    # 3. Pick one kernel or use all
    if args.kernel:
        if args.kernel not in kernels:
            sys.exit(f"Kernel '{args.kernel}' not found. Available: {list(kernels)}")
        to_compile = [args.kernel]
    else:
        to_compile = list(kernels)

    # 4. Translate each kernel → words, concatenate
    print(f"\n[3] Translating {len(to_compile)} kernel(s): {', '.join(to_compile)}")
    all_words = []
    sections  = []   # (name, word_count) for the listing
    addr      = 0

    for name in to_compile:
        print(f"\n  [{name}]  @ address {addr}")
        insns = translate(kernels[name], name)
        words = assemble(insns)
        print(f"  {len(words)} instructions  (addr {addr}–{addr+len(words)-1})")
        all_words.extend(words)
        sections.append((name, len(words)))
        addr += len(words)

    if addr > 256:
        sys.exit(f"Program too large: {addr} words exceeds imem depth of 256")

    # 5. Write outputs
    mem_path = args.o + ".mem"
    lst_path = args.o + ".lst"

    halt = encode("HALT")
    with open(mem_path, "w") as f:
        for w in all_words:
            f.write(f"{w:08X}\n")
        for _ in range(len(all_words), 256):
            f.write(f"{halt:08X}\n")

    with open(lst_path, "w") as f:
        f.write(f"// {cu.name} → {mem_path}\n")
        f.write("// addr  word      disassembly\n")
        idx = 0
        for name, count in sections:
            f.write(f"\n// ── {name} ──\n")
            for w in all_words[idx:idx+count]:
                f.write(f"[{idx:03d}]  {w:08X}  {disasm(w)}\n")
                idx += 1

    print(f"\n[4] Output  ({len(all_words)} total instructions)")
    print(f"  {mem_path}  ← $readmemh(\"{mem_path}\", imem) in gpu_core.v")
    print(f"  {lst_path}  ← listing\n")

    idx = 0
    for name, count in sections:
        print(f"  // {name}")
        for w in all_words[idx:idx+count]:
            print(f"  [{idx:03d}]  {w:08X}  {disasm(w)}")
            idx += 1
        print()

if __name__ == "__main__":
    main()