#!/usr/bin/env python3
"""
compile_gpu.py  -  kernel.cu -> gpu_program.mem

Usage:
    python compile_gpu.py kernel.cu
    python compile_gpu.py kernel.cu --kernel vadd_int16
    python compile_gpu.py kernel.cu --list

Requires nvcc on PATH:
    nvcc -ptx -arch=sm_80 kernel.cu -o kernel.ptx

─────────────────────────────────────────────────────────────────────────────
What nvcc actually emits for this kernel.cu (sm_80, CUDA 12.2):

  • Scalar PTX: one element per thread via threadIdx.x
  • u16 loads/stores (ld.global.u16, st.global.u16), NOT packed b64
  • Address chain: ld.param -> cvta.to.global -> add.s64 -> ld/st address
  • bf16 multiply uses an inline asm block with braces on the SAME line:
      {.reg .b16 c;
       mov.b16 c, 0x8000U;
       fma.rn.bf16 %rs1,%rs2,%rs3,c;}     <- closing brace attached!
  • bf16 fmac inline asm:
      {fma.rn.bf16 %rs1,%rs2,%rs3,%rs4;   <- opening brace attached!
      }

The translator handles all of this by:
  1. Stripping `{}; \\t` from BOTH ends of every line before matching,
     so the fma instruction is always clean regardless of attached braces.
  2. Tracking the full param->cvta->add.s64->ld/st address chain so each
     physical register maps back to its kernel argument slot.
  3. Assigning physical registers by CUDA argument position:
       2 params (a, out):          a->r1,  out->r3
       3 params (a, b, out):       a->r1,  b->r3,  out->r5
       4 params (a, b, acc, out):  a->r1,  b->r3,  acc->r5, out->r7
     This matches what the testbench pre-loads into regfile[1/3/5/7].
"""

import re, sys, subprocess, argparse
from pathlib import Path

# ── ISA ──────────────────────────────────────────────────────────────────────
OPCODES = {
    "VADD":(0x0,0x0), "VSUB":(0x1,0x0), "VMUL":(0x2,0x1),
    "FMAC":(0x3,0x1), "RELU":(0x4,0x0), "LD":(0x5,0x0),
    "ST":(0x6,0x0),   "HALT":(0xF,0x0),
}
def encode(mn, rd=0, rs1=0, rs2=0):
    op, dtype = OPCODES[mn]
    return (op<<28)|(dtype<<24)|(rd<<20)|(rs1<<16)|(rs2<<12)

def disasm(w):
    op=(w>>28)&0xF; rd=(w>>20)&0xF; rs1=(w>>16)&0xF; rs2=(w>>12)&0xF
    mn = {0:"VADD",1:"VSUB",2:"VMUL",3:"FMAC",4:"RELU",
          5:"LD",6:"ST",0xF:"HALT"}.get(op, f"?{op:X}")
    if op==0xF: return "HALT"
    if op==0x5: return f"LD    r{rd}, [r{rs1}]"
    if op==0x6: return f"ST    [r{rs1}], r{rs2}"
    if op==0x4: return f"RELU  r{rd}, r{rs1}"
    if op==0x3: return f"FMAC  r{rd}, r{rs1}, r{rs2}  ; acc=r{rd}"
    return f"{mn}  r{rd}, r{rs1}, r{rs2}"

# ── nvcc ─────────────────────────────────────────────────────────────────────
def run_nvcc(cu, ptx):
    cmd = ["nvcc", "-ptx", "-arch=sm_80", str(cu), "-o", str(ptx)]
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"nvcc failed:\n{r.stderr}")
    print(f"  -> {ptx}")

# ── kernel body extraction ────────────────────────────────────────────────────
def extract_kernels(ptx_text):
    kernels = {}
    for m in re.finditer(r'\.(?:visible\s+)?entry\s+(\w+)\s*\(', ptx_text):
        name  = m.group(1)
        start = ptx_text.index('{', m.start())
        depth = 0
        for i in range(start, len(ptx_text)):
            if ptx_text[i] == '{': depth += 1
            elif ptx_text[i] == '}':
                depth -= 1
                if depth == 0:
                    kernels[name] = ptx_text[start+1:i]
                    break
    return kernels

# ── param name extraction ─────────────────────────────────────────────────────
def get_param_names(ptx_text, kernel_name):
    m = re.search(rf'\.entry\s+{re.escape(kernel_name)}\s*\(([^)]*)\)', ptx_text)
    if not m:
        return []
    return re.findall(r'\.param\s+\.\w+\s+(\w+)', m.group(1))

# ── full address-chain alias map ──────────────────────────────────────────────
def build_alias(body):
    """
    Propagate: ld.param -> cvta -> add.s64
    Returns dict: any_ptx_reg -> root_param_name
    """
    alias = {}
    for raw in body.splitlines():
        line = re.sub(r'//.*', '', raw).strip().strip('{}; \t')
        m = re.match(r'ld\.param\.\w+\s+(%\w+)\s*,\s*\[(\w+)\]', line)
        if m:
            alias[m[1]] = m[2]

    def resolve(r, visited=None):
        if visited is None: visited = set()
        while r in alias and r not in visited:
            visited.add(r); r = alias[r]
        return r

    changed = True
    while changed:
        changed = False
        for raw in body.splitlines():
            line = re.sub(r'//.*', '', raw).strip().strip('{}; \t')
            # cvta.to.global dst, src
            m = re.match(r'cvta\S*\s+(%\w+)\s*,\s*(%\w+)', line)
            if m and m[2] in alias:
                new = resolve(m[2])
                if alias.get(m[1]) != new:
                    alias[m[1]] = new; changed = True
                continue
            # add.s64 dst, src, offset  (only propagate if src is param-derived)
            m = re.match(r'add\.[su]64\s+(%\w+)\s*,\s*(%\w+)\s*,', line)
            if m and m[2] in alias:
                new = resolve(m[2])
                if alias.get(m[1]) != new:
                    alias[m[1]] = new; changed = True

    return {k: resolve(k) for k in alias}

# ── translate one kernel ──────────────────────────────────────────────────────
def translate(body, kernel_name, ptx_text):
    params = get_param_names(ptx_text, kernel_name)
    alias  = build_alias(body)
    n      = len(params)

    # Physical register assignment by CUDA argument order:
    # CUDA passes arguments left-to-right as param_0, param_1, ...
    # The last parameter in each kernel signature is always the output pointer.
    #   vadd(a, b, out)       -> param_0=a, param_1=b, param_2=out
    #   vsub(a, b, out)       -> param_0=a, param_1=b, param_2=out
    #   relu(a, out)          -> param_0=a, param_1=out
    #   vmul(a, b, out)       -> param_0=a, param_1=b, param_2=out
    #   fmac(a, b, acc, out)  -> param_0=a, param_1=b, param_2=acc, param_3=out
    #
    # Testbench preloads:  regfile[1]=ADDR_A, regfile[3]=ADDR_B,
    #                      regfile[5]=ADDR_ACC/DST, regfile[7]=ADDR_DST(fmac)
    if n == 2:
        # (a, out)
        param_phys = {params[0]: 1,   # a   -> r1
                      params[1]: 3}   # out -> r3
    elif n == 3:
        # (a, b, out)
        param_phys = {params[0]: 1,   # a   -> r1
                      params[1]: 3,   # b   -> r3
                      params[2]: 5}   # out -> r5
    elif n == 4:
        # (a, b, acc, out)
        param_phys = {params[0]: 1,   # a   -> r1
                      params[1]: 3,   # b   -> r3
                      params[2]: 5,   # acc -> r5
                      params[3]: 7}   # out -> r7
    else:
        param_phys = {}

    used = set(param_phys.values())
    nxt  = [max(used) + 1 if used else 0]
    rmap = {}

    def phys(ptx_reg):
        root = alias.get(ptx_reg, ptx_reg)
        if root in param_phys:
            p = param_phys[root]
            rmap[ptx_reg] = p
            return p
        if ptx_reg not in rmap:
            while nxt[0] in used: nxt[0] += 1
            rmap[ptx_reg] = nxt[0]
            used.add(nxt[0])
            nxt[0] += 1
        return rmap[ptx_reg]

    insns   = []
    emitted = set()

    def emit(op, rd, rs1, rs2=None):
        key = (op, rd, rs1, rs2)
        if key in emitted: return
        emitted.add(key)
        insns.append({"op":op, "rd":rd, "rs1":rs1,
                      "rs2": rs2 if rs2 is not None else 0})

    R = r'(%\w+)'

    SKIP = re.compile(
        r'^\.(reg|loc|file|section|visible|entry|param|maxntid|reqntid)\b'
        r'|^ld\.param\b'
        r'|^cvta\b'
        r'|^mov\.u32\b'
        r'|^mul\.wide\b'
        r'|^add\.[su]64\b'
        r'|^mov\.b64\b'
        r'|^mov\.b16\b'
        r'|^mov\.\w+\b'
        r'|^@'
        r'|^bar\.sync\b'
        r'|^setp\b'
        r'|^bra\b'
    )

    for raw in body.splitlines():
        # ── strip braces and semicolons from both ends ────────────────────
        # This is the key fix for nvcc's inline asm blocks, which produce:
        #   {.reg .b16 c;\n  fma.rn.bf16 %rs1,%rs2,%rs3,c;}
        # After strip('{}; \t') each sub-line becomes a clean instruction.
        line = re.sub(r'//.*', '', raw).strip().strip('{}; \t')
        if not line or SKIP.search(line):
            continue

        # HALT
        if re.match(r'ret\b', line):
            insns.append({"op":"HALT","rd":0,"rs1":0,"rs2":0})
            continue

        # LD
        m = re.match(rf'ld\.global\.\w+\s+{R}\s*,\s*\[{R}\]', line)
        if m:
            insns.append({"op":"LD", "rd":phys(m[1]), "rs1":phys(m[2]), "rs2":0})
            continue

        # ST
        m = re.match(rf'st\.global\.\w+\s+\[{R}\]\s*,\s*{R}', line)
        if m:
            insns.append({"op":"ST", "rd":0, "rs1":phys(m[1]), "rs2":phys(m[2])})
            continue

        # VADD (add.s16 / add.u16)
        m = re.match(rf'add\.[su]\d+\s+{R}\s*,\s*{R}\s*,\s*{R}', line)
        if m:
            emit("VADD", phys(m[1]), phys(m[2]), phys(m[3]))
            continue

        # VSUB
        m = re.match(rf'sub\.[su]\d+\s+{R}\s*,\s*{R}\s*,\s*{R}', line)
        if m:
            emit("VSUB", phys(m[1]), phys(m[2]), phys(m[3]))
            continue

        # RELU (max with literal 0)
        m = re.match(rf'max\.[su]\d+\s+{R}\s*,\s*{R}\s*,\s*0\b', line)
        if m:
            emit("RELU", phys(m[1]), phys(m[2]))
            continue

        # VMUL (mul.rn.bf16 / mul.rn.bf16x2)
        m = re.match(rf'mul\.[a-z0-9.]*bf16[a-z0-9]*\s+{R}\s*,\s*{R}\s*,\s*{R}', line)
        if m:
            emit("VMUL", phys(m[1]), phys(m[2]), phys(m[3]))
            continue

        # VMUL or FMAC from fma instruction
        # fma.rn.bf16  rd, rs1, rs2, acc_or_literal
        # If 4th operand is a % register -> FMAC
        # If 4th operand is a literal (like 'c' = -0.0) -> VMUL
        m = re.match(
            rf'fma\.[a-z0-9.]*bf16[a-z0-9]*\s+{R}\s*,\s*{R}\s*,\s*{R}\s*,\s*(\S+)',
            line)
        if m:
            acc = m.group(4).strip(';}')
            if acc.startswith('%'):
                emit("FMAC", phys(m[1]), phys(m[2]), phys(m[3]))
            else:
                # literal accumulator (nvcc's __hmul emits fma with c=-0)
                emit("VMUL", phys(m[1]), phys(m[2]), phys(m[3]))
            continue

        # Warn on any compute-looking line that didn't match
        first = line.split()[0] if line.split() else ''
        if any(x in first for x in ('mul.','fma.','add.','sub.','max.')):
            print(f"  WARNING: unmatched compute line: {line!r}")

    print(f"  params: { {p: param_phys[p] for p in params if p in param_phys} }")
    return insns

# ── assemble ─────────────────────────────────────────────────────────────────
def assemble(insns):
    return [encode(i["op"], i.get("rd",0), i.get("rs1",0), i.get("rs2",0))
            for i in insns]

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="kernel.cu -> gpu_program.mem")
    ap.add_argument("cu")
    ap.add_argument("--kernel")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-o", default="gpu_program")
    args = ap.parse_args()

    cu  = Path(args.cu)
    ptx = cu.with_suffix(".ptx")

    print(f"\n[1] nvcc: {cu.name} -> {ptx.name}")
    run_nvcc(cu, ptx)

    text    = ptx.read_text()
    kernels = extract_kernels(text)
    print(f"\n[2] Kernels found: {', '.join(kernels)}")

    if args.list:
        for k in kernels: print(f"  {k}")
        return

    to_compile = [args.kernel] if args.kernel else list(kernels)
    if args.kernel and args.kernel not in kernels:
        sys.exit(f"Kernel {args.kernel!r} not found. Available: {list(kernels)}")

    print(f"\n[3] Translating {len(to_compile)} kernel(s):")
    all_words, sections, addr = [], [], 0

    for name in to_compile:
        print(f"\n  [{name}]  @ addr {addr}")
        insns = translate(kernels[name], name, text)
        words = assemble(insns)
        print(f"  {len(words)} instructions  (addr {addr}-{addr+len(words)-1})")
        all_words.extend(words)
        sections.append((name, len(words)))
        addr += len(words)

    if addr > 256:
        sys.exit(f"Program too large: {addr} words > 256")

    mem_path  = args.o + ".mem"
    lst_path  = args.o + ".lst"
    halt_word = encode("HALT")

    with open(mem_path, "w") as f:
        for w in all_words:          f.write(f"{w:08X}\n")
        for _ in range(len(all_words), 256): f.write(f"{halt_word:08X}\n")

    with open(lst_path, "w") as f:
        f.write(f"// {cu.name} -> {mem_path}\n// addr  word      disassembly\n")
        idx = 0
        for name, count in sections:
            f.write(f"\n// -- {name} --\n")
            for w in all_words[idx:idx+count]:
                f.write(f"[{idx:03d}]  {w:08X}  {disasm(w)}\n")
                idx += 1

    print(f"\n[4] Output ({len(all_words)} instructions)")
    print(f"  {mem_path}\n  {lst_path}\n")
    idx = 0
    for name, count in sections:
        print(f"  // {name}")
        for w in all_words[idx:idx+count]:
            print(f"  [{idx:03d}]  {w:08X}  {disasm(w)}")
            idx += 1
        print()

if __name__ == "__main__":
    main()