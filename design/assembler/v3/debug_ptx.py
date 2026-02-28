#!/usr/bin/env python3
"""
Drop this in the same folder as kernel.cu and run:
    python test_real_ptx.py kernel.ptx

It will print the RAW PTX body of every kernel so you can see
exactly what nvcc generated and what the translator is matching/missing.
"""
import sys, re
from pathlib import Path

def extract_kernels(ptx):
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

if len(sys.argv) < 2:
    print("Usage: python test_real_ptx.py kernel.ptx")
    sys.exit(1)

text    = Path(sys.argv[1]).read_text()
kernels = extract_kernels(text)

for name, body in kernels.items():
    print(f"\n{'='*60}")
    print(f"KERNEL: {name}")
    print('='*60)
    for i, line in enumerate(body.splitlines()):
        stripped = line.strip()
        if stripped:
            print(f"  {i:3d}: {stripped}")

    # Show what the s16->u64 map looks like after pass 1
    s16_to_u64 = {}
    R = r'(%\w+)'
    for raw in body.splitlines():
        line = re.sub(r'//.*', '', raw).strip().rstrip(';').strip()
        m = re.match(r'mov\.b64\s+\{([^}]+)\}\s*,\s*'+R, line)
        if m:
            lanes = re.findall(r'%\w+', m.group(1))
            for l in lanes: s16_to_u64[l] = m.group(2)
        m = re.match(r'mov\.b64\s+'+R+r'\s*,\s*\{([^}]+)\}', line)
        if m:
            lanes = re.findall(r'%\w+', m.group(2))
            for l in lanes: s16_to_u64[l] = m.group(1)

    print(f"\n  --- s16->u64 map (Pass 1) ---")
    for k,v in s16_to_u64.items():
        print(f"    {k} -> {v}")

    # Show which compute lines match/don't match
    print(f"\n  --- Pattern matching (Pass 2) ---")
    patterns = {
        "add.[su]16":        r'add\.[su]16\s+(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)',
        "sub.[su]16":        r'sub\.[su]16\s+(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)',
        "max.[su]16":        r'max\.[su]16\s+(%\w+)\s*,\s*(%\w+)\s*,\s*0',
        "mul.rn.bf16(x2)?":  r'mul\.rn\.bf16x?2?\s+(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)',
        "fma.rn.bf16(x2)?":  r'fma\.rn\.bf16x?2?\s+(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)',
        "ld.global":         r'ld\.global\.\w+\s+(%\w+)\s*,\s*\[(%\w+)\]',
        "st.global":         r'st\.global\.\w+\s+\[(%\w+)\]\s*,\s*(%\w+)',
    }
    for raw in body.splitlines():
        line = re.sub(r'//.*', '', raw).strip().rstrip(';').strip()
        if not line: continue
        for label, pat in patterns.items():
            if re.match(pat, line):
                m = re.match(pat, line)
                regs = [s16_to_u64.get(g, g) for g in m.groups()]
                print(f"    MATCH [{label}]: {line}")
                print(f"           -> resolved regs: {regs}")
                break
        else:
            # didn't match any pattern - show lines that look like compute ops
            if any(x in line for x in ['mul.', 'fma.', 'add.', 'sub.', 'max.']) \
               and 'param' not in line and 'wide' not in line \
               and 'lo' not in line and 'u64' not in line:
                print(f"    NO MATCH (compute-looking): {line}")