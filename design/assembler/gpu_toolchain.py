#!/usr/bin/env python3
"""
gpu_toolchain.py  –  Main entry point for the GPU compiler toolchain.

Full pipeline:
    kernel.cu  ──nvcc──►  kernel.ptx  ──ptx_parser──►  instructions
    ──assembler──►  gpu_program.mem   (load into hardware via $readmemh)
                    gpu_program.hex.txt  (human-readable listing)

The .mem file is fed directly to gpu_core.v:
    $readmemh("gpu_program.mem", imem);

Usage examples:
    # Compile PTX from nvcc output:
    python gpu_toolchain.py kernel.ptx

    # Compile and immediately simulate:
    python gpu_toolchain.py kernel.ptx --sim

    # Assemble hand-written .asm:
    python gpu_toolchain.py kernel.asm

    # Run built-in tests (all 5 kernel types):
    python gpu_toolchain.py --test

    # Interactive demo trace for all 5 kernels:
    python gpu_toolchain.py --demo
"""

import argparse, re, sys
from pathlib import Path

from isa        import encode, decode, disasm, OPCODES, DTYPE_ENC, HALT_WORD
from assembler  import assemble, write_outputs, write_mem, write_listing
from ptx_parser import PTXParser
from simulator  import (GPUSim, pack_i16, pack_bf16, unpack_i16, unpack_bf16,
                        bf16_lanes_to_floats, _float_to_bf16, _bf16_to_float)


# ─────────────────────────────────────────────────────────────────────────────
# Hand-written GPU assembly parser
# Syntax (case-insensitive, no type suffix needed):
#   VADD  rd, rs1, rs2
#   VSUB  rd, rs1, rs2
#   VMUL  rd, rs1, rs2
#   FMAC  rd, rs1, rs2    ; rd is also the accumulator
#   RELU  rd, rs1
#   LD    rd, [rs1]
#   ST    [rs1], rs2
#   HALT
# ─────────────────────────────────────────────────────────────────────────────
def parse_asm(text):
    instructions = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = re.sub(r'[#;].*', '', raw).strip()
        if not line: continue

        if re.match(r'HALT\b', line, re.I):
            instructions.append({"op":"HALT","rd":0,"rs1":0,"rs2":0}); continue

        m = re.match(r'LD\s+r(\d+)\s*,\s*\[\s*r(\d+)\s*\]', line, re.I)
        if m:
            instructions.append({"op":"LD","rd":int(m.group(1)),"rs1":int(m.group(2)),"rs2":0}); continue

        m = re.match(r'ST\s+\[\s*r(\d+)\s*\]\s*,\s*r(\d+)', line, re.I)
        if m:
            instructions.append({"op":"ST","rd":0,"rs1":int(m.group(1)),"rs2":int(m.group(2))}); continue

        m = re.match(r'RELU\s+r(\d+)\s*,\s*r(\d+)', line, re.I)
        if m:
            instructions.append({"op":"RELU","rd":int(m.group(1)),"rs1":int(m.group(2)),"rs2":0}); continue

        m = re.match(r'(\w+)\s+r(\d+)\s*,\s*r(\d+)\s*,\s*r(\d+)', line, re.I)
        if m:
            instructions.append({"op":m.group(1).upper(),"rd":int(m.group(2)),
                                  "rs1":int(m.group(3)),"rs2":int(m.group(4))}); continue

        print(f"  [ASM line {lineno}] unrecognised: {line}", file=sys.stderr)
    return instructions


# ─────────────────────────────────────────────────────────────────────────────
# Self-test suite  (all 5 kernel types + encode/PTX/assembler/mem file)
# ─────────────────────────────────────────────────────────────────────────────
def run_tests():
    BF = _float_to_bf16
    ok = fail = 0

    def check(name, got, exp):
        nonlocal ok, fail
        if got == exp:
            print(f"  PASS  {name}"); ok += 1
        else:
            print(f"  FAIL  {name}"); print(f"         got: {got!r}"); print(f"         exp: {exp!r}"); fail += 1

    print("\n[1] Encode / decode round-trip")
    for mn,(oc,dt) in OPCODES.items():
        w = encode(oc, DTYPE_ENC[dt], 5, 3, 7)
        d = decode(w)
        check(f"{mn:5s} 0x{w:08X}", (d["opcode"],d["dtype"],d["rd"],d["rs1"],d["rs2"]),
              (oc,DTYPE_ENC[dt],5,3,7))

    print("\n[2] Disassembly")
    check("VADD",  disasm(encode(0,0,2,0,1)),  "VADD     r2, r0, r1")
    check("VSUB",  disasm(encode(1,0,2,0,1)),  "VSUB     r2, r0, r1")
    check("VMUL",  disasm(encode(2,1,2,0,1)),  "VMUL     r2, r0, r1")
    check("FMAC",  disasm(encode(3,1,3,1,2)),  "FMAC     r3, r1, r2   ; rd = rs1*rs2 + rd")
    check("RELU",  disasm(encode(4,0,1,0,0)),  "RELU     r1, r0")
    check("LD",    disasm(encode(5,0,1,0,0)),  "LD       r1, [r0]")
    check("ST",    disasm(encode(6,0,0,0,3)),  "ST       [r0], r3")
    check("HALT",  disasm(HALT_WORD),           "HALT")

    print("\n[3] PTX parser")
    p = PTXParser(verbose=False)
    for ptx,exp_op in [("add.s16 %r2,%r0,%r1;\nret;","VADD"),
                       ("sub.s16 %r2,%r0,%r1;\nret;","VSUB"),
                       ("mul.rn.bf16 %f2,%f0,%f1;\nret;","VMUL"),
                       ("fma.rn.bf16 %f3,%f1,%f2,%f3;\nret;","FMAC"),
                       ("max.s16 %r1,%r0,0;\nret;","RELU"),
                       ("ld.global.u64 %rd1,[%rd0];\nret;","LD"),
                       ("st.global.u64 [%rd0],%rd2;\nret;","ST")]:
        ins = p.parse(ptx)
        check(f"PTX→{exp_op}", ins[0]["op"], exp_op)

    print("\n[4] Assembler encoding")
    w = assemble([{"op":"VADD","rd":2,"rs1":0,"rs2":1}])[0]
    check("VADD word", w, encode(0,0,2,0,1))
    w = assemble([{"op":"FMAC","rd":3,"rs1":1,"rs2":2}])[0]
    check("FMAC word", w, encode(3,1,3,1,2))
    w = assemble([{"op":"LD","rd":1,"rs1":0,"rs2":0}])[0]
    check("LD word",   w, encode(5,0,1,0,0))
    w = assemble([{"op":"ST","rd":0,"rs1":0,"rs2":3}])[0]
    check("ST word",   w, encode(6,0,0,0,3))

    print("\n[5] .mem file format")
    import tempfile, os
    prog = assemble([{"op":"VADD","rd":2,"rs1":0,"rs2":1},{"op":"HALT","rd":0,"rs1":0,"rs2":0}])
    tf = tempfile.NamedTemporaryFile(mode='w', suffix='.mem', delete=False)
    tf.close()
    write_mem(prog, tf.name)
    lines = open(tf.name).read().splitlines()
    os.unlink(tf.name)
    check("256 lines",          len(lines), 256)
    check("line[0]=VADD word",  lines[0],   f"{prog[0]:08X}")
    check("line[1]=HALT",       lines[1],   f"{HALT_WORD:08X}")
    check("line[255]=HALT pad", lines[255], f"{HALT_WORD:08X}")

    print("\n[6] Simulator: VADD int16  [1,2,3,4]+[10,20,30,40]=[11,22,33,44]")
    prog = assemble([{"op":"VADD","rd":2,"rs1":0,"rs2":1},{"op":"HALT","rd":0,"rs1":0,"rs2":0}])
    sim = GPUSim(prog, verbose=False)
    sim.regfile[0] = pack_i16([1,2,3,4]); sim.regfile[1] = pack_i16([10,20,30,40])
    sim.run()
    check("r2=[11,22,33,44]", unpack_i16(sim.regfile[2]), [11,22,33,44])

    print("\n[7] Simulator: VSUB int16  [10,20,30,40]-[1,2,3,4]=[9,18,27,36]")
    prog = assemble([{"op":"VSUB","rd":2,"rs1":0,"rs2":1},{"op":"HALT","rd":0,"rs1":0,"rs2":0}])
    sim = GPUSim(prog, verbose=False)
    sim.regfile[0] = pack_i16([10,20,30,40]); sim.regfile[1] = pack_i16([1,2,3,4])
    sim.run()
    check("r2=[9,18,27,36]", unpack_i16(sim.regfile[2]), [9,18,27,36])

    print("\n[8] Simulator: RELU int16  max(0,[-5,-1,0,7])=[0,0,0,7]")
    prog = assemble([{"op":"RELU","rd":1,"rs1":0,"rs2":0},{"op":"HALT","rd":0,"rs1":0,"rs2":0}])
    sim = GPUSim(prog, verbose=False)
    sim.regfile[0] = pack_i16([-5,-1,0,7])
    sim.run()
    check("r1=[0,0,0,7]", unpack_i16(sim.regfile[1]), [0,0,0,7])

    print("\n[9] Simulator: VMUL bf16  [2]*[3]=[6]")
    prog = assemble([{"op":"VMUL","rd":2,"rs1":0,"rs2":1},{"op":"HALT","rd":0,"rs1":0,"rs2":0}])
    sim = GPUSim(prog, verbose=False)
    sim.regfile[0] = pack_bf16([BF(2)]*4); sim.regfile[1] = pack_bf16([BF(3)]*4)
    sim.run()
    got = [round(_bf16_to_float(x),1) for x in unpack_bf16(sim.regfile[2])]
    check("r2=[6,6,6,6]", got, [6.0]*4)

    print("\n[10] Simulator: FMAC bf16  2*3+1=7  (r3 is acc)")
    prog = assemble([{"op":"FMAC","rd":3,"rs1":1,"rs2":2},{"op":"HALT","rd":0,"rs1":0,"rs2":0}])
    sim = GPUSim(prog, verbose=False)
    sim.regfile[1] = pack_bf16([BF(2)]*4); sim.regfile[2] = pack_bf16([BF(3)]*4)
    sim.regfile[3] = pack_bf16([BF(1)]*4)
    sim.run()
    got = [round(_bf16_to_float(x),1) for x in unpack_bf16(sim.regfile[3])]
    check("r3=[7,7,7,7]", got, [7.0]*4)

    print("\n[11] Simulator: VMUL stalls PC for 2 cycles")
    prog = assemble([{"op":"VMUL","rd":2,"rs1":0,"rs2":1},{"op":"HALT","rd":0,"rs1":0,"rs2":0}])
    sim = GPUSim(prog, verbose=False)
    sim.regfile[0] = pack_bf16([BF(2)]*4); sim.regfile[1] = pack_bf16([BF(3)]*4)
    sim.run()
    check("3 cycles total (1 issue + 1 stall + 1 done+halt)", sim.cycle, 3)

    print("\n[12] Simulator: FMAC stalls PC for 5 cycles")
    prog = assemble([{"op":"FMAC","rd":3,"rs1":1,"rs2":2},{"op":"HALT","rd":0,"rs1":0,"rs2":0}])
    sim = GPUSim(prog, verbose=False)
    sim.regfile[1] = pack_bf16([BF(2)]*4); sim.regfile[2] = pack_bf16([BF(3)]*4)
    sim.regfile[3] = pack_bf16([BF(1)]*4)
    sim.run()
    check("6 cycles total (1 issue + 4 stall + 1 done+halt)", sim.cycle, 6)

    print("\n[13] Simulator: LD / VADD / ST  (byte addr → word addr)")
    prog = assemble([
        {"op":"LD",  "rd":1, "rs1":0, "rs2":0},
        {"op":"LD",  "rd":2, "rs1":4, "rs2":0},
        {"op":"VADD","rd":3, "rs1":1, "rs2":2},
        {"op":"ST",  "rd":0, "rs1":5, "rs2":3},
        {"op":"HALT","rd":0, "rs1":0, "rs2":0},
    ])
    sim = GPUSim(prog, verbose=False)
    sim.regfile[0] = 0; sim.regfile[4] = 8; sim.regfile[5] = 16
    sim.dmem[0] = pack_i16([1,2,3,4]); sim.dmem[1] = pack_i16([5,6,7,8])
    sim.run()
    check("dmem[2]=[6,8,10,12]", unpack_i16(sim.dmem.get(2,0)), [6,8,10,12])

    print(f"\n{'='*55}")
    print(f"Results: {ok} passed, {fail} failed")
    return fail == 0


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────
def run_demo():
    BF = _float_to_bf16
    print("\n" + "="*65)
    print("  GPU Toolchain Demo  —  five kernel types, cycle trace")
    print("="*65)
    kernels = [
        {"name":"VADD int16  r2=[1,2,3,4]+[10,20,30,40]",
         "prog":[{"op":"VADD","rd":2,"rs1":0,"rs2":1},{"op":"HALT","rd":0,"rs1":0,"rs2":0}],
         "regs":{0:pack_i16([1,2,3,4]),1:pack_i16([10,20,30,40])}},
        {"name":"VSUB int16  r2=[10,20,30,40]-[1,2,3,4]",
         "prog":[{"op":"VSUB","rd":2,"rs1":0,"rs2":1},{"op":"HALT","rd":0,"rs1":0,"rs2":0}],
         "regs":{0:pack_i16([10,20,30,40]),1:pack_i16([1,2,3,4])}},
        {"name":"VMUL bf16   r2=[2]*[3]=[6]",
         "prog":[{"op":"VMUL","rd":2,"rs1":0,"rs2":1},{"op":"HALT","rd":0,"rs1":0,"rs2":0}],
         "regs":{0:pack_bf16([BF(2)]*4),1:pack_bf16([BF(3)]*4)}},
        {"name":"FMAC bf16   r3=r1*r2+r3=2*3+1=7  (r3 is accumulator)",
         "prog":[{"op":"FMAC","rd":3,"rs1":1,"rs2":2},{"op":"HALT","rd":0,"rs1":0,"rs2":0}],
         "regs":{1:pack_bf16([BF(2)]*4),2:pack_bf16([BF(3)]*4),3:pack_bf16([BF(1)]*4)}},
        {"name":"RELU int16  r1=max(0,[-5,-1,0,7])=[0,0,0,7]",
         "prog":[{"op":"RELU","rd":1,"rs1":0,"rs2":0},{"op":"HALT","rd":0,"rs1":0,"rs2":0}],
         "regs":{0:pack_i16([-5,-1,0,7])}},
    ]
    for k in kernels:
        print(f"\n{'─'*65}\n  {k['name']}\n{'─'*65}")
        prog = assemble(k["prog"])
        sim  = GPUSim(prog, verbose=True, max_cycles=20)
        for r,v in k["regs"].items(): sim.regfile[r] = v
        sim.run()

    fmac = assemble([{"op":"FMAC","rd":3,"rs1":1,"rs2":2},
                     {"op":"HALT","rd":0,"rs1":0,"rs2":0}])
    write_outputs(fmac, "gpu_program")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="GPU Compiler Toolchain  (PTX/ASM → gpu_program.mem for gpu_core.v)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?",
                    help=".ptx (from nvcc -ptx -arch=sm_80) or .asm (hand-written)")
    ap.add_argument("-o","--output", default="gpu_program",
                    help="Output base name (default: gpu_program)")
    ap.add_argument("--sim",        action="store_true", help="Run simulator after assembling")
    ap.add_argument("--quiet",      action="store_true", help="Suppress simulator cycle trace")
    ap.add_argument("--max-cycles", type=int, default=5000, dest="max_cycles")
    ap.add_argument("--test",       action="store_true", help="Run self-test suite")
    ap.add_argument("--demo",       action="store_true", help="Run demo for all 5 kernel types")
    args = ap.parse_args()

    if args.test:
        sys.exit(0 if run_tests() else 1)
    if args.demo:
        run_demo(); return
    if not args.input:
        ap.print_help(); return

    path = Path(args.input)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr); sys.exit(1)

    text = path.read_text()
    print(f"\n=== Parsing {path} ===")
    if path.suffix.lower() == ".ptx":
        parser = PTXParser(verbose=not args.quiet)
        instructions = parser.parse(text)
        if not args.quiet: print(f"  Register map: {parser.reg_map}")
    else:
        instructions = parse_asm(text)

    print(f"  {len(instructions)} instruction(s) parsed")
    words = assemble(instructions)
    write_outputs(words, args.output)

    print("\nDisassembly:")
    for i,w in enumerate(words):
        print(f"  [{i:3d}]  {w:08X}   {disasm(w)}")

    if args.sim:
        print("\n=== Simulation ===")
        sim = GPUSim(words, verbose=not args.quiet, max_cycles=args.max_cycles)
        sim.run()
        sim.dump_regs()

if __name__ == "__main__":
    main()