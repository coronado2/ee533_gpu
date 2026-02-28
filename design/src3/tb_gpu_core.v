// tb_gpu_system.v  ?  Comprehensive testbench (Verilog-2001 compatible)
// Tests:
//   A. tensor_bf16_4lane  unit-level (VADD, VSUB, VMUL, FMAC, RELU + edge cases)
//   B. bf16_add  targeted edge cases via tensor VADD/VSUB
//   C. bf16_mul  targeted edge cases via tensor VMUL
//   D. gpu_core  system-level (VADD, VSUB, RELU, VMUL, FMAC kernels)
//   E. HALT / stall behaviour
//
// Verilog-2001 rules obeyed:
//   - No variable declarations inside named begin/end blocks
//   - No wire declarations inside initial/always blocks
//   - No part-select on a function return value  (use a temp reg)
//   - task inputs declared with explicit type
// ?????????????????????????????????????????????????????????????????????????????
`timescale 1ns/1ps

module tb_gpu_system;

// ??? Clock / reset ????????????????????????????????????????????????????????
reg clk, rst_n;
always #5 clk = ~clk;   // 100 MHz

task do_reset;
    begin
        rst_n = 0;
        repeat(4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
    end
endtask

// ??? Pass / fail bookkeeping ??????????????????????????????????????????????
integer pass_count, fail_count;

// check: display PASS or FAIL.
// name_str argument is unused at runtime but documents the test in source.
task check;
    input [127:0] name_str;
    input         cond;
    begin
        if (cond) begin
            pass_count = pass_count + 1;
            $display("  PASS");
        end else begin
            fail_count = fail_count + 1;
            $display("  FAIL ***");
        end
    end
endtask

// ??? BF16 constants ???????????????????????????????????????????????????????
localparam [15:0] BF_0  = 16'h0000;  //  0.0
localparam [15:0] BF_1  = 16'h3F80;  //  1.0
localparam [15:0] BF_2  = 16'h4000;  //  2.0
localparam [15:0] BF_3  = 16'h4040;  //  3.0
localparam [15:0] BF_4  = 16'h4080;  //  4.0
localparam [15:0] BF_5  = 16'h40A0;  //  5.0
localparam [15:0] BF_6  = 16'h40C0;  //  6.0
localparam [15:0] BF_7  = 16'h40E0;  //  7.0
localparam [15:0] BF_8  = 16'h4100;  //  8.0
localparam [15:0] BF_9  = 16'h4110;  //  9.0
localparam [15:0] BF_N1 = 16'hBF80;  // -1.0
localparam [15:0] BF_N2 = 16'hC000;  // -2.0

// ??? Utility functions ????????????????????????????????????????????????????

// Pack 4 bf16 values into one 64-bit register (lane0 = bits[15:0])
function [63:0] pack4;
    input [15:0] l0, l1, l2, l3;
    pack4 = {l3, l2, l1, l0};
endfunction

// Extract one bf16 lane (n = 0..3) from a 64-bit register.
// Returns 16-bit value. Caller must store in a reg before part-selecting.
function [15:0] lane;
    input [63:0] v;
    input integer n;
    case (n)
        0: lane = v[15:0];
        1: lane = v[31:16];
        2: lane = v[47:32];
        3: lane = v[63:48];
        default: lane = 16'hxxxx;
    endcase
endfunction

// BF16 approximate equality: pass if values match or differ by at most 1 ULP.
function bf16_eq;
    input [15:0] a, b;
    reg [14:0] ua, ub;
    begin
        ua = a[14:0];
        ub = b[14:0];
        bf16_eq = (a == b) ||
                  ((a[15] == b[15]) && (ua == ub + 15'd1)) ||
                  ((a[15] == b[15]) && (ub == ua + 15'd1));
    end
endfunction

// ??? Tensor DUT ???????????????????????????????????????????????????????????
reg        t_start;
reg  [2:0] t_op;
reg [63:0] t_a, t_b, t_acc;
wire [63:0] t_rd;
wire        t_done;

tensor_bf16_4lane dut_tensor (
    .clk    (clk),
    .rst_n  (rst_n),
    .start  (t_start),
    .op     (t_op),
    .a_reg  (t_a),
    .b_reg  (t_b),
    .acc_reg(t_acc),
    .rd_reg (t_rd),
    .done   (t_done)
);

// ??? Tensor helper tasks ??????????????????????????????????????????????????

// Wait for t_done with a cycle timeout.
task tensor_wait;
    input integer timeout_cycles;
    integer cnt;
    begin
        cnt = 0;
        while (!t_done && cnt < timeout_cycles) begin
            @(posedge clk);
            cnt = cnt + 1;
        end
        if (cnt >= timeout_cycles)
            $display("  TIMEOUT: tensor done never asserted!");
    end
endtask

// Issue a one-cycle start pulse to the tensor unit.
task tensor_fire;
    input [2:0]  op;
    input [63:0] a, b, acc;
    begin
        @(negedge clk);
        t_start = 1'b1;
        t_op    = op;
        t_a     = a;
        t_b     = b;
        t_acc   = acc;
        @(posedge clk);
        @(negedge clk);
        t_start = 1'b0;
    end
endtask

// ??? GPU core DUT ?????????????????????????????????????????????????????????
gpu_core dut_gpu (.clk(clk), .rst_n(rst_n));

// Memory byte addresses (data_mem_bram indexes as mem[addr[9:3]])
localparam [63:0] ADDR_INT_A  = 64'h080;  // int16 A  = [1,2,3,4]
localparam [63:0] ADDR_INT_B  = 64'h0A0;  // int16 B  = [10,20,30,40]
localparam [63:0] ADDR_BF_A   = 64'h100;  // bf16  A  = [1.0,2.0,3.0,4.0]
localparam [63:0] ADDR_BF_B   = 64'h120;  // bf16  B  = [2.0,2.0,2.0,2.0]
localparam [63:0] ADDR_BF_ACC = 64'h140;  // bf16 acc = [1.0,1.0,1.0,1.0]
localparam [63:0] ADDR_NEG    = 64'h180;  // int16 neg= [-1,-2,-3,-4]
localparam [63:0] ADDR_DST_I  = 64'h200;  // store dst int
localparam [63:0] ADDR_DST_B  = 64'h240;  // store dst bf16 vmul
localparam [63:0] ADDR_DST_F  = 64'h280;  // store dst bf16 fmac

// Pre-load integer kernel address registers via hierarchical path.
task gpu_load_regs_int;
    input [63:0] addr_a, addr_b, addr_dst;
    begin
        dut_gpu.regfile[1] = addr_a;
        dut_gpu.regfile[3] = addr_b;
        dut_gpu.regfile[5] = addr_dst;
    end
endtask

// Pre-load bf16 kernel address registers.
task gpu_load_regs_bf16;
    input [63:0] addr_a, addr_b, addr_acc, addr_dst;
    begin
        dut_gpu.regfile[1] = addr_a;
        dut_gpu.regfile[3] = addr_b;
        dut_gpu.regfile[5] = addr_acc;
        dut_gpu.regfile[6] = addr_dst;
    end
endtask

// Wait for GPU to assert halted, with a cycle timeout.
task gpu_wait_halt;
    input integer timeout_cycles;
    integer cnt;
    begin
        cnt = 0;
        while (!dut_gpu.halted && cnt < timeout_cycles) begin
            @(posedge clk);
            cnt = cnt + 1;
        end
        if (cnt >= timeout_cycles)
            $display("  GPU TIMEOUT: halted never asserted!");
    end
endtask

// Jump the GPU PC to a specific word address.
task gpu_set_pc;
    input [31:0] word_addr;
    begin
        force dut_gpu.u_pc.pc_out = word_addr;
        @(posedge clk);
        release dut_gpu.u_pc.pc_out;
    end
endtask

// ??? Module-level scratch registers ??????????????????????????????????????
// All temporaries must be declared here (not inside begin/end in Verilog-2001)
reg [63:0] scratch_result;
reg [31:0] pc_snap1, pc_snap2;
reg [15:0] tmp_lane;   // holds function return before part-select

// ??? Main test sequence ???????????????????????????????????????????????????
initial begin
    // Initialise driven signals
    clk        = 1'b0;
    rst_n      = 1'b0;
    pass_count = 0;
    fail_count = 0;
    t_start    = 1'b0;
    t_op       = 3'd0;
    t_a        = 64'd0;
    t_b        = 64'd0;
    t_acc      = 64'd0;

    $display("=================================================================");
    $display("  GPU BF16 Tensor + Core Testbench  (Verilog-2001 compatible)");
    $display("=================================================================");

    do_reset;

    // =====================================================================
    // PART A ? tensor_bf16_4lane direct unit tests
    // =====================================================================
    $display("\n--- Part A: Tensor Unit Direct Tests ---");

    // ?? A1: RELU ?????????????????????????????????????????????????????????
    $display("\n[A1] RELU: max(0, [-1.0, 2.0, -2.0, 4.0]) = [0, 2, 0, 4]");
    tensor_fire(3'b100,
        pack4(BF_N1, BF_2, BF_N2, BF_4),
        64'd0, 64'd0);
    tensor_wait(10);

    tmp_lane = lane(t_rd, 0);
    $write("  lane0 expect 0x0000  got 0x%04h ", tmp_lane);
    check("A1_relu_lane0", tmp_lane == BF_0);

    tmp_lane = lane(t_rd, 1);
    $write("  lane1 expect 0x4000  got 0x%04h ", tmp_lane);
    check("A1_relu_lane1", tmp_lane == BF_2);

    tmp_lane = lane(t_rd, 2);
    $write("  lane2 expect 0x0000  got 0x%04h ", tmp_lane);
    check("A1_relu_lane2", tmp_lane == BF_0);

    tmp_lane = lane(t_rd, 3);
    $write("  lane3 expect 0x4080  got 0x%04h ", tmp_lane);
    check("A1_relu_lane3", tmp_lane == BF_4);

    @(posedge clk); @(posedge clk);

    // ?? A2: VADD ?????????????????????????????????????????????????????????
    $display("\n[A2] VADD: [1,2,3,4] + [1,1,1,1] = [2,3,4,5]");
    tensor_fire(3'b000,
        pack4(BF_1, BF_2, BF_3, BF_4),
        pack4(BF_1, BF_1, BF_1, BF_1),
        64'd0);
    tensor_wait(10);

    tmp_lane = lane(t_rd, 0);
    $write("  lane0 expect 0x%04h  got 0x%04h ", BF_2, tmp_lane);
    check("A2_vadd_lane0", bf16_eq(tmp_lane, BF_2));

    tmp_lane = lane(t_rd, 1);
    $write("  lane1 expect 0x%04h  got 0x%04h ", BF_3, tmp_lane);
    check("A2_vadd_lane1", bf16_eq(tmp_lane, BF_3));

    tmp_lane = lane(t_rd, 2);
    $write("  lane2 expect 0x%04h  got 0x%04h ", BF_4, tmp_lane);
    check("A2_vadd_lane2", bf16_eq(tmp_lane, BF_4));

    tmp_lane = lane(t_rd, 3);
    $write("  lane3 expect 0x%04h  got 0x%04h ", BF_5, tmp_lane);
    check("A2_vadd_lane3", bf16_eq(tmp_lane, BF_5));

    @(posedge clk); @(posedge clk);

    // ?? A3: VSUB ?????????????????????????????????????????????????????????
    $display("\n[A3] VSUB: [4,3,2,1] - [1,1,1,1] = [3,2,1,0]");
    tensor_fire(3'b001,
        pack4(BF_4, BF_3, BF_2, BF_1),
        pack4(BF_1, BF_1, BF_1, BF_1),
        64'd0);
    tensor_wait(10);

    tmp_lane = lane(t_rd, 0);
    $write("  lane0 expect 0x%04h  got 0x%04h ", BF_3, tmp_lane);
    check("A3_vsub_lane0", bf16_eq(tmp_lane, BF_3));

    tmp_lane = lane(t_rd, 1);
    $write("  lane1 expect 0x%04h  got 0x%04h ", BF_2, tmp_lane);
    check("A3_vsub_lane1", bf16_eq(tmp_lane, BF_2));

    tmp_lane = lane(t_rd, 2);
    $write("  lane2 expect 0x%04h  got 0x%04h ", BF_1, tmp_lane);
    check("A3_vsub_lane2", bf16_eq(tmp_lane, BF_1));

    tmp_lane = lane(t_rd, 3);
    $write("  lane3 expect 0x0000  got 0x%04h ", tmp_lane);
    check("A3_vsub_lane3", bf16_eq(tmp_lane, BF_0));

    @(posedge clk); @(posedge clk);

    // ?? A4: VMUL ?????????????????????????????????????????????????????????
    $display("\n[A4] VMUL: [1,2,3,4] * [2,2,2,2] = [2,4,6,8]");
    tensor_fire(3'b010,
        pack4(BF_1, BF_2, BF_3, BF_4),
        pack4(BF_2, BF_2, BF_2, BF_2),
        64'd0);
    tensor_wait(10);

    tmp_lane = lane(t_rd, 0);
    $write("  lane0 expect 0x%04h  got 0x%04h ", BF_2, tmp_lane);
    check("A4_vmul_lane0", bf16_eq(tmp_lane, BF_2));

    tmp_lane = lane(t_rd, 1);
    $write("  lane1 expect 0x%04h  got 0x%04h ", BF_4, tmp_lane);
    check("A4_vmul_lane1", bf16_eq(tmp_lane, BF_4));

    tmp_lane = lane(t_rd, 2);
    $write("  lane2 expect 0x%04h  got 0x%04h ", BF_6, tmp_lane);
    check("A4_vmul_lane2", bf16_eq(tmp_lane, BF_6));

    tmp_lane = lane(t_rd, 3);
    $write("  lane3 expect 0x%04h  got 0x%04h ", BF_8, tmp_lane);
    check("A4_vmul_lane3", bf16_eq(tmp_lane, BF_8));

    @(posedge clk); @(posedge clk);

    // ?? A5: FMAC ?????????????????????????????????????????????????????????
    $display("\n[A5] FMAC: [1,2,3,4]*[2,2,2,2] + [1,1,1,1] = [3,5,7,9]");
    tensor_fire(3'b011,
        pack4(BF_1, BF_2, BF_3, BF_4),
        pack4(BF_2, BF_2, BF_2, BF_2),
        pack4(BF_1, BF_1, BF_1, BF_1));
    tensor_wait(20);

    tmp_lane = lane(t_rd, 0);
    $write("  lane0 expect 0x%04h  got 0x%04h ", BF_3, tmp_lane);
    check("A5_fmac_lane0", bf16_eq(tmp_lane, BF_3));

    tmp_lane = lane(t_rd, 1);
    $write("  lane1 expect 0x%04h  got 0x%04h ", BF_5, tmp_lane);
    check("A5_fmac_lane1", bf16_eq(tmp_lane, BF_5));

    tmp_lane = lane(t_rd, 2);
    $write("  lane2 expect 0x%04h  got 0x%04h ", BF_7, tmp_lane);
    check("A5_fmac_lane2", bf16_eq(tmp_lane, BF_7));

    tmp_lane = lane(t_rd, 3);
    $write("  lane3 expect 0x%04h  got 0x%04h ", BF_9, tmp_lane);
    check("A5_fmac_lane3", bf16_eq(tmp_lane, BF_9));

    @(posedge clk); @(posedge clk);

    // ?? A6: VMUL by zero ?????????????????????????????????????????????????
    $display("\n[A6] VMUL: [1,2,3,4] * 0 = [0,0,0,0]");
    tensor_fire(3'b010,
        pack4(BF_1, BF_2, BF_3, BF_4),
        64'd0, 64'd0);
    tensor_wait(10);
    $write("  rd = 0x%016h  expect 0 ", t_rd);
    check("A6_vmul_zero", t_rd == 64'd0);

    @(posedge clk); @(posedge clk);

    // ?? A7: VADD additive identity ????????????????????????????????????????
    $display("\n[A7] VADD: [3,3,3,3] + 0 = [3,3,3,3]");
    tensor_fire(3'b000,
        pack4(BF_3, BF_3, BF_3, BF_3),
        64'd0, 64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd, 0);
    $write("  lane0 expect 0x%04h  got 0x%04h ", BF_3, tmp_lane);
    check("A7_vadd_identity", bf16_eq(tmp_lane, BF_3));

    @(posedge clk); @(posedge clk);

    // ?? A8: RELU all-negative ?????????????????????????????????????????????
    $display("\n[A8] RELU: all-negative -> all-zero");
    tensor_fire(3'b100,
        pack4(BF_N1, BF_N2, BF_N1, BF_N2),
        64'd0, 64'd0);
    tensor_wait(10);
    $write("  rd = 0x%016h  expect 0 ", t_rd);
    check("A8_relu_all_neg", t_rd == 64'd0);

    @(posedge clk); @(posedge clk);

    // ?? A9: RELU all-positive passthrough ????????????????????????????????
    $display("\n[A9] RELU: all-positive passthrough [1,2,3,4]");
    tensor_fire(3'b100,
        pack4(BF_1, BF_2, BF_3, BF_4),
        64'd0, 64'd0);
    tensor_wait(10);
    $write("  expect 0x%016h  got 0x%016h ",
           pack4(BF_1,BF_2,BF_3,BF_4), t_rd);
    check("A9_relu_pos_pass", t_rd == pack4(BF_1,BF_2,BF_3,BF_4));

    @(posedge clk); @(posedge clk);

    // =====================================================================
    // PART B ? bf16_add targeted edge cases (via tensor VADD/VSUB)
    // =====================================================================
    $display("\n--- Part B: bf16_add Edge Cases ---");

    // ?? B1: +inf + 1 = +inf ???????????????????????????????????????????????
    $display("\n[B1] VADD: +inf + 1.0 = +inf");
    tensor_fire(3'b000,
        pack4(16'h7F80, BF_0, BF_0, BF_0),
        pack4(BF_1,     BF_0, BF_0, BF_0),
        64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd, 0);
    $write("  lane0 expect 0x7F80  got 0x%04h ", tmp_lane);
    check("B1_inf_plus_1", tmp_lane == 16'h7F80);

    @(posedge clk); @(posedge clk);

    // ?? B2: NaN propagation ???????????????????????????????????????????????
    $display("\n[B2] VADD: qNaN + 1.0 = NaN  (exponent must be 0xFF)");
    tensor_fire(3'b000,
        pack4(16'hFF81, BF_0, BF_0, BF_0),
        pack4(BF_1,     BF_0, BF_0, BF_0),
        64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd, 0);
    $write("  lane0 = 0x%04h  expect exp field = 0xFF ", tmp_lane);
    check("B2_nan_prop", tmp_lane[14:7] == 8'hFF);

    @(posedge clk); @(posedge clk);

    // ?? B3: 1.0 - 1.0 = ±0 ???????????????????????????????????????????????
    $display("\n[B3] VSUB: 1.0 - 1.0 = 0  (0x0000 or 0x8000 both acceptable)");
    tensor_fire(3'b001,
        pack4(BF_1, BF_0, BF_0, BF_0),
        pack4(BF_1, BF_0, BF_0, BF_0),
        64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd, 0);
    $write("  lane0 = 0x%04h  ", tmp_lane);
    check("B3_sub_cancel",
          (tmp_lane == 16'h0000) || (tmp_lane == 16'h8000));

    @(posedge clk); @(posedge clk);

    // ?? B4: Large exponent difference ????????????????????????????????????
    $display("\n[B4] VADD: 1.0 + very_small -> 1.0  (subnormal absorbed)");
    // 2^-6 = 0x3B80 (well below 1.0 ULP in bf16); adding to 1.0 should give 1.0
    tensor_fire(3'b000,
        pack4(BF_1,     BF_0, BF_0, BF_0),
        pack4(16'h3B80, BF_0, BF_0, BF_0),
        64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd, 0);
    $write("  lane0 = 0x%04h  expect ~0x%04h ", tmp_lane, BF_1);
    check("B4_exp_diff_absorb", bf16_eq(tmp_lane, BF_1));

    @(posedge clk); @(posedge clk);

    // =====================================================================
    // PART C ? bf16_mul targeted edge cases (via tensor VMUL)
    // =====================================================================
    $display("\n--- Part C: bf16_mul Edge Cases ---");

    // ?? C1: 1.0 * 1.0 = 1.0 ?????????????????????????????????????????????
    $display("\n[C1] VMUL: 1.0 * 1.0 = 1.0");
    tensor_fire(3'b010,
        pack4(BF_1, BF_1, BF_1, BF_1),
        pack4(BF_1, BF_1, BF_1, BF_1),
        64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd, 0);
    $write("  lane0 expect 0x%04h  got 0x%04h ", BF_1, tmp_lane);
    check("C1_mul_1x1", bf16_eq(tmp_lane, BF_1));

    @(posedge clk); @(posedge clk);

    // ?? C2: +inf * 0 = NaN ???????????????????????????????????????????????
    $display("\n[C2] VMUL: +inf * 0 = NaN");
    tensor_fire(3'b010,
        pack4(16'h7F80, BF_0, BF_0, BF_0),
        pack4(BF_0,     BF_0, BF_0, BF_0),
        64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd, 0);
    $write("  lane0 = 0x%04h  expect exp=FF mant!=0 ", tmp_lane);
    check("C2_inf_x_0",
          (tmp_lane[14:7] == 8'hFF) && (tmp_lane[6:0] != 7'd0));

    @(posedge clk); @(posedge clk);

    // ?? C3: (-1) * (-1) = +1 ?????????????????????????????????????????????
    $display("\n[C3] VMUL: -1.0 * -1.0 = +1.0");
    tensor_fire(3'b010,
        pack4(BF_N1, BF_0, BF_0, BF_0),
        pack4(BF_N1, BF_0, BF_0, BF_0),
        64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd, 0);
    $write("  lane0 expect 0x%04h  got 0x%04h ", BF_1, tmp_lane);
    check("C3_neg_x_neg", bf16_eq(tmp_lane, BF_1));

    @(posedge clk); @(posedge clk);

    // ?? C4: 0 * 0 = 0 ????????????????????????????????????????????????????
    $display("\n[C4] VMUL: 0 * 0 = 0");
    tensor_fire(3'b010, 64'd0, 64'd0, 64'd0);
    tensor_wait(10);
    $write("  rd = 0x%016h  expect 0 ", t_rd);
    check("C4_zero_x_zero", t_rd == 64'd0);

    @(posedge clk); @(posedge clk);

    // =====================================================================
    // PART D ? GPU core system-level tests
    // =====================================================================
    $display("\n--- Part D: GPU Core System Tests ---");

    // ?? D1: VADD int kernel ???????????????????????????????????????????????
    $display("\n[D1] GPU VADD: [1,2,3,4]+[10,20,30,40] = [11,22,33,44]");
    do_reset;
    gpu_load_regs_int(ADDR_INT_A, ADDR_INT_B, ADDR_DST_I);
    gpu_wait_halt(60);

    scratch_result = dut_gpu.u_datamem.mem[ADDR_DST_I[9:3]];
    $display("  stored = 0x%016h", scratch_result);

    $write("  lane0 expect 11  got %0d ", $signed(scratch_result[15:0]));
    check("D1_vadd_l0", scratch_result[15:0] == 16'd11);

    $write("  lane1 expect 22  got %0d ", $signed(scratch_result[31:16]));
    check("D1_vadd_l1", scratch_result[31:16] == 16'd22);

    $write("  lane2 expect 33  got %0d ", $signed(scratch_result[47:32]));
    check("D1_vadd_l2", scratch_result[47:32] == 16'd33);

    $write("  lane3 expect 44  got %0d ", $signed(scratch_result[63:48]));
    check("D1_vadd_l3", scratch_result[63:48] == 16'd44);

    // ?? D2: VSUB int kernel ???????????????????????????????????????????????
    $display("\n[D2] GPU VSUB: [1,2,3,4]-[10,20,30,40] = [-9,-18,-27,-36]");
    do_reset;
    gpu_set_pc(32'd5);
    gpu_load_regs_int(ADDR_INT_A, ADDR_INT_B, ADDR_DST_I);
    gpu_wait_halt(60);

    scratch_result = dut_gpu.u_datamem.mem[ADDR_DST_I[9:3]];
    $display("  stored = 0x%016h", scratch_result);

    $write("  lane0 expect -9  got %0d ", $signed(scratch_result[15:0]));
    check("D2_vsub_l0", $signed(scratch_result[15:0]) == -16'sd9);

    $write("  lane3 expect -36 got %0d ", $signed(scratch_result[63:48]));
    check("D2_vsub_l3", $signed(scratch_result[63:48]) == -16'sd36);

    // ?? D3: RELU int kernel ???????????????????????????????????????????????
    $display("\n[D3] GPU RELU: [-1,-2,-3,-4] -> [0,0,0,0]");
    do_reset;
    gpu_set_pc(32'd10);
    dut_gpu.regfile[1] = ADDR_NEG;
    dut_gpu.regfile[3] = ADDR_DST_I;
    gpu_wait_halt(60);

    scratch_result = dut_gpu.u_datamem.mem[ADDR_DST_I[9:3]];
    $display("  stored = 0x%016h  (expect 0)", scratch_result);
    check("D3_relu_neg", scratch_result == 64'd0);

    // ?? D4: VMUL bf16 kernel ?????????????????????????????????????????????
    $display("\n[D4] GPU VMUL bf16: [1,2,3,4]*[2,2,2,2] = [2,4,6,8]");
    do_reset;
    gpu_set_pc(32'd14);
    dut_gpu.regfile[1] = ADDR_BF_A;
    dut_gpu.regfile[3] = ADDR_BF_B;
    dut_gpu.regfile[5] = ADDR_DST_B;
    gpu_wait_halt(120);

    scratch_result = dut_gpu.u_datamem.mem[ADDR_DST_B[9:3]];
    $display("  stored = 0x%016h", scratch_result);

    $write("  lane0 expect 0x%04h  got 0x%04h ", BF_2, scratch_result[15:0]);
    check("D4_vmul_l0", bf16_eq(scratch_result[15:0], BF_2));

    $write("  lane1 expect 0x%04h  got 0x%04h ", BF_4, scratch_result[31:16]);
    check("D4_vmul_l1", bf16_eq(scratch_result[31:16], BF_4));

    $write("  lane2 expect 0x%04h  got 0x%04h ", BF_6, scratch_result[47:32]);
    check("D4_vmul_l2", bf16_eq(scratch_result[47:32], BF_6));

    $write("  lane3 expect 0x%04h  got 0x%04h ", BF_8, scratch_result[63:48]);
    check("D4_vmul_l3", bf16_eq(scratch_result[63:48], BF_8));

    // ?? D5: FMAC bf16 kernel ?????????????????????????????????????????????
    $display("\n[D5] GPU FMAC bf16: [1,2,3,4]*[2,2,2,2]+[1,1,1,1] = [3,5,7,9]");
    do_reset;
    gpu_set_pc(32'd19);
    gpu_load_regs_bf16(ADDR_BF_A, ADDR_BF_B, ADDR_BF_ACC, ADDR_DST_F);
    gpu_wait_halt(180);

    scratch_result = dut_gpu.u_datamem.mem[ADDR_DST_F[9:3]];
    $display("  stored = 0x%016h", scratch_result);

    $write("  lane0 expect 3.0 (0x%04h) got 0x%04h ", BF_3, scratch_result[15:0]);
    check("D5_fmac_l0", bf16_eq(scratch_result[15:0], BF_3));

    $write("  lane1 expect 5.0 (0x%04h) got 0x%04h ", BF_5, scratch_result[31:16]);
    check("D5_fmac_l1", bf16_eq(scratch_result[31:16], BF_5));

    $write("  lane2 expect 7.0 (0x%04h) got 0x%04h ", BF_7, scratch_result[47:32]);
    check("D5_fmac_l2", bf16_eq(scratch_result[47:32], BF_7));

    $write("  lane3 expect 9.0 (0x%04h) got 0x%04h ", BF_9, scratch_result[63:48]);
    check("D5_fmac_l3", bf16_eq(scratch_result[63:48], BF_9));

    // =====================================================================
    // PART E ? HALT / stall behaviour
    // =====================================================================
    $display("\n--- Part E: HALT and Stall Behaviour ---");

    // ?? E1: PC freezes after HALT ?????????????????????????????????????????
    $display("\n[E1] PC must not advance after HALT");
    do_reset;
    gpu_load_regs_int(ADDR_INT_A, ADDR_INT_B, ADDR_DST_I);
    gpu_wait_halt(60);
    @(posedge clk); pc_snap1 = dut_gpu.u_pc.pc_out;
    @(posedge clk); pc_snap2 = dut_gpu.u_pc.pc_out;
    $write("  pc_snap1=%0d  pc_snap2=%0d  ", pc_snap1, pc_snap2);
    check("E1_pc_freeze", pc_snap1 == pc_snap2);

    // ?? E2: halted flag is permanently latched ????????????????????????????
    $display("\n[E2] halted flag must remain asserted");
    $write("  halted = %b  ", dut_gpu.halted);
    check("E2_halt_latch", dut_gpu.halted === 1'b1);

    // ?? E3: stall is asserted when halted ?????????????????????????????????
    $display("\n[E3] stall must be asserted when halted");
    $write("  stall = %b  ", dut_gpu.stall);
    check("E3_stall_asserted", dut_gpu.stall === 1'b1);

    // =====================================================================
    // Summary
    // =====================================================================
    $display("\n=================================================================");
    $display("  Test Summary:  PASS = %0d   FAIL = %0d   TOTAL = %0d",
             pass_count, fail_count, pass_count + fail_count);
    if (fail_count == 0)
        $display("  *** ALL TESTS PASSED ***");
    else
        $display("  *** %0d TEST(S) FAILED - see markers above ***", fail_count);
    $display("=================================================================\n");

    #100;
    $finish;
end

// Watchdog
initial begin
    #100_000;
    $display("WATCHDOG: exceeded 100 us - aborting");
    $finish;
end

// VCD waveform dump
initial begin
    $dumpfile("tb_gpu_system.vcd");
    $dumpvars(0, tb_gpu_system);
end

endmodule