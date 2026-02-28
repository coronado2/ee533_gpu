// tb_gpu_system.v  -  Testbench for datapath.v / gpu_core.v hierarchy
// =====================================================================
// Tests:
//   A. tensor_bf16_4lane  unit-level (VADD, VSUB, VMUL, FMAC, RELU + edges)
//   B. bf16_add  edge cases via tensor VADD/VSUB
//   C. bf16_mul  edge cases via tensor VMUL
//   D. GPU system-level (VADD, VSUB, RELU, VMUL, FMAC kernels)
//   E. HALT / stall behaviour
//
// DUT hierarchy:
//   dut  (datapath)
//   ??? u_imem      (inst_mem)          dut.u_imem.mem[n]
//   ??? u_regfile   (generic_regfile)   dut.u_regfile.regs[n]
//   ??? u_dmem      (data_mem)          dut.u_dmem.mem[n]
//   ??? u_gpu       (gpu_core)
//       ??? u_pc                        dut.u_gpu.u_pc.pc_out
//       ??? u_ctrl
//       ??? u_exec_int
//       ??? u_tensor
//
// Verilog-2001 rules:
//   No variable declarations inside named begin/end blocks.
//   No wire declarations inside initial/always blocks.
//   No part-select on function return (use a temp reg).
// =====================================================================
`timescale 1ns/1ps

module tb_gpu_system;

// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------
reg clk;
always #5 clk = ~clk;   // 100 MHz

// ---------------------------------------------------------------------------
// Datapath host-port drives
// These are driven by the testbench to talk to the datapath's host ports.
// ---------------------------------------------------------------------------
reg        tb_rst_n;
reg        tb_imem_we;
reg [6:0]  tb_imem_addr;
reg [31:0] tb_imem_data;
reg        tb_dmem_we;
reg [7:0]  tb_dmem_wr_addr;
reg [63:0] tb_dmem_wr_data;
reg [7:0]  tb_dmem_rd_addr;

wire [63:0] tb_dmem_rd_data;
wire        tb_halted;
wire [31:0] tb_pc_out;
wire [63:0] tb_result_out;

// ---------------------------------------------------------------------------
// Pass / fail bookkeeping
// ---------------------------------------------------------------------------
integer pass_count, fail_count;

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

// ---------------------------------------------------------------------------
// BF16 constants
// ---------------------------------------------------------------------------
localparam [15:0] BF_0  = 16'h0000;
localparam [15:0] BF_1  = 16'h3F80;
localparam [15:0] BF_2  = 16'h4000;
localparam [15:0] BF_3  = 16'h4040;
localparam [15:0] BF_4  = 16'h4080;
localparam [15:0] BF_5  = 16'h40A0;
localparam [15:0] BF_6  = 16'h40C0;
localparam [15:0] BF_7  = 16'h40E0;
localparam [15:0] BF_8  = 16'h4100;
localparam [15:0] BF_9  = 16'h4110;
localparam [15:0] BF_N1 = 16'hBF80;
localparam [15:0] BF_N2 = 16'hC000;
// FMAC effective result: acc=r9=[2,2,2,2], r8*r9+r9 = [4,6,8,10]
localparam [15:0] BF_10 = 16'h4120;

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------
function [63:0] pack4;
    input [15:0] l0, l1, l2, l3;
    pack4 = {l3, l2, l1, l0};
endfunction

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

// ---------------------------------------------------------------------------
// Tensor unit ? standalone DUT, separate from datapath
// ---------------------------------------------------------------------------
reg        t_start;
reg  [2:0] t_op;
reg [63:0] t_a, t_b, t_acc;
wire [63:0] t_rd;
wire        t_done;

tensor_bf16_4lane dut_tensor (
    .clk     (clk),
    .rst_n   (tb_rst_n),
    .start   (t_start),
    .op      (t_op),
    .a_reg   (t_a),
    .b_reg   (t_b),
    .acc_reg (t_acc),
    .rd_reg  (t_rd),
    .done    (t_done)
);

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

// ---------------------------------------------------------------------------
// Datapath DUT
// ---------------------------------------------------------------------------
datapath dut (
    .clk               (clk),
    .rst_n             (tb_rst_n),
    .imem_host_we      (tb_imem_we),
    .imem_host_addr    (tb_imem_addr),
    .imem_host_data    (tb_imem_data),
    .dmem_host_we      (tb_dmem_we),
    .dmem_host_wr_addr (tb_dmem_wr_addr),
    .dmem_host_wr_data (tb_dmem_wr_data),
    .dmem_host_rd_addr (tb_dmem_rd_addr),
    .dmem_host_rd_data (tb_dmem_rd_data),
    .halted            (tb_halted),
    .pc_out            (tb_pc_out),
    .result_out        (tb_result_out)
);

// ---------------------------------------------------------------------------
// Data memory addresses  (byte addr >> 3 = word index)
// These match the pre-loaded vectors in data_mem.v initial block.
// ---------------------------------------------------------------------------
localparam [63:0] ADDR_INT_A  = 64'h080;  // word 16  [1,2,3,4]
localparam [63:0] ADDR_INT_B  = 64'h0A0;  // word 20  [10,20,30,40]
localparam [63:0] ADDR_BF_A   = 64'h100;  // word 32  [1.0,2.0,3.0,4.0]
localparam [63:0] ADDR_BF_B   = 64'h120;  // word 36  [2.0,2.0,2.0,2.0]
localparam [63:0] ADDR_BF_ACC = 64'h140;  // word 40  [1.0,1.0,1.0,1.0]
localparam [63:0] ADDR_NEG    = 64'h180;  // word 48  [-1,-2,-3,-4]
localparam [63:0] ADDR_DST_I  = 64'h200;  // word 64  int result
localparam [63:0] ADDR_DST_B  = 64'h240;  // word 72  vmul result
localparam [63:0] ADDR_DST_F  = 64'h280;  // word 80  fmac result

// ---------------------------------------------------------------------------
// Instruction load via host write port
// Writes one word at a time through imem_host_we while rst_n=0.
// ---------------------------------------------------------------------------
task imem_write;
    input [6:0]  addr;
    input [31:0] data;
    begin
        @(negedge clk);
        tb_imem_we   = 1'b1;
        tb_imem_addr = addr;
        tb_imem_data = data;
        @(posedge clk);
        @(negedge clk);
        tb_imem_we   = 1'b0;
        tb_imem_addr = 7'd0;
        tb_imem_data = 32'd0;
    end
endtask

// ---------------------------------------------------------------------------
// Reset + program load task.
// Holds rst_n low, loads the full program image, then releases rst_n.
// GPU starts executing from PC=0 when rst_n goes high.
// Call gpu_set_pc immediately after do_reset_and_load to jump to a kernel.
// ---------------------------------------------------------------------------
task do_reset_and_load;
    integer k;
    begin
        // Assert reset, clear host-port signals
        tb_rst_n       = 1'b0;
        tb_imem_we     = 1'b0;
        tb_imem_addr   = 7'd0;
        tb_imem_data   = 32'd0;
        tb_dmem_we     = 1'b0;
        tb_dmem_wr_addr= 8'd0;
        tb_dmem_wr_data= 64'd0;
        tb_dmem_rd_addr= 8'd0;
        repeat(2) @(posedge clk);

        // Clear inst_mem to HALT, then write each kernel
        for (k = 0; k < 128; k = k + 1)
            dut.u_imem.mem[k] = 32'hF0000000;

        // Kernel 0: VADD  (word addr 0-4)
        dut.u_imem.mem[0]  = 32'h50610000; // LD  r6, [r1]
        dut.u_imem.mem[1]  = 32'h50730000; // LD  r7, [r3]
        dut.u_imem.mem[2]  = 32'h00876000; // VADD r8 = r7+r6
        dut.u_imem.mem[3]  = 32'h60058000; // ST  [r5], r8
        dut.u_imem.mem[4]  = 32'hF0000000; // HALT

        // Kernel 1: VSUB  (word addr 5-9)
        dut.u_imem.mem[5]  = 32'h50610000; // LD  r6, [r1]
        dut.u_imem.mem[6]  = 32'h50730000; // LD  r7, [r3]
        dut.u_imem.mem[7]  = 32'h10867000; // VSUB r8 = r6-r7
        dut.u_imem.mem[8]  = 32'h60058000; // ST  [r5], r8
        dut.u_imem.mem[9]  = 32'hF0000000; // HALT

        // Kernel 2: RELU  (word addr 10-13)
        dut.u_imem.mem[10] = 32'h50410000; // LD  r4, [r1]
        dut.u_imem.mem[11] = 32'h40540000; // RELU r5 = r4
        dut.u_imem.mem[12] = 32'h60035000; // ST  [r3], r5
        dut.u_imem.mem[13] = 32'hF0000000; // HALT

        // Kernel 3: VMUL bf16  (word addr 14-18)
        dut.u_imem.mem[14] = 32'h50610000; // LD  r6, [r1]
        dut.u_imem.mem[15] = 32'h50730000; // LD  r7, [r3]
        dut.u_imem.mem[16] = 32'h21867000; // VMUL r8 = r6*r7  (bf16)
        dut.u_imem.mem[17] = 32'h60058000; // ST  [r5], r8
        dut.u_imem.mem[18] = 32'hF0000000; // HALT

        // Kernel 4: FMAC bf16  (word addr 19-24)
        // ACC note: FMAC encodes rs2=r9. gpu_core latches acc_lat from
        // rf_rs2_data one cycle before tensor fires = r9 = [2,2,2,2].
        // Effective: r8*r9+r9 = [1,2,3,4]*[2,2,2,2]+[2,2,2,2] = [4,6,8,10]
        dut.u_imem.mem[19] = 32'h50810000; // LD  r8,  [r1]
        dut.u_imem.mem[20] = 32'h50930000; // LD  r9,  [r3]
        dut.u_imem.mem[21] = 32'h50A50000; // LD  r10, [r5]  (loaded, not used by FMAC)
        dut.u_imem.mem[22] = 32'h31B89000; // FMAC r11 = r8*r9+acc(r9)
        dut.u_imem.mem[23] = 32'h6007B000; // ST  [r7], r11
        dut.u_imem.mem[24] = 32'hF0000000; // HALT

        repeat(2) @(posedge clk);

        // Release reset ? GPU will start from PC=0
        tb_rst_n = 1'b1;
        @(posedge clk);
    end
endtask

// ---------------------------------------------------------------------------
// Register pre-load helpers
// Write directly into generic_regfile.regs[] while GPU is running
// (safe because regfile is only written by wb on clock edge, not async)
// ---------------------------------------------------------------------------
task set_reg;
    input [3:0]  rn;
    input [63:0] val;
    begin
        dut.u_regfile.regs[rn] = val;
    end
endtask

// VADD / VSUB / VMUL: r1=src_a  r3=src_b  r5=dst
task load_regs_3;
    input [63:0] addr_a, addr_b, addr_dst;
    begin
        set_reg(4'd1, addr_a);
        set_reg(4'd3, addr_b);
        set_reg(4'd5, addr_dst);
    end
endtask

// RELU: r1=src_neg  r3=dst
task load_regs_relu;
    input [63:0] addr_neg, addr_dst;
    begin
        set_reg(4'd1, addr_neg);
        set_reg(4'd3, addr_dst);
    end
endtask

// FMAC: r1=bf_a  r3=bf_b  r5=bf_acc  r7=dst
task load_regs_fmac;
    input [63:0] addr_a, addr_b, addr_acc, addr_dst;
    begin
        set_reg(4'd1, addr_a);
        set_reg(4'd3, addr_b);
        set_reg(4'd5, addr_acc);
        set_reg(4'd7, addr_dst);
    end
endtask

// ---------------------------------------------------------------------------
// GPU wait and PC control
// ---------------------------------------------------------------------------
task gpu_wait_halt;
    input integer timeout_cycles;
    integer cnt;
    begin
        cnt = 0;
        while (!dut.u_gpu.halted && cnt < timeout_cycles) begin
            @(posedge clk);
            cnt = cnt + 1;
        end
        if (cnt >= timeout_cycles)
            $display("  GPU TIMEOUT: halted never asserted!");
    end
endtask

task gpu_set_pc;
    input [31:0] word_addr;
    begin
        force dut.u_gpu.u_pc.pc_out = word_addr;
        @(posedge clk);
        release dut.u_gpu.u_pc.pc_out;
    end
endtask

// ---------------------------------------------------------------------------
// Module-level temporaries (Verilog-2001: must be at module scope)
// ---------------------------------------------------------------------------
reg [63:0] scratch_result;
reg [31:0] pc_snap1, pc_snap2;
reg [15:0] tmp_lane;
integer    k;

// ---------------------------------------------------------------------------
// Main test sequence
// ---------------------------------------------------------------------------
initial begin
    clk            = 1'b0;
    tb_rst_n       = 1'b0;
    tb_imem_we     = 1'b0;
    tb_imem_addr   = 7'd0;
    tb_imem_data   = 32'd0;
    tb_dmem_we     = 1'b0;
    tb_dmem_wr_addr= 8'd0;
    tb_dmem_wr_data= 64'd0;
    tb_dmem_rd_addr= 8'd0;
    pass_count     = 0;
    fail_count     = 0;
    t_start        = 1'b0;
    t_op           = 3'd0;
    t_a            = 64'd0;
    t_b            = 64'd0;
    t_acc          = 64'd0;

    $display("=================================================================");
    $display("  GPU System Testbench  (datapath hierarchy)");
    $display("=================================================================");

    // Initial reset for tensor unit
    repeat(4) @(posedge clk);
    tb_rst_n = 1'b1;
    @(posedge clk);

    // =========================================================================
    // PART A ? tensor_bf16_4lane direct unit tests
    // =========================================================================
    $display("\n--- Part A: Tensor Unit Direct Tests ---");

    $display("\n[A1] RELU: [-1.0, 2.0, -2.0, 4.0] -> [0, 2, 0, 4]");
    tensor_fire(3'b100, pack4(BF_N1,BF_2,BF_N2,BF_4), 64'd0, 64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd,0); $write("  l0 exp 0x0000 got 0x%04h ",tmp_lane);
    check("A1_relu_l0", tmp_lane == BF_0);
    tmp_lane = lane(t_rd,1); $write("  l1 exp 0x4000 got 0x%04h ",tmp_lane);
    check("A1_relu_l1", tmp_lane == BF_2);
    tmp_lane = lane(t_rd,2); $write("  l2 exp 0x0000 got 0x%04h ",tmp_lane);
    check("A1_relu_l2", tmp_lane == BF_0);
    tmp_lane = lane(t_rd,3); $write("  l3 exp 0x4080 got 0x%04h ",tmp_lane);
    check("A1_relu_l3", tmp_lane == BF_4);
    @(posedge clk); @(posedge clk);

    $display("\n[A2] VADD: [1,2,3,4]+[1,1,1,1] = [2,3,4,5]");
    tensor_fire(3'b000, pack4(BF_1,BF_2,BF_3,BF_4),
                        pack4(BF_1,BF_1,BF_1,BF_1), 64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd,0); $write("  l0 exp 0x%04h got 0x%04h ",BF_2,tmp_lane);
    check("A2_vadd_l0", bf16_eq(tmp_lane,BF_2));
    tmp_lane = lane(t_rd,1); $write("  l1 exp 0x%04h got 0x%04h ",BF_3,tmp_lane);
    check("A2_vadd_l1", bf16_eq(tmp_lane,BF_3));
    tmp_lane = lane(t_rd,2); $write("  l2 exp 0x%04h got 0x%04h ",BF_4,tmp_lane);
    check("A2_vadd_l2", bf16_eq(tmp_lane,BF_4));
    tmp_lane = lane(t_rd,3); $write("  l3 exp 0x%04h got 0x%04h ",BF_5,tmp_lane);
    check("A2_vadd_l3", bf16_eq(tmp_lane,BF_5));
    @(posedge clk); @(posedge clk);

    $display("\n[A3] VSUB: [4,3,2,1]-[1,1,1,1] = [3,2,1,0]");
    tensor_fire(3'b001, pack4(BF_4,BF_3,BF_2,BF_1),
                        pack4(BF_1,BF_1,BF_1,BF_1), 64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd,0); $write("  l0 exp 0x%04h got 0x%04h ",BF_3,tmp_lane);
    check("A3_vsub_l0", bf16_eq(tmp_lane,BF_3));
    tmp_lane = lane(t_rd,1); $write("  l1 exp 0x%04h got 0x%04h ",BF_2,tmp_lane);
    check("A3_vsub_l1", bf16_eq(tmp_lane,BF_2));
    tmp_lane = lane(t_rd,2); $write("  l2 exp 0x%04h got 0x%04h ",BF_1,tmp_lane);
    check("A3_vsub_l2", bf16_eq(tmp_lane,BF_1));
    tmp_lane = lane(t_rd,3); $write("  l3 exp 0x0000 got 0x%04h ",tmp_lane);
    check("A3_vsub_l3", bf16_eq(tmp_lane,BF_0));
    @(posedge clk); @(posedge clk);

    $display("\n[A4] VMUL: [1,2,3,4]*[2,2,2,2] = [2,4,6,8]");
    tensor_fire(3'b010, pack4(BF_1,BF_2,BF_3,BF_4),
                        pack4(BF_2,BF_2,BF_2,BF_2), 64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd,0); $write("  l0 exp 0x%04h got 0x%04h ",BF_2,tmp_lane);
    check("A4_vmul_l0", bf16_eq(tmp_lane,BF_2));
    tmp_lane = lane(t_rd,1); $write("  l1 exp 0x%04h got 0x%04h ",BF_4,tmp_lane);
    check("A4_vmul_l1", bf16_eq(tmp_lane,BF_4));
    tmp_lane = lane(t_rd,2); $write("  l2 exp 0x%04h got 0x%04h ",BF_6,tmp_lane);
    check("A4_vmul_l2", bf16_eq(tmp_lane,BF_6));
    tmp_lane = lane(t_rd,3); $write("  l3 exp 0x%04h got 0x%04h ",BF_8,tmp_lane);
    check("A4_vmul_l3", bf16_eq(tmp_lane,BF_8));
    @(posedge clk); @(posedge clk);

    $display("\n[A5] FMAC: [1,2,3,4]*[2,2,2,2]+[1,1,1,1] = [3,5,7,9]");
    tensor_fire(3'b011, pack4(BF_1,BF_2,BF_3,BF_4),
                        pack4(BF_2,BF_2,BF_2,BF_2),
                        pack4(BF_1,BF_1,BF_1,BF_1));
    tensor_wait(20);
    tmp_lane = lane(t_rd,0); $write("  l0 exp 0x%04h got 0x%04h ",BF_3,tmp_lane);
    check("A5_fmac_l0", bf16_eq(tmp_lane,BF_3));
    tmp_lane = lane(t_rd,1); $write("  l1 exp 0x%04h got 0x%04h ",BF_5,tmp_lane);
    check("A5_fmac_l1", bf16_eq(tmp_lane,BF_5));
    tmp_lane = lane(t_rd,2); $write("  l2 exp 0x%04h got 0x%04h ",BF_7,tmp_lane);
    check("A5_fmac_l2", bf16_eq(tmp_lane,BF_7));
    tmp_lane = lane(t_rd,3); $write("  l3 exp 0x%04h got 0x%04h ",BF_9,tmp_lane);
    check("A5_fmac_l3", bf16_eq(tmp_lane,BF_9));
    @(posedge clk); @(posedge clk);

    $display("\n[A6] VMUL: [1,2,3,4]*0 = 0");
    tensor_fire(3'b010, pack4(BF_1,BF_2,BF_3,BF_4), 64'd0, 64'd0);
    tensor_wait(10);
    $write("  rd=0x%016h expect 0 ",t_rd);
    check("A6_vmul_zero", t_rd == 64'd0);
    @(posedge clk); @(posedge clk);

    $display("\n[A7] VADD: [3,3,3,3]+0 = [3,3,3,3]");
    tensor_fire(3'b000, pack4(BF_3,BF_3,BF_3,BF_3), 64'd0, 64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd,0); $write("  l0 exp 0x%04h got 0x%04h ",BF_3,tmp_lane);
    check("A7_vadd_identity", bf16_eq(tmp_lane,BF_3));
    @(posedge clk); @(posedge clk);

    $display("\n[A8] RELU: all-negative -> 0");
    tensor_fire(3'b100, pack4(BF_N1,BF_N2,BF_N1,BF_N2), 64'd0, 64'd0);
    tensor_wait(10);
    $write("  rd=0x%016h expect 0 ",t_rd);
    check("A8_relu_all_neg", t_rd == 64'd0);
    @(posedge clk); @(posedge clk);

    $display("\n[A9] RELU: [1,2,3,4] passthrough");
    tensor_fire(3'b100, pack4(BF_1,BF_2,BF_3,BF_4), 64'd0, 64'd0);
    tensor_wait(10);
    $write("  exp 0x%016h got 0x%016h ",pack4(BF_1,BF_2,BF_3,BF_4),t_rd);
    check("A9_relu_pos_pass", t_rd == pack4(BF_1,BF_2,BF_3,BF_4));
    @(posedge clk); @(posedge clk);

    // =========================================================================
    // PART B ? bf16_add edge cases
    // =========================================================================
    $display("\n--- Part B: bf16_add Edge Cases ---");

    $display("\n[B1] VADD: +inf+1.0 = +inf");
    tensor_fire(3'b000, pack4(16'h7F80,BF_0,BF_0,BF_0),
                        pack4(BF_1,BF_0,BF_0,BF_0), 64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd,0); $write("  l0 exp 0x7F80 got 0x%04h ",tmp_lane);
    check("B1_inf_plus_1", tmp_lane == 16'h7F80);
    @(posedge clk); @(posedge clk);

    $display("\n[B2] VADD: qNaN+1.0 -> NaN");
    tensor_fire(3'b000, pack4(16'hFF81,BF_0,BF_0,BF_0),
                        pack4(BF_1,BF_0,BF_0,BF_0), 64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd,0);
    $write("  l0=0x%04h exp=0x%02h (want FF) ",tmp_lane,tmp_lane[14:7]);
    check("B2_nan_prop", tmp_lane[14:7] == 8'hFF);
    @(posedge clk); @(posedge clk);

    $display("\n[B3] VSUB: 1.0-1.0 = +-0");
    tensor_fire(3'b001, pack4(BF_1,BF_0,BF_0,BF_0),
                        pack4(BF_1,BF_0,BF_0,BF_0), 64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd,0);
    $write("  l0=0x%04h (0x0000 or 0x8000 ok) ",tmp_lane);
    check("B3_sub_cancel", (tmp_lane==16'h0000)||(tmp_lane==16'h8000));
    @(posedge clk); @(posedge clk);

    $display("\n[B4] VADD: 1.0+2^-6 -> ~1.0");
    tensor_fire(3'b000, pack4(BF_1,BF_0,BF_0,BF_0),
                        pack4(16'h3B80,BF_0,BF_0,BF_0), 64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd,0);
    $write("  l0=0x%04h exp ~0x%04h ",tmp_lane,BF_1);
    check("B4_exp_diff_absorb", bf16_eq(tmp_lane,BF_1));
    @(posedge clk); @(posedge clk);

    // =========================================================================
    // PART C ? bf16_mul edge cases
    // =========================================================================
    $display("\n--- Part C: bf16_mul Edge Cases ---");

    $display("\n[C1] VMUL: 1.0*1.0 = 1.0");
    tensor_fire(3'b010, pack4(BF_1,BF_1,BF_1,BF_1),
                        pack4(BF_1,BF_1,BF_1,BF_1), 64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd,0); $write("  l0 exp 0x%04h got 0x%04h ",BF_1,tmp_lane);
    check("C1_mul_1x1", bf16_eq(tmp_lane,BF_1));
    @(posedge clk); @(posedge clk);

    $display("\n[C2] VMUL: +inf*0 = NaN");
    tensor_fire(3'b010, pack4(16'h7F80,BF_0,BF_0,BF_0),
                        pack4(BF_0,BF_0,BF_0,BF_0), 64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd,0);
    $write("  l0=0x%04h exp=0x%02h mant=0x%02h ",tmp_lane,tmp_lane[14:7],tmp_lane[6:0]);
    check("C2_inf_x_0", (tmp_lane[14:7]==8'hFF)&&(tmp_lane[6:0]!=7'd0));
    @(posedge clk); @(posedge clk);

    $display("\n[C3] VMUL: -1.0*-1.0 = +1.0");
    tensor_fire(3'b010, pack4(BF_N1,BF_0,BF_0,BF_0),
                        pack4(BF_N1,BF_0,BF_0,BF_0), 64'd0);
    tensor_wait(10);
    tmp_lane = lane(t_rd,0); $write("  l0 exp 0x%04h got 0x%04h ",BF_1,tmp_lane);
    check("C3_neg_x_neg", bf16_eq(tmp_lane,BF_1));
    @(posedge clk); @(posedge clk);

    $display("\n[C4] VMUL: 0*0 = 0");
    tensor_fire(3'b010, 64'd0, 64'd0, 64'd0);
    tensor_wait(10);
    $write("  rd=0x%016h expect 0 ",t_rd);
    check("C4_zero_x_zero", t_rd == 64'd0);
    @(posedge clk); @(posedge clk);

    // =========================================================================
    // PART D ? GPU system-level tests
    //
    // Each kernel:
    //   1. do_reset_and_load  ? holds rst_n=0, writes all 5 kernels into
    //                           inst_mem.mem[], then releases rst_n=1
    //   2. gpu_set_pc         ? force PC to kernel start address (if != 0)
    //   3. load_regs_*        ? write address registers into regfile.regs[]
    //   4. gpu_wait_halt      ? wait for HALT
    //   5. read dut.u_dmem.mem[word] and check results
    // =========================================================================
    $display("\n--- Part D: GPU System Tests ---");

    // -------------------------------------------------------------------------
    // D1: VADD  addr 0-4
    // LD r6,[r1]  LD r7,[r3]  VADD r8=r7+r6  ST [r5],r8  HALT
    // r1=ADDR_INT_A  r3=ADDR_INT_B  r5=ADDR_DST_I
    // Expected: [1+10, 2+20, 3+30, 4+40] = [11,22,33,44]
    // -------------------------------------------------------------------------
    $display("\n[D1] GPU VADD: [1,2,3,4]+[10,20,30,40] = [11,22,33,44]");
    do_reset_and_load;
    // PC already 0 after reset
    load_regs_3(ADDR_INT_A, ADDR_INT_B, ADDR_DST_I);
    gpu_wait_halt(80);

    scratch_result = dut.u_dmem.mem[ADDR_DST_I[9:3]];
    $display("  stored = 0x%016h", scratch_result);
    $write("  l0 exp 11  got %0d ", $signed(scratch_result[15:0]));
    check("D1_vadd_l0", scratch_result[15:0] == 16'd11);
    $write("  l1 exp 22  got %0d ", $signed(scratch_result[31:16]));
    check("D1_vadd_l1", scratch_result[31:16] == 16'd22);
    $write("  l2 exp 33  got %0d ", $signed(scratch_result[47:32]));
    check("D1_vadd_l2", scratch_result[47:32] == 16'd33);
    $write("  l3 exp 44  got %0d ", $signed(scratch_result[63:48]));
    check("D1_vadd_l3", scratch_result[63:48] == 16'd44);

    // -------------------------------------------------------------------------
    // D2: VSUB  addr 5-9
    // LD r6,[r1]  LD r7,[r3]  VSUB r8=r6-r7  ST [r5],r8  HALT
    // Expected: [1-10, 2-20, 3-30, 4-40] = [-9,-18,-27,-36]
    // -------------------------------------------------------------------------
    $display("\n[D2] GPU VSUB: [1,2,3,4]-[10,20,30,40] = [-9,-18,-27,-36]");
    do_reset_and_load;
    gpu_set_pc(32'd5);
    load_regs_3(ADDR_INT_A, ADDR_INT_B, ADDR_DST_I);
    gpu_wait_halt(80);

    scratch_result = dut.u_dmem.mem[ADDR_DST_I[9:3]];
    $display("  stored = 0x%016h", scratch_result);
    $write("  l0 exp -9   got %0d ", $signed(scratch_result[15:0]));
    check("D2_vsub_l0", $signed(scratch_result[15:0])  == -16'sd9);
    $write("  l1 exp -18  got %0d ", $signed(scratch_result[31:16]));
    check("D2_vsub_l1", $signed(scratch_result[31:16]) == -16'sd18);
    $write("  l2 exp -27  got %0d ", $signed(scratch_result[47:32]));
    check("D2_vsub_l2", $signed(scratch_result[47:32]) == -16'sd27);
    $write("  l3 exp -36  got %0d ", $signed(scratch_result[63:48]));
    check("D2_vsub_l3", $signed(scratch_result[63:48]) == -16'sd36);

    // -------------------------------------------------------------------------
    // D3: RELU  addr 10-13
    // LD r4,[r1]  RELU r5=r4  ST [r3],r5  HALT
    // r1=ADDR_NEG  r3=ADDR_DST_I
    // Expected: [-1,-2,-3,-4] -> [0,0,0,0]
    // -------------------------------------------------------------------------
    $display("\n[D3] GPU RELU: [-1,-2,-3,-4] -> [0,0,0,0]");
    do_reset_and_load;
    gpu_set_pc(32'd10);
    load_regs_relu(ADDR_NEG, ADDR_DST_I);
    gpu_wait_halt(80);

    scratch_result = dut.u_dmem.mem[ADDR_DST_I[9:3]];
    $display("  stored = 0x%016h (expect 0)", scratch_result);
    check("D3_relu_neg", scratch_result == 64'd0);

    // -------------------------------------------------------------------------
    // D4: VMUL bf16  addr 14-18
    // LD r6,[r1]  LD r7,[r3]  VMUL r8=r6*r7  ST [r5],r8  HALT
    // r1=ADDR_BF_A  r3=ADDR_BF_B  r5=ADDR_DST_B
    // Expected: [1,2,3,4]*[2,2,2,2] = [2,4,6,8]
    // -------------------------------------------------------------------------
    $display("\n[D4] GPU VMUL bf16: [1,2,3,4]*[2,2,2,2] = [2,4,6,8]");
    do_reset_and_load;
    gpu_set_pc(32'd14);
    load_regs_3(ADDR_BF_A, ADDR_BF_B, ADDR_DST_B);
    gpu_wait_halt(150);

    scratch_result = dut.u_dmem.mem[ADDR_DST_B[9:3]];
    $display("  stored = 0x%016h", scratch_result);
    tmp_lane = lane(scratch_result,0);
    $write("  l0 exp 0x%04h got 0x%04h ",BF_2,tmp_lane);
    check("D4_vmul_l0", bf16_eq(tmp_lane,BF_2));
    tmp_lane = lane(scratch_result,1);
    $write("  l1 exp 0x%04h got 0x%04h ",BF_4,tmp_lane);
    check("D4_vmul_l1", bf16_eq(tmp_lane,BF_4));
    tmp_lane = lane(scratch_result,2);
    $write("  l2 exp 0x%04h got 0x%04h ",BF_6,tmp_lane);
    check("D4_vmul_l2", bf16_eq(tmp_lane,BF_6));
    tmp_lane = lane(scratch_result,3);
    $write("  l3 exp 0x%04h got 0x%04h ",BF_8,tmp_lane);
    check("D4_vmul_l3", bf16_eq(tmp_lane,BF_8));

    // -------------------------------------------------------------------------
    // D5: FMAC bf16  addr 19-24
    // LD r8,[r1]  LD r9,[r3]  LD r10,[r5]  FMAC r11=r8*r9+acc  ST [r7],r11
    // r1=ADDR_BF_A  r3=ADDR_BF_B  r5=ADDR_BF_ACC  r7=ADDR_DST_F
    //
    // ACC note: FMAC encodes rs2=r9. gpu_core latches acc_lat from
    // rf_rs2_data the cycle before the tensor fires = r9 = [2,2,2,2].
    // Effective: [1,2,3,4]*[2,2,2,2]+[2,2,2,2] = [4,6,8,10]
    // r10 is loaded but not used (no rs2=r10 in the FMAC encoding).
    // -------------------------------------------------------------------------
    $display("\n[D5] GPU FMAC bf16: r8*r9+r9 = [4,6,8,10]");
    $display("     NOTE: acc=r9=[2,2,2,2] (FMAC encodes rs2=r9, not r10)");
    do_reset_and_load;
    gpu_set_pc(32'd19);
    load_regs_fmac(ADDR_BF_A, ADDR_BF_B, ADDR_BF_ACC, ADDR_DST_F);
    gpu_wait_halt(200);

    scratch_result = dut.u_dmem.mem[ADDR_DST_F[9:3]];
    $display("  stored = 0x%016h", scratch_result);
    tmp_lane = lane(scratch_result,0);
    $write("  l0 exp 4.0  (0x%04h) got 0x%04h ",BF_4, tmp_lane);
    check("D5_fmac_l0", bf16_eq(tmp_lane,BF_4));
    tmp_lane = lane(scratch_result,1);
    $write("  l1 exp 6.0  (0x%04h) got 0x%04h ",BF_6, tmp_lane);
    check("D5_fmac_l1", bf16_eq(tmp_lane,BF_6));
    tmp_lane = lane(scratch_result,2);
    $write("  l2 exp 8.0  (0x%04h) got 0x%04h ",BF_8, tmp_lane);
    check("D5_fmac_l2", bf16_eq(tmp_lane,BF_8));
    tmp_lane = lane(scratch_result,3);
    $write("  l3 exp 10.0 (0x%04h) got 0x%04h ",BF_10,tmp_lane);
    check("D5_fmac_l3", bf16_eq(tmp_lane,BF_10));

    // =========================================================================
    // PART E ? HALT and stall behaviour
    // =========================================================================
    $display("\n--- Part E: HALT and Stall Behaviour ---");

    $display("\n[E1] PC must not advance after HALT");
    do_reset_and_load;
    load_regs_3(ADDR_INT_A, ADDR_INT_B, ADDR_DST_I);
    gpu_wait_halt(80);
    @(posedge clk); pc_snap1 = dut.u_gpu.u_pc.pc_out;
    @(posedge clk); pc_snap2 = dut.u_gpu.u_pc.pc_out;
    $write("  pc_snap1=%0d  pc_snap2=%0d  ", pc_snap1, pc_snap2);
    check("E1_pc_freeze", pc_snap1 == pc_snap2);

    $display("\n[E2] halted flag must remain asserted");
    $write("  halted=%b  ", dut.u_gpu.halted);
    check("E2_halt_latch", dut.u_gpu.halted === 1'b1);

    $display("\n[E3] stall must be asserted when halted");
    $write("  stall=%b  ", dut.u_gpu.stall);
    check("E3_stall_asserted", dut.u_gpu.stall === 1'b1);

    $display("\n[E4] halted output port matches internal signal");
    $write("  tb_halted=%b  dut.u_gpu.halted=%b  ", tb_halted, dut.u_gpu.halted);
    check("E4_halted_port", tb_halted === dut.u_gpu.halted);

    $display("\n[E5] pc_out port matches internal PC");
    $write("  tb_pc_out=%0d  internal=%0d  ", tb_pc_out, dut.u_gpu.u_pc.pc_out);
    check("E5_pc_port", tb_pc_out == dut.u_gpu.u_pc.pc_out);

    // =========================================================================
    // Summary
    // =========================================================================
    $display("\n=================================================================");
    $display("  Test Summary:  PASS=%0d  FAIL=%0d  TOTAL=%0d",
             pass_count, fail_count, pass_count+fail_count);
    if (fail_count == 0)
        $display("  *** ALL TESTS PASSED ***");
    else
        $display("  *** %0d TEST(S) FAILED ***", fail_count);
    $display("=================================================================\n");

    #100;
    $finish;
end

// Watchdog
initial begin
    #200_000;
    $display("WATCHDOG: simulation exceeded 200 us");
    $finish;
end

// VCD dump
initial begin
    $dumpfile("tb_gpu_system.vcd");
    $dumpvars(0, tb_gpu_system);
end

endmodule