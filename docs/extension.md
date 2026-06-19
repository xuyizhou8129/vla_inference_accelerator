# Project Extensions

Extensions are grouped into a **Hardware / Codesign Track** (PCIe integration + RTL architecture) and a **Software Track** (CUDA kernels + GPU baseline). Both tracks are independently valuable and pair naturally at benchmark time.

---

## Hardware / Codesign Track — Recommended Path

This track is ordered as a learning path: establish the baseline PCIe path first, add measurement, redesign the compute engine, then overlap PCIe with compute, and optionally explore completion models on the host.

Recommended sequence:

0. **Baseline PCIe path** — H2C DMA → MMIO start → compute → poll done → C2H DMA (prerequisite; not an extension)
1. **Hardware Performance Counters** — measure where time goes before optimizing
2. **Systolic Array Dataflow** — architecture playground for the MAC datapath
3. **Double Buffering / Ping-Pong** — PCIe deep-dive: overlap DMA and compute
4. **Interrupt-Driven Completion vs. Polling** *(optional)* — host-side completion tradeoffs

At batch size 1 (~512 B in, ~224 B out), counters will likely show that PCIe transfer dominates FPGA compute. That result guides whether architecture work or DMA overlap matters more for the robot use case.

### 0. Baseline PCIe Path (Prerequisite)

Per inference:

1. Host → DMA write input [512 B] → input BRAM (~1–2 µs PCIe)
2. Host → MMIO write `CTRL.start=1`
3. FPGA: FSM runs MLP (~sub-µs on FPGA)
4. Host → poll MMIO `STATUS.done`
5. Host → DMA read output [224 B] → host buffer (~1 µs PCIe)

Until this works end-to-end (XDMA, BAR mapping, buffer addresses, driver), later extensions add confusion rather than insight. See `Pipeline.md` Phase 3 and `resources.md` § PCIe Communication.

### 1. Hardware Performance Counters

Add MMIO-readable cycle counters that the host software reads after each inference:

| Counter | Measures |
|---------|---------|
| `CYCLE_DMA_H2C` | PCIe write latency (host → FPGA input buffer) |
| `CYCLE_LAYER_0/1/2` | Compute cycles per MLP layer |
| `CYCLE_DMA_C2H` | PCIe read latency (FPGA output buffer → host) |
| `CYCLE_IDLE` | Cycles the compute unit spent waiting |

The software runtime reads these registers after each call and logs a per-inference breakdown. This makes benchmarking tell you *where* time is spent rather than just total latency — which is necessary to guide any further optimization.

**Add this immediately after baseline.** Hardware cost is a handful of 32-bit counters gated by FSM state signals — low area, high value. Without counters, PCIe vs. compute tradeoffs are guesswork.

### 2. Systolic Array Dataflow

Replace the flat MAC array with a weight-stationary or output-stationary systolic array for matrix multiplication. Each PE (processing element) computes one partial sum and passes data to its neighbor, enabling high utilization without a central accumulator bottleneck. PCIe stays on the same burst-DMA pattern; this extension redesigns the compute engine inside the FPGA.

Design decisions this forces:
- **Dataflow choice**: weight stationary vs. output stationary vs. input stationary — each has different reuse patterns and BRAM/register tradeoffs
- **Stall-free feeding**: input and weight streams must be scheduled to keep all PEs busy
- **Pipeline fill and drain**: first and last cycles of each layer are partial — the controller must handle this correctly
- **Array sizing**: number of PEs is a resource vs. latency tradeoff
- **Weight BRAM layout**: access order must match the systolic array's fetch pattern (same intuition as CUDA memory layout tuning in the software track)

Reference: Kung (1978) systolic array paper; Google TPU architecture.

### 3. Double Buffering / Ping-Pong

Overlap PCIe transfer of inference N+1 with FPGA compute of inference N using two input buffer banks. While the MLP processes data in bank A, the DMA engine fills bank B. The controller flips the active bank after each inference completes.

Why this is the main PCIe deep-dive:
- Requires coordinating two independent FSMs: the DMA controller and the compute controller
- Software must issue DMA writes speculatively (before the previous result is read back)
- Turns a pure latency story into a throughput story — relevant when the robot controller queries the policy at fixed frequency
- Exposes a real tradeoff: double buffering doubles input BRAM usage

**Add after baseline and counters work.** This is where PCIe stops being "one transfer per inference" and becomes a system you coordinate.

### 4. Interrupt-Driven Completion vs. Polling *(Optional)*

Implement both completion mechanisms:
- **Polling**: host spins on a STATUS register until `done=1`
- **MSI interrupt**: FPGA raises a PCIe Message Signaled Interrupt on completion; host sleeps until the ISR fires

Selectable by the software runtime via a control register at startup.

Then measure: at what CPU load level does interrupt-driven outperform polling? Polling wastes a CPU core but has lower latency when the CPU is otherwise idle. Interrupts free the CPU but add ISR overhead. This is a real systems tradeoff with a quantitative answer.

Hardware requirement: MSI wiring in RTL (XDMA exposes this) and a Linux interrupt handler in the host driver. Low datapath cost — good host-side PCIe learning without further RTL architecture changes.

---

## Software Track — Kernel Optimization & PyTorch Integration

These extensions run alongside the hardware track and are independently valuable. They are ordered as a learning path: start by writing CUDA kernels from scratch (threads, blocks, warps, memory coalescing), then profile and tune what you wrote, then benchmark against compiler-optimized PyTorch to establish a realistic GPU baseline for FPGA comparison.

Recommended sequence:

1. **Custom CUDA Kernel** — learn how kernels and threads work by building the MLP by hand
2. **Nsight Compute Roofline Analysis** — measure and tune the kernel you wrote
3. **torch.compile Profiling** — compare your hand-written kernel against an already-optimized GPU path and the FPGA

### 1. Custom CUDA Kernel (GPU Baseline for the MLP)

Write a hand-rolled CUDA kernel that performs the same three-layer MLP computation the FPGA accelerates. This gives a like-for-like GPU vs. FPGA comparison rather than comparing the FPGA against unoptimized PyTorch eager mode. It is also the primary vehicle for learning how GPU kernels, threads, and the memory hierarchy work.

Build incrementally rather than fusing all three layers on day one:

1. Implement a **single-layer GEMM** kernel first
2. Add **tiling and shared memory** reuse
3. **Fuse GELU** into the GEMM loop to avoid a round-trip to HBM between compute and activation
4. Stack layers into the full three-layer MLP

What this forces you to reason about:
- **Thread block tiling**: choose tile dimensions (e.g., 16×16 or 32×32) that balance register pressure against shared memory reuse for each GEMM
- **Memory layout**: row-major vs. column-major weight storage; padding rows to 128-byte alignment for coalesced loads
- **Activation fusion**: fold the GELU into the GEMM loop to avoid a round-trip to HBM between layers
- **Occupancy vs. ILP**: wider tiles improve reuse but reduce the number of resident warps

Tools: `nvcc` for compilation; expose the kernel to PyTorch via `torch.utils.cpp_extension` so it can be called from the existing MLP benchmark harness.

### 2. Nsight Compute Roofline Analysis and Memory Layout Tuning

Profile the CUDA MLP kernel from step 1 with `ncu --set full` to get the roofline position (is the kernel compute-bound or memory-bandwidth-bound?), then iterate on weight tensor layout to improve utilization:

- Transpose weight matrices to match the access order inside the tiled GEMM (avoid strided column reads)
- Pad rows to the next multiple of 32 elements (128 bytes) to guarantee coalesced loads
- Interleave channel data if the activation function reads adjacent channels together

This step teaches profiling and performance reasoning on top of kernel writing — connect code structure to achieved vs. theoretical FLOP/s, cache hit rates, and bandwidth utilization.

The same analysis applies directly to the systolic array RTL: if the GPU kernel's bottleneck is non-sequential HBM access, the same bottleneck exists in the FPGA unless the weight BRAM is laid out to match the array's access pattern. Working through this in software first builds the intuition needed to fix it in hardware.

### 3. torch.compile Profiling and End-to-End Latency Comparison

Apply `torch.compile(mlp, backend="inductor")` to the MLP head and systematically benchmark four configurations: eager PyTorch, compiled PyTorch (Inductor), the custom CUDA kernel, and the FPGA. Use `torch.profiler` with `profile_memory=True` and Chrome trace export to get per-operator breakdowns.

This answers the central question of the project quantitatively: how much does the FPGA actually save over an already-optimized GPU path, and at what batch size?

At batch size 1 — the regime a real robot controller runs in — GPU kernel launch overhead and driver latency dominate. This is where the FPGA's deterministic, fixed-latency path should be most visible. If it isn't, the profiler trace tells you exactly why. The comparison table (latency, variance, CPU utilization) becomes the core result section of any write-up.

### Tie-In Between Tracks

- **Systolic Array** ↔ CUDA tiling and memory layout — same reuse and access-pattern thinking
- **Hardware Performance Counters** ↔ torch.compile comparison table — both answer "where does time go?"
- **Double Buffering** ↔ batch-size-1 latency analysis — counters show whether PCIe overlap is worth the BRAM cost

---

## Recommended Combination (Best Depth-to-Scope Ratio)

For a strong hardware-software codesign project without excessive scope:

**Hardware / codesign:**
1. **Hardware performance counters** — makes benchmarking rich before and after every other change
2. **Systolic array** — replaces flat MAC with a principled dataflow architecture
3. **Double buffering** — overlaps DMA and compute, requires FSM coordination
4. **Interrupt vs. polling** *(optional)* — host-side PCIe completion tradeoffs

**Software (parallel track):**
1. **Custom CUDA kernel** → **Nsight roofline** → **torch.compile profiling**

These extensions share a common theme: hardware and software are designed *together* rather than independently, which is the core of codesign.
