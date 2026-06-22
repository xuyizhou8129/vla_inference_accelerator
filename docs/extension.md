# Project Extensions

Extensions are grouped into a **Hardware / Codesign Track** (PCIe integration + RTL architecture) and a **Software Track** (CUDA kernels + GPU baseline). Both tracks are independently valuable and pair naturally at benchmark time.

The baseline accelerator targets **`L1RegressionActionHead` / `MLPResNet`** from `docs/reference/action_heads.py` (LIBERO: input `(56, 4096)`, eight shared-weight chunk steps, ~151M INT8 params).

---

## Hardware / Codesign Track — Recommended Path

This track is ordered as a learning path: establish the baseline PCIe path first, add measurement, redesign the compute engine, then overlap PCIe with compute, and optionally explore completion models on the host.

Recommended sequence:

0. **Baseline PCIe path** — H2C DMA → MMIO start → compute → poll done → C2H DMA (prerequisite; not an extension)
1. **Hardware Performance Counters** — measure where time goes before optimizing
2. **Systolic Array Dataflow** — architecture playground for the MAC datapath
3. **Double Buffering / Ping-Pong** — PCIe deep-dive: overlap DMA and compute
4. **Interrupt-Driven Completion vs. Polling** *(optional)* — host-side completion tradeoffs

At batch size 1 (~224 KiB in, ~224 B out), counters may show **either** PCIe H2C **or** MLPResNet compute (8 × large GEMMs + weight fetch from DDR/HBM) as the bottleneck — unlike a toy 512-byte MLP where PCIe always wins. Measure before optimizing.

### 0. Baseline PCIe Path (Prerequisite)

**One-time at init:** load ~151 MB INT8 weights into FPGA DDR/HBM (dedicated weight DMA or host populate).

**Per inference (LIBERO, batch=1):**

1. Host → DMA write input [**~224 KiB**] → input buffer — `(56, 4096)` INT8 action-token hidden states
2. Host → MMIO write `CTRL.start=1`
3. FPGA: FSM runs **MLPResNet × 8 chunk steps** (LayerNorm, 28672→4096, 2× residual block, 4096→7; ReLU activations)
4. Host → poll MMIO `STATUS.done`
5. Host → DMA read output [**224 B**] → host buffer — `(8, 7)` FP32 action chunk

Until this works end-to-end (XDMA, BAR mapping, buffer addresses, off-chip weight path, driver), later extensions add confusion rather than insight. See `Pipeline.md` Phase 3 and `resources.md` § PCIe Communication.

### 1. Hardware Performance Counters

Add MMIO-readable cycle counters that the host software reads after each inference:

| Counter | Measures |
|---------|-----------|
| `CYCLE_DMA_H2C` | PCIe write latency (host → FPGA input buffer) |
| `CYCLE_WEIGHT_FETCH` | Cycles stalled waiting on DDR/HBM weight reads |
| `CYCLE_LAYERNORM` | LayerNorm passes (input + hidden) |
| `CYCLE_FC1` | Linear(28672 → 4096) + ReLU |
| `CYCLE_RESBLOCK_0/1` | Residual MLP blocks |
| `CYCLE_FC2` | Linear(4096 → 7) |
| `CYCLE_CHUNK_STEP` | Per chunk-step subtotal (×8 per inference) |
| `CYCLE_DMA_C2H` | PCIe read latency (FPGA output buffer → host) |
| `CYCLE_IDLE` | Cycles the compute unit spent waiting |

The software runtime reads these registers after each call and logs a per-inference breakdown. This makes benchmarking tell you *where* time goes rather than just total latency — essential when compute and PCIe are comparable.

**Add this immediately after baseline.** Hardware cost is a handful of 32-bit counters gated by FSM state signals — low area, high value.

### 2. Systolic Array Dataflow

Replace the flat MAC array with a weight-stationary or output-stationary systolic array for matrix multiplication. The first target is **`Linear(28672, 4096)`** — the dominant layer — then reuse the same array for the 4096→4096 blocks and smaller projections.

Design decisions this forces:
- **Dataflow choice**: weight stationary vs. output stationary vs. input stationary — each has different reuse patterns and BRAM/register tradeoffs
- **Stall-free feeding**: input activation and weight streams must be scheduled to keep all PEs busy while streaming from DDR/HBM
- **Pipeline fill and drain**: first and last cycles of each layer are partial — the controller must handle this correctly
- **Array sizing**: number of PEs is a resource vs. latency tradeoff; 28672→4096 sets the critical tile geometry
- **Weight layout**: off-chip storage order must match the systolic fetch pattern (same intuition as CUDA memory layout tuning in the software track)

Reference: Kung (1978) systolic array paper; Google TPU architecture.

### 3. Double Buffering / Ping-Pong

Overlap PCIe transfer of inference N+1 with FPGA compute of inference N using two input buffer banks. While MLPResNet processes chunk steps from bank A, the DMA engine fills bank B with the next `(56, 4096)` hidden-state tensor.

Why this is the main PCIe deep-dive:
- Requires coordinating two independent FSMs: the DMA controller and the compute controller
- Software must issue DMA writes speculatively (before the previous result is read back)
- Turns a pure latency story into a throughput story — relevant when the robot controller queries the policy at fixed frequency
- Exposes a real tradeoff: double buffering doubles **input** buffer size (~448 KiB for LIBERO INT8 inputs)

**Add after baseline and counters work.** If counters show compute dominates H2C, ping-pong input buffers help throughput more than raw latency.

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

1. **Custom CUDA Kernel** — learn how kernels and threads work by building MLPResNet layers by hand
2. **Nsight Compute Roofline Analysis** — measure and tune the kernel you wrote
3. **torch.compile Profiling** — compare your hand-written kernel against an already-optimized GPU path and the FPGA

### 1. Custom CUDA Kernel (GPU Baseline for MLPResNet)

Write hand-rolled CUDA kernels that perform the same **`MLPResNet`** computation the FPGA accelerates (matching `docs/reference/action_heads.py`). This gives a like-for-like GPU vs. FPGA comparison rather than comparing the FPGA against unoptimized PyTorch eager mode.

Build incrementally:

1. Implement **`Linear(28672, 4096)` GEMM** first — the dominant layer
2. Add **tiling and shared memory** reuse
3. **Fuse ReLU** into the GEMM epilogue (reference head uses ReLU, not GELU)
4. Add **LayerNorm** and **residual add** for `MLPResNetBlock`
5. Stack into full MLPResNet and wrap in an **8-iteration chunk loop** (shared weights)

What this forces you to reason about:
- **Thread block tiling**: tile dimensions (e.g., 16×16 or 32×32) for the large and small GEMMs
- **Memory layout**: row-major vs. column-major weight storage; padding for coalesced loads
- **Activation fusion**: fold ReLU into the GEMM epilogue to avoid an HBM round-trip
- **Occupancy vs. ILP**: wider tiles improve reuse but reduce resident warps
- **Weight size**: ~151M params — kernel design must respect GPU HBM bandwidth

Tools: `nvcc` for compilation; expose the kernel to PyTorch via `torch.utils.cpp_extension` so it can be called from the `L1RegressionActionHead` benchmark harness.

### 2. Nsight Compute Roofline Analysis and Memory Layout Tuning

Profile the CUDA MLPResNet kernel from step 1 with `ncu --set full` to get the roofline position (compute-bound vs. memory-bandwidth-bound?), then iterate on weight tensor layout:

- Transpose weight matrices to match tiled GEMM access order (avoid strided column reads)
- Pad rows to the next multiple of 32 elements (128 bytes) for coalesced loads
- Separate analysis for **28672→4096** vs. **4096→4096** layers

The same analysis applies directly to the systolic array RTL: if the GPU kernel is weight-bandwidth-bound, the FPGA faces the same pressure from DDR/HBM unless weights are laid out for sequential burst reads.

### 3. torch.compile Profiling and End-to-End Latency Comparison

Apply `torch.compile` to `L1RegressionActionHead` and systematically benchmark four configurations: eager PyTorch, compiled PyTorch (Inductor), the custom CUDA MLPResNet kernel, and the FPGA. Use `torch.profiler` with `profile_memory=True` and Chrome trace export.

This answers the central question quantitatively: how much does the FPGA save over an already-optimized GPU path for the **full action head** (8 chunk steps), and at what batch size?

At batch size 1 — the regime a real robot controller runs in — compare GPU kernel launch overhead, weight bandwidth, and FPGA deterministic compute. The comparison table (latency, variance, CPU utilization, layer breakdown) becomes the core result section of any write-up.

### Tie-In Between Tracks

- **Systolic Array** ↔ CUDA tiling and weight layout — same reuse and access-pattern thinking for 28672→4096
- **Hardware Performance Counters** ↔ torch.compile comparison table — both answer "where does time go?"
- **Double Buffering** ↔ throughput at fixed control rate — counters show whether input ping-pong is worth ~448 KiB buffer cost

---

## Recommended Combination (Best Depth-to-Scope Ratio)

For a strong hardware-software codesign project without excessive scope:

**Hardware / codesign:**
1. **Hardware performance counters** — per-layer and per-chunk-step breakdown
2. **Systolic array** — target 28672→4096 first
3. **Double buffering** — overlap H2C with 8-step compute
4. **Interrupt vs. polling** *(optional)* — host-side PCIe completion tradeoffs

**Software (parallel track):**
1. **Custom CUDA MLPResNet kernel** → **Nsight roofline** → **torch.compile profiling**

These extensions share a common theme: hardware and software are designed *together* rather than independently, which is the core of codesign.
