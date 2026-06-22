# Pipeline

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                        Host (CPU/GPU)                   │
│                                                         │
│  Camera / Sensor ──► Vision Encoder (SigLIP)            │
│                           │                             │
│  Language Prompt ──► LLM (LLaMA)                        │
│                           │                             │
│               Action-token hidden states                │
│               shape (56, 4096) for LIBERO               │
└───────────────────────────┬─────────────────────────────┘
                            │ PCIe (DMA write ~224 KiB)
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   FPGA Accelerator                      │
│                                                         │
│        L1RegressionActionHead / MLPResNet (RTL)         │
│        ┌──────────────────────────────┐                 │
│        │  LayerNorm + Linear + ReLU   │                 │
│        │  Residual MLP blocks (×2)    │                 │
│        │  × 8 chunk steps (shared W)  │                 │
│        └──────────────────────────────┘                 │
│        Weights (~151 MB INT8) in DDR/HBM                │
│                                                         │
│               Robot Action Chunk (8 × 7)                │
└───────────────────────────┬─────────────────────────────┘
                            │ PCIe (DMA read 224 B)
                            ▼
┌─────────────────────────────────────────────────────────┐
│                        Host (CPU)                       │
│                                                         │
│        Denormalize (BOUNDS_Q99)                         │
│        OpenVLA Backend Interface                        │
│                   │                                     │
│        ROS Node Wrapper                                 │
│                   │                                     │
│        /robot_action  (ROS topic)                       │
└─────────────────────────────────────────────────────────┘
```

## Stages

### 1. Feature Extraction (Host)

The vision encoder and language model run on the host (GPU/CPU) as part of the normal OpenVLA-OFT forward pass. The accelerator input is the **last-layer hidden states at action-token positions**, shape `(NUM_ACTIONS_CHUNK × ACTION_DIM, hidden_dim)` — for LIBERO, `(56, 4096)`.

This matches `L1RegressionActionHead.predict_action()` in `docs/reference/action_heads.py`. The host does **not** apply a separate 512-dim projection before the FPGA boundary.

### 2. PCIe Transfer (Host → FPGA)

The host DMA-writes the quantized hidden-state tensor into a pre-mapped FPGA BAR region (~224 KiB INT8 per inference at batch size 1). A lightweight PCIe driver handles address mapping, doorbell signaling, and completion polling or interrupts.

Model weights (~151M INT8 parameters) are loaded once at initialization via a separate weight-load path (DDR/HBM populate + optional DMA).

### 3. Action-Head Inference (FPGA)

The FPGA implements `MLPResNet` in RTL:

1. Reshape / interpret input as `(8, 28672)` — seven concatenated 4096-dim token hiddens per chunk step
2. For each of 8 chunk steps (shared weights):
   - LayerNorm(28672) → Linear(28672→4096) → ReLU
   - Two residual blocks (Pre-LN → Linear(4096→4096) → ReLU + skip)
   - LayerNorm(4096) → Linear(4096→7)
3. Write output buffer `(8, 7)` = 56 FP32 values

Architecture targets:
- Fixed-point arithmetic (INT8 weights/activations, INT32 accumulators)
- Weight streaming from off-chip memory; on-chip BRAM for activation tiles
- Pipelined datapath or dedicated FSM loop over chunk steps
- Parameterized widths tied to `constants.py` (`NUM_ACTIONS_CHUNK`, `ACTION_DIM`, `hidden_dim`)

### 4. PCIe Transfer (FPGA → Host)

The host DMA-reads (or polls) the output buffer to retrieve the action chunk once the FPGA signals completion.

### 5. OpenVLA Backend Interface

A Python backend adapter wraps the PCIe driver and implements the same contract as `L1RegressionActionHead.predict_action()`:

```python
def predict_action(actions_hidden_states: Tensor) -> Tensor:
    # in:  (batch, 56, 4096)   LIBERO
    # out: (batch, 8, 7)
```

This makes the FPGA a drop-in replacement within the OpenVLA inference loop.

### 6. ROS Node Wrapper

A ROS node subscribes to sensor/observation topics, calls the OpenVLA backend (which dispatches to the FPGA), and publishes the resulting action on a standard ROS action topic. This is the integration point for real robot hardware.

## Development Phases

### Phase 1 — Software MVP

- Implement `L1RegressionActionHead` / `MLPResNet` as a standalone PyTorch module matching `docs/reference/action_heads.py`
- Validate numeric outputs against an OpenVLA-OFT checkpoint
- Establish CPU/GPU latency baseline with benchmarking harness (include 8-step MLPResNet loop)

### Phase 2 — RTL Implementation

- Write RTL for `MLPResNet` targeting a development FPGA board with off-chip weight storage
- Simulate and verify functional correctness against software MVP (per chunk step, then full 8-step inference)
- Synthesize, place, and route; measure hardware latency and resource use

### Phase 3 — PCIe Integration

- Develop PCIe BAR-mapped DMA driver (Linux kernel module or XDMA-based)
- Integrate host-side Python driver with software MVP
- Run end-to-end latency benchmarks: CPU path vs FPGA path; break out weight load vs per-inference cost

### Phase 4 — OpenVLA Integration

- Replace OpenVLA `L1RegressionActionHead` call with FPGA backend adapter
- Validate that robot command quality is preserved
- Profile full inference pipeline (vision encode + LLM + FPGA action head)

### Phase 5 — ROS Wrapper

- Implement ROS node that wraps the integrated OpenVLA + FPGA stack
- Test in simulation and on hardware robot

## Interfaces

| Interface | Type | Shape (LIBERO) | Size (B=1, INT8 in / FP32 out) |
|-----------|------|----------------|--------------------------------|
| Action-token hiddens in | INT8 tensor, PCIe DMA | `(56, 4096)` | ~224 KiB |
| Action chunk out | FP32 tensor, PCIe DMA | `(8, 7)` | 224 B |
| Model weights | INT8, loaded at init | ~151M params | ~151 MB |
| FPGA control | MMIO registers | — | Start, status, chunk count, interrupt enable |
| OpenVLA backend | Python callable | `predict_action(hiddens) -> actions` | Same as reference head |
| ROS topic | `sensor_msgs`, custom action msg | Observation in, action command out | — |

## Benchmarking

The primary metric is **action-head inference latency** measured from the moment action-token hidden states are ready on the host to the moment the `(8, 7)` action chunk is available for dispatch.

Measurements to collect:
- CPU/GPU baseline: mean, p50, p95, p99 latency over N=1000 calls
- FPGA path: same statistics including PCIe round-trip overhead
- Breakdown: PCIe H2C vs compute (per layer / per chunk step) vs PCIe C2H vs weight-fetch stalls
- Compare against expectation that ~151M-param compute may dominate ~224 KiB PCIe at batch 1
