# OpenVLA-OFT: Architecture and Pipeline

## What is OpenVLA-OFT?

OpenVLA-OFT is a variant of OpenVLA that replaces autoregressive action token decoding with an MLP action head that directly regresses continuous robot actions from the LLM's hidden states at action-token positions. "OFT" refers to the fine-tuning approach used to train this action head on top of the frozen or lightly fine-tuned VLA backbone.

This is the architecture targeted by the FPGA accelerator in this project. The software reference is `L1RegressionActionHead` in `docs/reference/action_heads.py`.

---

## Pipeline

```
Camera ──► Vision Encoder (SigLIP ViT)
                │
                │  ~256 visual tokens
                ▼
Language ──► LLM Backbone (LLaMA)
           (vision tokens + language tokens + action tokens)
                │
                │  hidden states at action-token positions
                │  shape: (56, 4096)  = 8 chunk steps × 7 DOF tokens
                ▼
         L1RegressionActionHead          ← FPGA accelerator
         reshape → MLPResNet × 8 steps
                │
                │  [56] = 8 timesteps × 7 DOF
                ▼
         Robot Action Chunk
```

---

## Stage-by-Stage Breakdown

### 1. Vision Encoder — SigLIP ViT

- Input: one or more camera frames
- A Vision Transformer (ViT) splits the image into fixed-size patches and encodes each as a high-dimensional embedding
- SigLIP (Google) is pretrained to align image and text representations
- Output: ~256 visual tokens, each a vector in the LLM's embedding space after a learned projection

### 2. LLM Backbone — LLaMA

- Receives streams concatenated into a single token sequence:
  - Visual tokens (projected to LLaMA hidden dimension, 4096)
  - Tokenized language instruction (e.g. "pick up the red cube")
  - Action placeholder tokens (one per DOF per chunk timestep)
- Runs a full transformer forward pass over the combined sequence
- **Hidden states at action-token positions** are extracted after the last transformer layer

For LIBERO (`NUM_ACTIONS_CHUNK=8`, `ACTION_DIM=7`), this yields **56 hidden vectors**, each of dimension **4096** — not a single pooled embedding.

### 3. Action Head — `L1RegressionActionHead` / `MLPResNet`

The reference head reshapes hidden states and runs a shared `MLPResNet` once per chunk timestep:

```
Input hidden states:  (batch, 56, 4096)
Reshape:              (batch, 8, 28672)     where 28672 = 7 × 4096

Per chunk step, MLPResNet:
  LayerNorm(28672)
  → Linear(28672 → 4096) + ReLU
  → MLPResNetBlock(4096) × 2     [Pre-LN → Linear → ReLU + residual]
  → LayerNorm(4096)
  → Linear(4096 → 7)

Output:               (batch, 8, 7)  →  56 action values
```

| Dimension | Meaning |
|-----------|---------|
| 8 timesteps | predicted actions for the next 8 control steps (`NUM_ACTIONS_CHUNK`) |
| 7 DOF | 6 joint angles + 1 gripper value (`ACTION_DIM`) |
| 4096 | LLM / MLP hidden width (`input_dim`, `hidden_dim`) |
| 28672 | concatenated hidden states for all 7 action tokens at one chunk step |

**Activation function:** ReLU (no GELU in the reference head).

**Parameter count (LIBERO, one shared MLPResNet):** ~151M trainable parameters (~151 MB as INT8 weights). The dominant term is `Linear(28672, 4096)`. Weights must live in **off-chip memory** (DDR/HBM); on-chip BRAM holds activations and tiling buffers.

The LLM backbone remains ~7B parameters on the host. The action head is smaller than the backbone but **much larger than a toy 3-layer MLP** — FPGA design must account for weight streaming and/or layer tiling.

---

## Comparison: Standard OpenVLA vs OpenVLA-OFT

| | Standard OpenVLA | OpenVLA-OFT |
|---|---|---|
| Action output | Discrete tokens → detokenized to joint values | Continuous float values directly |
| Actions per inference | 1 | 8 (action chunk) |
| Inference path | Autoregressive token generation + detokenizer | Single `predict_action()` call (8 internal MLPResNet passes, shared weights) |
| Latency | High — N sequential decoding steps | Lower — fixed compute graph, no token recurrence |
| FPGA suitability | Poor — token decoding is sequential and irregular | Good — parallel MAC-friendly layers, but large weights and 8 passes |

The action detokenizer present in standard OpenVLA is **not part of this project's pipeline**. OpenVLA-OFT bypasses it entirely.

---

## FPGA Accelerator Boundary

The FPGA accelerates `L1RegressionActionHead.predict_action()` — the full `MLPResNet` forward pass, executed **8 times** per inference (once per chunk timestep, shared weights).

| | Details |
|---|---------|
| Input to FPGA | INT8-quantized action-token hidden states, shape `(56, 4096)` → **229,376 B (~224 KiB)** per inference, transferred over PCIe DMA |
| Optional host preprocess | Reshape to `(8, 28672)` before DMA; same byte count |
| Compute on FPGA | INT8 `MLPResNet` with INT32 accumulators, LayerNorm, ReLU, residual adds, requantization between layers; loop or pipeline over 8 chunk steps |
| Weights | ~151M INT8 params loaded once at init (DDR/HBM + DMA or dedicated weight load) |
| Output from FPGA | Action chunk `(8, 7)` = **56 FP32 values → 224 B**, transferred back over PCIe DMA |
| Host responsibility | Vision encoding, LLM forward pass, hidden-state extraction, action denormalization (`BOUNDS_Q99`), ROS dispatch |

**Software callable to replace:** `L1RegressionActionHead.predict_action(actions_hidden_states)` with `actions_hidden_states` of shape `(1, 56, 4096)`.

The PCIe round-trip plus FPGA compute (8 × MLPResNet) is the latency this project benchmarks against a CPU/GPU INT8 baseline. At this scale, **compute may dominate PCIe**, unlike a tiny 512-byte MLP — hardware performance counters are essential to know which.

---

## Platform Constants (LIBERO default)

From `docs/reference/constants.py`:

| Constant | Value |
|----------|-------|
| `NUM_ACTIONS_CHUNK` | 8 |
| `ACTION_DIM` | 7 |
| `PROPRIO_DIM` | 8 |
| `ACTION_PROPRIO_NORMALIZATION_TYPE` | `bounds_q99` |

Other platforms (ALOHA, BRIDGE) change chunk size and DOF; the accelerator should expose these as compile-time or MMIO parameters.

---

## Resources

- OpenVLA base repo: https://github.com/openvla/openvla
- OpenVLA-OFT paper: search arXiv for "OpenVLA-OFT" (approximately arXiv:2411.xxxxx)
- Local reference: `docs/reference/action_heads.py`, `docs/reference/constants.py`
- See also: `Pipeline.md` for the full end-to-end data flow and PCIe interface details
