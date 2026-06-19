# OpenVLA-OFT: Architecture and Pipeline

## What is OpenVLA-OFT?

OpenVLA-OFT is a variant of OpenVLA that replaces autoregressive action token decoding with a small MLP action head that directly regresses continuous robot actions from the LLM's last hidden state. "OFT" refers to the fine-tuning approach used to train this action head on top of the frozen or lightly fine-tuned VLA backbone.

This is the architecture targeted by the FPGA accelerator in this project.

---

## Pipeline

```
Camera ──► Vision Encoder (SigLIP ViT)
                │
                │  ~256 visual tokens
                ▼
Language ──► LLM Backbone (LLaMA)
           (vision tokens + language tokens fused)
                │
                │  last hidden state [4096]
                ▼
           Linear projection
                │
                │  [512]
                ▼
         Action Head MLP          ← FPGA accelerator
         512 → 256 → 128 → 56
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

- Receives two streams concatenated into a single token sequence:
  - Visual tokens (projected to match LLaMA's hidden dimension, e.g. 4096)
  - Tokenized language instruction (e.g. "pick up the red cube")
- Runs a full transformer forward pass over the combined sequence
- The **hidden state at the final token position** is extracted after the last transformer layer

This vector is the "fused robot-policy embedding" referenced in Pipeline.md. It encodes what action the robot should take given the current image and instruction.

### 3. Action Head MLP

The hidden state is projected down and passed through a small MLP:

```
[512] → Linear + ReLU → [256] → Linear + ReLU → [128] → Linear → [56]
```

Output: 56 values representing an **action chunk** — 8 future timesteps × 7 degrees of freedom.

| Dimension | Meaning |
|-----------|---------|
| 8 timesteps | predicted actions for the next 8 control steps |
| 7 DOF | 6 joint angles + 1 gripper value (typical robot arm) |

The MLP is approximately 200K parameters — orders of magnitude smaller than the LLM backbone (~7B parameters).

---

## Comparison: Standard OpenVLA vs OpenVLA-OFT

| | Standard OpenVLA | OpenVLA-OFT |
|---|---|---|
| Action output | Discrete tokens → detokenized to joint values | Continuous float values directly |
| Actions per inference | 1 | 8 (action chunk) |
| Inference path | Autoregressive token generation + detokenizer | Single MLP forward pass |
| Latency | High — N sequential decoding steps | Low — fixed compute, no recurrence |
| FPGA suitability | Poor — token decoding is sequential and irregular | Excellent — MLP is parallel, fixed-size, quantizable |

The action detokenizer present in standard OpenVLA is **not part of this project's pipeline**. OpenVLA-OFT bypasses it entirely.

---

## FPGA Accelerator Boundary

The FPGA accelerator handles exactly the action head MLP:

| | Details |
|---|---------|
| Input to FPGA | INT8-quantized feature vector [512], transferred over PCIe DMA |
| Compute on FPGA | INT8 MLP forward pass with INT32 accumulators and requantization between layers |
| Output from FPGA | Action vector [56], transferred back to host over PCIe DMA |
| Host responsibility | Everything else: vision encoding, LLM forward pass, hidden state projection, ROS dispatch |

The PCIe round-trip (host→FPGA→host) plus FPGA compute time is the latency this project benchmarks against a CPU-only INT8 baseline.

---

## Resources

- OpenVLA base repo: https://github.com/openvla/openvla
- OpenVLA-OFT paper: search arXiv for "OpenVLA-OFT" (approximately arXiv:2411.xxxxx)
- See also: `Pipeline.md` for the full end-to-end data flow and PCIe interface details
