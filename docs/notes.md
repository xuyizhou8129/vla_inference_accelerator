# Notes
## Jun 18, 2026
- OpenVLA-OFT is a modified version of the OpenVLA with an empahasis on the modification of the original Action-detokenizer into a MLP-head
- The MVP of the hardware accelerator is `L1RegressionActionHead` / `MLPResNet` from `docs/reference/action_heads.py`
- Will need a PCIe DMA block and off-chip weight storage (~151 MB INT8)
Per inference (LIBERO, batch=1):
  1. host → DMA write input [~224 KiB]  → input BRAM/DDR  (56×4096 INT8 action-token hiddens)
  2. host → MMIO write CTRL.start=1
  3. FPGA: FSM runs MLPResNet × 8 chunk steps (shared weights)
  4. host → poll MMIO STATUS.done
  5. host → DMA read output [224 B]  → host buffer     (8×7 FP32 actions)

## Jun 19, 2026
- Added some potential software extensions to play around with in extension.md, focuses on learning about kernals and CUDA
- Hardware design steps also documented in extension.md
- Planning on milestones and goals

Milestones:
1-Golden software model in python (L1RegressionActionHead / MLPResNet)
2-RTL sim vs Golden
2'- CUDA track
3 - Hardware extensions
4 - OpenVLA integration

## Jun 21, 2026
- Reading the action head software code on OpenVLA-OFT repo

L1RegressionActionHead
    └── MLPResNet
            └── MLPResNetBlock

DiffusionActionHead
    ├── NoisePredictionModel → MLPResNet → MLPResNetBlock
    ├── SinusoidalPositionalEncoding   (timestep embedding)
    └── DDIMScheduler                    (noise schedule)

- Work on L1RegressionActionHead (matches reference, not simplified 512→256→128 MVP)
- Treat DiffusionHead as an extension for later

## Architecture summary (reference-aligned MVP)
- Input: action-token hidden states (56, 4096), reshape to (8, 28672) per batch item
- MLPResNet per chunk step: LN → 28672→4096 ReLU → 2× residual 4096 blocks → LN → 4096→7
- Activation: ReLU
- Params: ~151M (~151 MB INT8 weights)
- Output: (8, 7) = 56 action values

x
│
▼
LayerNorm(input_dim)          ← normalize each feature
│
▼
Linear(input_dim → hidden_dim)
│
▼
ReLU
│
▼
┌─────────────────────────┐
│  MLPResNetBlock × N     │   (reference uses N=2)
│    LayerNorm            │
│    Linear(h → h)        │
│    ReLU                 │
│    + skip from input    │   ← residual add
└─────────────────────────┘
│
▼
LayerNorm(hidden_dim)
│
▼
Linear(hidden_dim → output_dim)
│
▼
y
