# Resources

## 1. Understanding Pipeline and VLA (High Level)

- [OpenVLA](https://github.com/openvla/openvla) — open-source VLA; start with the architecture section of the paper
- [LeRobot (HuggingFace)](https://github.com/huggingface/lerobot) — includes pi0 and other VLAs with runnable examples
- [Survey on Vision-Language-Action Models](https://arxiv.org/abs/2405.09175) (arXiv 2405.09175) — high-level map of the VLA space
- RT-2 blog post (Google DeepMind) — readable intro to how language+vision feeds into robot actions

## 2. Understanding Inference Algorithm

- [llm.c](https://github.com/karpathy/llm.c) — from-scratch transformer inference; excellent for understanding every compute step
- [LLM-AWQ (MIT-HAN-Lab)](https://github.com/mit-han-lab/llm-awq) — AWQ quantization and efficient inference, FPGA-relevant
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — key attention algorithm to implement; paper explains memory hierarchy tradeoffs
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic) — best single article on compute/memory bottleneck analysis

## 3. FPGA Board and Host Computer Selection

- [Vitis AI (Xilinx/AMD)](https://github.com/Xilinx/Vitis-AI) — official ML inference stack; shows supported boards (ZCU102, VCK190, Alveo U280)
- [FINN Framework (Xilinx)](https://github.com/Xilinx/finn) — neural net to FPGA dataflow architecture; good for understanding what's realistic on a given board
- Target boards to evaluate: **Alveo U280 / U50** (PCIe accelerator cards with HBM), **ZCU104** (embedded/standalone); HBM bandwidth is the key spec for transformer inference
- BittWare 250-SoC and AMD Alveo datasheets for detailed specs

## 4. PCIe Communication

- [XDMA Driver (Xilinx)](https://github.com/Xilinx/dma_ip_drivers) — standard Xilinx PCIe DMA driver for host↔FPGA data transfer
- [PCIe for FPGA Tutorial](https://github.com/enjoydigital/PCIe) — step-by-step PCIe on FPGAs with Vivado
- [RIFFA](https://github.com/KastnerRG/riffa) — well-documented PCIe framework, good for learning the protocol before using XDMA
- Xilinx PG195 (XDMA product guide, free PDF from AMD) — definitive reference for the PCIe DMA IP core

---

**Suggested learning order**: 1 → 2 → 4 → 3 (understand what you're running before picking hardware). The FINN repo bridges #2 and #3 well since it targets specific boards with realistic models.
