# Notes
## Jun 18, 2026
- OpenVLA-OFT is a modified version of the OpenVLA with an empahasis on the modification of the original Action-detokenizer into a MLP-head
- The MVP of the hardware accelerator is a MLP, while in the actual using case the design can make be made more configurable
- Will need a PCIe DMA block
Per inference:
  1. host → DMA write input [512 B]  → input BRAM     (~1–2 µs PCIe)
  2. host → MMIO write CTRL.start=1
  3. FPGA: FSM runs MLP              (~sub-µs on FPGA)
  4. host → poll MMIO STATUS.done
  5. host → DMA read output [224 B]  → host buffer     (~1 µs PCIe)

## Jun 19, 2026
- Added some potential software extensions to play around with in extension.md, focuses on learning about kernals and CUDA
- Hardware design steps also documented in extension.md
- Planning on milestones and goals
