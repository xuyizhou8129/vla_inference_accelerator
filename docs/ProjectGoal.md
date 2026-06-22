# Project Goal: Tiny VLA / Robot Policy Inference Accelerator

## Summary

Build an FPGA-based accelerator that executes the OpenVLA-OFT `L1RegressionActionHead` (`MLPResNet`, ~151M params, 8 chunk steps per inference) and benchmark whether the hardware path achieves lower and more deterministic latency than CPU/GPU execution.

## Problem Statement

Vision-Language-Action (VLA) models like OpenVLA produce robot commands by running a large neural network at inference time. The action-generation (policy head) stage is the latency-critical path during closed-loop robot control: every millisecond of jitter translates directly to control instability.

CPUs can run this stage, but they suffer from:
- Non-deterministic latency due to OS scheduling, cache pressure, and thermal throttling
- Poor throughput when the policy is queried at high frequency

An FPGA offers fixed-function, cycle-deterministic execution with a predictable power envelope, making it a strong candidate for the real-time leg of a robot stack.

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Functional correctness | FPGA output matches software model to within acceptable numeric tolerance |
| Latency | FPGA path achieves measurably lower mean latency than CPU baseline |
| Determinism | FPGA p99 latency is within 2x of mean (low jitter) |
| Integration | Accelerator backend plugs into OpenVLA inference pipeline via a clean interface |
| ROS compatibility | A ROS node wrapper exposes the accelerator as a standard action topic |

## Scope

**In scope**
- Software golden model of `L1RegressionActionHead` / `MLPResNet` matching `docs/reference/action_heads.py`
- RTL implementation of the action-generation network on FPGA
- PCIe host-device driver and DMA data path
- Latency benchmarking harness (CPU vs FPGA)
- OpenVLA backend integration
- ROS node wrapper

**Out of scope (for now)**
- Vision encoder or language model acceleration (run on host GPU/CPU)
- Full end-to-end VLA model on FPGA
- Multi-FPGA scaling

## Relationship to OpenVLA

OpenVLA (https://github.com/openvla/openvla) is the upstream VLA framework this project targets. The FPGA accelerator is designed to be dropped in as a backend inference engine for the action-generation stage, with the rest of the OpenVLA stack (vision encoder, language model, tokenizer) continuing to run on the host.
