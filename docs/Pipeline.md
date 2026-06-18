# Pipeline

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                        Host (CPU)                       │
│                                                         │
│  Camera / Sensor ──► Vision Encoder                     │
│                           │                             │
│  Language Prompt ──► Language Model                     │
│                           │                             │
│               Fused Robot-Policy Embedding              │
│               (compact numeric feature vector)          │
└───────────────────────────┬─────────────────────────────┘
                            │ PCIe (DMA write)
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   FPGA Accelerator                      │
│                                                         │
│        Neural-Network Action Head (RTL)                 │
│        ┌──────────────────────────────┐                 │
│        │  Linear / MLP layers         │                 │
│        │  Activation functions        │                 │
│        │  Output projection           │                 │
│        └──────────────────────────────┘                 │
│                                                         │
│               Robot Action Vector                       │
└───────────────────────────┬─────────────────────────────┘
                            │ PCIe (DMA read)
                            ▼
┌─────────────────────────────────────────────────────────┐
│                        Host (CPU)                       │
│                                                         │
│        OpenVLA Backend Interface                        │
│                   │                                     │
│        ROS Node Wrapper                                 │
│                   │                                     │
│        /robot_action  (ROS topic)                       │
└─────────────────────────────────────────────────────────┘
```

## Stages

### 1. Feature Extraction (Host)

The vision encoder and language model run on the host (GPU/CPU) as part of the normal OpenVLA forward pass. Their combined output is a **fused robot-policy embedding**: a compact, fixed-size numeric feature vector. This vector is the sole input to the accelerator.

### 2. PCIe Transfer (Host → FPGA)

The host DMA-writes the feature vector into a pre-mapped FPGA BAR region. A lightweight PCIe driver handles address mapping, doorbell signaling, and completion polling or interrupts.

### 3. Action-Head Inference (FPGA)

The FPGA implements the action-generation network in RTL. On receiving the feature vector, it runs the forward pass through the policy head layers and writes the resulting action vector to an output buffer.

Architecture targets:
- Fixed-point arithmetic (INT8 or INT16) for area and power efficiency
- Pipelined datapath for throughput; fully unrolled small layers for latency
- Parameterized layer widths to allow retargeting without full RTL rewrite

### 4. PCIe Transfer (FPGA → Host)

The host DMA-reads (or polls) the output buffer to retrieve the robot action vector once the FPGA signals completion.

### 5. OpenVLA Backend Interface

A Python backend adapter wraps the PCIe driver and presents the same interface as the software policy head, making the FPGA a drop-in replacement within the OpenVLA inference loop.

### 6. ROS Node Wrapper

A ROS node subscribes to sensor/observation topics, calls the OpenVLA backend (which dispatches to the FPGA), and publishes the resulting action on a standard ROS action topic. This is the integration point for real robot hardware.

## Development Phases

### Phase 1 — Software MVP

- Implement the action head as a standalone Python/PyTorch module
- Validate numeric outputs against reference OpenVLA checkpoint
- Establish CPU latency baseline with benchmarking harness

### Phase 2 — RTL Implementation

- Write RTL for the action-head layers targeting a development FPGA board
- Simulate and verify functional correctness against software MVP
- Synthesize, place, and route; measure hardware latency

### Phase 3 — PCIe Integration

- Develop PCIe BAR-mapped DMA driver (Linux kernel module or XDMA-based)
- Integrate host-side Python driver with software MVP
- Run end-to-end latency benchmarks: CPU path vs FPGA path

### Phase 4 — OpenVLA Integration

- Replace OpenVLA action-head call with FPGA backend adapter
- Validate that robot command quality is preserved
- Profile full inference pipeline (vision encode + language model + FPGA action head)

### Phase 5 — ROS Wrapper

- Implement ROS node that wraps the integrated OpenVLA + FPGA stack
- Test in simulation and on hardware robot

## Interfaces

| Interface | Type | Description |
|-----------|------|-------------|
| Feature vector in | Float32 / INT8 tensor, PCIe DMA | Fused embedding from host model |
| Action vector out | Float32 / INT8 tensor, PCIe DMA | Per-joint or end-effector action |
| FPGA control | MMIO registers | Start, status, interrupt enable |
| OpenVLA backend | Python callable | `infer_action(obs) -> action` |
| ROS topic | `sensor_msgs`, custom action msg | Observation in, action command out |

## Benchmarking

The primary metric is **action-head inference latency** measured from the moment the feature vector is ready on the host to the moment the action vector is available for dispatch.

Measurements to collect:
- CPU baseline: mean, p50, p95, p99 latency over N=1000 calls
- FPGA path: same statistics including PCIe round-trip overhead
- Breakdown: PCIe transfer time vs compute time on FPGA
