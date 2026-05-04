# ConvDiffLGNs Paper Mapping

This note maps `Convolutional Differentiable Logic Gate Networks` (`ConvDiffLGNs.pdf` in the repo root) to the current `torchlogix` codebase.

## What the paper introduces

The paper extends differentiable logic gate networks with:

- Convolutional logic-tree kernels instead of flat random dense connectivity
- Logical `or`-pooling, relaxed as a max operation
- Residual initialization so deep logic networks do not collapse toward `0.5`
- CIFAR-10 and MNIST architectures built from repeated conv-block-plus-pooling stages

The core paper claim is that these pieces make logic networks practical for image tasks and much more hardware-efficient than prior logic/BNN baselines.

## Direct code mapping

- Logic-tree convolutions: `src/torchlogix/layers/conv.py`
  - `LogicConv2d` and `LogicConv3d` implement kernels as trees of LUT-based logic gates.
  - `tree_depth=3` matches the paper's default conv blocks: `2^3 = 8` leaves, `7` trainable gates per kernel.
- Fixed random receptive-field connections: `src/torchlogix/connections.py`
  - `FixedConvConnections` samples tree leaves from a receptive field and reuses the same kernel parameters across spatial positions.
  - `channel_group_size` is the main knob that restricts how many input channels a kernel can draw from.
- Logical `or`-pooling: `src/torchlogix/layers/pool.py`
  - `OrPooling2d`/`OrPooling3d` are implemented with `max_pool`, which is the relaxation the paper uses for logical OR.
- Residual initialization: `src/torchlogix/parametrization.py`
  - `weight_init="residual"` and `residual_probability` are the code-level version of the paper's residual gate initialization.
- Appendix architectures: `src/torchlogix/models/conv.py`
  - `ClgnMnist*` matches the MNIST appendix pattern: `5x5` conv, then `3x3`, `3x3`, each followed by `or`-pooling.
  - `ClgnCifar10*` matches the CIFAR appendix pattern: four `3x3` conv blocks with pooling after each block, then three dense logic layers and `GroupSum`.

## Important repo-specific clarifications

- `ClgnCifar10Small` (`k=32`) and `ClgnCifar10Medium` (`k=256`) line up with the paper's `S` and `M` widths.
- `ClgnCifar10Large` uses `k=512`, which corresponds to the paper's `B` width, not its `L` (`1024`) or `G` (`2048`) widths.
- `ClgnCifar10Small2` and `ClgnCifar10Medium2` are repo variants that restrict the first conv layer to a single input channel via `group_size_input = 1`. This matches the README's stronger CIFAR-10 setup, not the paper's base architecture naming.
- The paper describes additional hardware-routing constraints with fixed multi-group channel partitioning. The repo clearly supports local channel restriction via `channel_group_size`, but it does not obviously implement the paper's full `k=8` routing partition as a separate architectural constraint.
- The paper also discusses fused low-level CUDA kernels, teacher-based training for larger models, and 5-bit edge/curvature preprocessing for larger CIFAR variants. Those pieces are not obvious in the current high-level Python model definitions and training script, so requests for exact paper reproduction should be checked carefully.

## Practical guidance for repo work

When a change is supposed to stay faithful to the paper, preserve these defaults unless the task says otherwise:

- Use conv blocks with `tree_depth=3` and `receptive_field_size=3` on CIFAR-10.
- Keep `OrPooling2d(kernel_size=2, stride=2)` after each conv block.
- Prefer `weight_init="residual"` over random initialization for deep conv models.
- For paper-like CIFAR models, treat channel restriction as part of the architecture, not just an optimization detail.
- Distinguish "paper-faithful" questions from "TorchLogix-specific" questions: this repo adds learnable binarization, extra parametrizations (`warp`, `light`), compiled inference, and Verilog export that are beyond the original paper.
