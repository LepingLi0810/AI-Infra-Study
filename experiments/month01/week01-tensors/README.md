# Month 1, week 1: tensors and your first GPU operation

Start with session 1 below. The goal for today is to understand and run a small
matrix projection, not to install an inference server or download a model.

## The month in four checkpoints

| Week | Main question | Evidence to produce |
| --- | --- | --- |
| 1 | What executes when I multiply tensors on a GPU? | Correct CPU/GPU projection, shape/layout explanation, machine inventory |
| 2 | How does a transformer turn tokens into next-token scores? | Tiny causal attention example, annotated shapes, small-model forward pass |
| 3 | What does cached decoding avoid recomputing? | Cached/uncached generation comparison and a KV-memory estimate |
| 4 | Where does inference spend time and memory? | Small prefill/decode experiment with warm-up, clear timing boundaries, and a report |

These checkpoints implement the [study plan](../../../STUDY_PLAN.md). Only the
first exercise is implemented here; later exercises should build on what you learn.

## Session 1 — storage, shapes, and a projection (60–90 minutes)

A dense tensor combines storage with metadata. For this lesson, track five things:

- **Shape:** the logical dimensions, such as two tokens with three features each.
- **Dtype:** the representation of each element, such as 32-bit floating point.
- **Device:** the CPU or GPU on which the storage lives.
- **Strides:** how many elements to advance in storage for each dimension.
- **Storage sharing:** whether two tensor objects reference the same memory.

This is close to reasoning about array descriptors and aliases in systems code.
For a dense strided tensor, an element's position is its storage offset plus the
sum of each index multiplied by that dimension's stride. Strides are measured in
elements, not bytes. A transpose can change shape/strides without moving data;
making a noncontiguous transpose contiguous requires a copy in this example.

Now consider a linear projection:

```text
X: [tokens, input_features]   = [2, 3]
W: [input_features, output_features] = [3, 2]
b: [output_features]         = [2]
Y = X @ W + b                = [2, 2]
```

Each output element is a dot product plus bias. `@` is matrix multiplication;
`*` is elementwise multiplication subject to broadcasting. The bias broadcasts
over rows without requiring you to construct a separate bias vector per token.
PyTorch's `nn.Linear` stores its weight as `[output_features, input_features]`,
so its computation is `X @ weight.T + bias`; our hand-written `W` is transposed
relative to that convention.

Before running the script, calculate the following on paper:

```text
X = [[1, 2, 3],       W = [[1, 0],       b = [0.5, 1.5]
     [4, 5, 6]]           [0, 1],
                          [1, 1]]
```

Predict the output, `X`'s logical byte size in float32, and the shapes/strides of
`X` and `X.T`. Then use the exercise to check your prediction.

## Setup: run on the machine you actually have

From your repository root in WSL, first run:

```bash
nvidia-smi
python3 --version
```

Record the GPU model, VRAM, driver, and Python version. The CUDA version displayed
by `nvidia-smi` describes driver compatibility; it does not prove that a local CUDA
toolkit is installed. These exercises use prebuilt PyTorch wheels and do not need
`nvcc`. WSL GPU support depends on the Windows NVIDIA driver and a supported WSL
configuration; do not install a Linux display driver inside WSL to fix visibility.

The tested CPU setup uses Python 3.12 and PyTorch 2.8.0. This is an explicit study
baseline, not a claim that it is the newest release. Create a dedicated environment:

```bash
python3 -m venv .venv-month1
source .venv-month1/bin/activate
```

If venv creation fails on Ubuntu because `ensurepip` is unavailable, install the
matching Python venv package through your system package manager and retry.

Choose **one** installation path for this environment:

**CPU-only machine, or learning the tensor semantics before GPU setup:**

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  -r experiments/month01/week01-tensors/requirements.txt
```

**RTX 3060 visible in WSL, with a driver compatible with the CUDA 12.6 wheel:**

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu126 \
  -r experiments/month01/week01-tensors/requirements.txt
```

The WSL driver and GPU checks are still pending; the CUDA command has not been
validated on your machine. Check the official [PyTorch version/build matrix](https://pytorch.org/get-started/previous-versions/)
and [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
if the installed Python/driver differs. Do not disable TLS verification to fix
installation errors. Use separate environments for CPU and CUDA builds if switching
between them; the same package version can otherwise leave the previous build installed.

Run from the repository root:

```bash
python experiments/month01/week01-tensors/tensors.py --device cpu
# Also run this on the GPU-equipped machine:
python experiments/month01/week01-tensors/tensors.py --device cuda
```

CPU mode explicitly reports the CUDA comparison as unrun. CUDA mode fails if the
GPU is unavailable; it does not silently fall back to CPU. Read the code before
running it and inspect each printed shape, stride, and correctness check.

## Session 2 — explore layout and broadcasting (60–90 minutes)

In a separate scratch script, try the following one at a time:

1. Change `bias` from shape `[2]` to `[1, 2]`. Predict whether the result changes.
2. Try a bias with shape `[3]`. Explain the broadcasting error in terms of trailing dimensions.
3. Compare `X.T` and `X.T.contiguous()`: same values, different layout and aliasing.
4. Create token features of shape `[B, S, H]` and weights `[H, O]`. Predict the
   output shape, then check it. Which dimensions represent independent tokens?
5. Change float32 to float64 and predict the logical storage size. Keep comparisons
   between compatible dtypes; byte size alone does not predict runtime.

Do not weaken the reference checks in the supplied script to make an altered
calculation pass. For an intentional new calculation, derive its expected result.

## Session 3 — connect to GPU execution (60–90 minutes)

Run the exercise in CUDA mode after verifying the driver and build. Explain:

- Creating a CUDA tensor places storage on the GPU. Moving existing data with
  `.to("cuda")` generally requires a transfer and may allocate device storage.
- Many CUDA operations are asynchronous relative to the host. Returning from a
  Python call does not necessarily mean the GPU has finished computing.
- Moving a result back to CPU to inspect it introduces transfer and synchronization
  costs. This exercise checks correctness and deliberately does not benchmark.
- A tiny multiplication may be slower on a GPU once overhead is included. We will
  study workload size and timing boundaries before claiming speedups.

Think of the distinction between submitting asynchronous work and observing its
completion, drawing on your concurrency/RDMA background. CUDA has its own ordering
rules; the analogy is a starting point, not a substitute for learning those rules.

`torch.inference_mode()` disables autograd tracking for this exercise. Later, when
using a model, also call `model.eval()` to select evaluation behavior for layers
such as dropout. These two operations solve different problems.

## Read only what you need this week

Use the [PyTorch tensor tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
alongside sessions 1–2. Read the [broadcasting notes](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
when explaining session 2. Save attention papers and serving-runtime internals
for the later checkpoints.

## Checkpoint and handoff

Write a short report in `reports/` with the machine inventory, exact commands,
predictions, observed outputs, and unresolved questions. Record the source commit
and any local edits, plus `python -m pip freeze` for the actual package versions.
Inspect captured metadata before committing it; do not record credential-bearing URLs.

You are ready for week 2 when you can:

- Explain `[B, S, H] @ [H, O] -> [B, S, O]` and compute the tiny example by hand.
- Explain shape versus stride, broadcasting, and when a transpose shares memory.
- Run the CPU checks and, on your RTX 3060, the CUDA/CPU comparison.
- Explain why CPU wall time around an asynchronous launch is not sufficient for
  kernel timing.

If GPU setup is blocked, continue the CPU lessons and record GPU validation as
pending. Passing this supplied script is evidence about the environment and example;
your explanations and modifications are the evidence of learning.
