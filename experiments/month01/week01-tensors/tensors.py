"""A small correctness exercise, not a performance benchmark."""

import argparse
import platform

import torch


def describe(name, tensor):
    print(
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, stride={tensor.stride()}, "
        f"contiguous={tensor.is_contiguous()}, "
        f"logical_bytes={tensor.numel() * tensor.element_size()}"
    )


@torch.inference_mode()
def run(device):
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}; wheel CUDA runtime: {torch.version.cuda}")
    print(f"Selected device: {device}")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Check GPU visibility, "
                "the driver, and the installed PyTorch build. "
                "Use --device cpu only for the CPU exercise."
            )
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}; VRAM: {props.total_memory / 2**30:.2f} GiB")

    # X: two token vectors, each with three features. W: a 3 -> 2 projection.
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device)
    w = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32, device=device)
    bias = torch.tensor([0.5, 1.5], dtype=torch.float32, device=device)
    for name, tensor in (("X", x), ("W", w), ("bias", bias)):
        describe(name, tensor)

    y = x @ w + bias
    expected = torch.tensor([[4.5, 6.5], [10.5, 12.5]])
    # Hand-computed reference. Exact equality is appropriate for these tiny,
    # exactly representable values, not for general floating-point model outputs.
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    print("X @ W + bias:", y.cpu().tolist())
    print("PASS: projection and broadcast bias match the hand calculation")

    # Leading batch/sequence dimensions survive projection over the last axis.
    tokens = x.unsqueeze(0)  # [batch=1, sequence=2, hidden=3]
    projected = tokens @ w + bias
    describe("batched tokens", tokens)
    describe("projected tokens", projected)
    torch.testing.assert_close(projected.cpu(), expected.unsqueeze(0), rtol=0, atol=0)
    print("PASS: batched projection has the expected values and shape")

    transposed = x.T
    packed = transposed.contiguous()
    describe("X.T", transposed)
    describe("X.T.contiguous()", packed)
    assert transposed.data_ptr() == x.data_ptr(), "Transpose should share storage"
    assert packed.data_ptr() != x.data_ptr(), "This contiguous copy needs new storage"
    torch.testing.assert_close(transposed, packed, rtol=0, atol=0)
    print("PASS: transpose shares storage; contiguous copy preserves values")

    # General CPU/GPU comparisons require declared numerical tolerances.
    if device == "cuda":
        cpu_result = x.cpu() @ w.cpu() + bias.cpu()
        torch.testing.assert_close(y.cpu(), cpu_result, rtol=1e-5, atol=1e-6)
        print("PASS: CUDA result agrees with CPU")
    else:
        print("UNRUN: CUDA/CPU comparison (CPU mode selected)")
    print("No timing or performance conclusions are produced by this exercise.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run(parser.parse_args().device)
