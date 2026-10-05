# Week 1 preparation — 2026-10-05

## State and question

Prepared the first lesson on tensor storage, shape, strides, broadcasting, device
placement, and matrix projection. This records agent validation of the exercise;
it does not mark the learner's week-one work complete.

Source base: `a9bf680817d18484caeb4035736c903f87383f12`, with local additions for
the lesson, requirements, ignore rules, this report, and a README link. These
lesson changes have not been committed or pushed as of this report.

Question: does the small projection match a hand-computed reference and preserve
its semantics under batching and changes of memory layout?

Prediction: `X @ W + bias` is `[[4.5, 6.5], [10.5, 12.5]]`; float32 `X` occupies
24 logical bytes; `X.T` shares storage, while its contiguous copy does not.

## Environment and commands

Validated on the Linux cloud node using Python 3.12.14, PyTorch 2.8.0+cpu, and
NumPy 2.2.6. No GPU or `nvidia-smi` is available here. The virtual environment
is outside the checkout at `/workspace/venvs/ai-infra-month1`; this path is local
state, not a portable requirement. Use the lesson's setup commands on other nodes.

From the repository root, with that Python environment active:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  -r experiments/month01/week01-tensors/requirements.txt
python experiments/month01/week01-tensors/tensors.py --device cpu
python -m pip check
```

## Observed results

- Passed: projection/bias matches the hand-computed reference.
- Passed: batched projection matches expected shape and values.
- Passed: transpose shares storage; contiguous copy uses different storage and preserves values.
- Passed: dependency consistency check.
- Passed: explicitly requesting CUDA on this CPU node returns a nonzero exit with
  a diagnostic, rather than silently running on CPU.
- Unrun: actual CUDA operations and CPU/GPU numerical comparison.
- No timing or performance claims were evaluated.

The observed CPU results agree with the predictions. The script uses tiny,
exactly representable values; this does not establish general numerical accuracy
for transformer inference or tolerance choices for larger calculations.

## Next step

Obtain `nvidia-smi` and `python3 --version` output from the learner's WSL machine,
confirm its VRAM/driver compatibility, and select the corresponding PyTorch wheel.
The learner should predict the example output and explain the shape/stride metadata,
then run the CPU exercise and the CUDA comparison. Begin attention only after the
tensor and device concepts are clear. Weeks 2–4 remain planned, not implemented.
