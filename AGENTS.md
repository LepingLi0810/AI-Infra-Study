# Project guidance

This file applies throughout the repository. Keep shared guidance here; add a
nested AGENTS.md only when a subproject needs different instructions.

## Purpose and learner background

- Help a computer systems PhD student transition into LLM inference infrastructure
  during the year before graduation.
- Existing strengths: concurrency, synchronization, distributed systems, RDMA,
  and resource scheduling.
- Starting knowledge: beginner in PyTorch, transformer inference, and CUDA/Triton.
  Explain new ML/GPU concepts clearly and connect them to familiar systems ideas.
- The learner has an RTX 3060 and may rent cloud GPUs. The GPU's VRAM capacity is
  not yet confirmed. Do not assume it is attached to the current node.
- The study plan assumes 8–10 hours per week; this is not a confirmed commitment.

## Start each task

- Read README.md and the relevant sections of STUDY_PLAN.md. The study plan is
  the source for milestones and reading priorities; avoid duplicating it here.
- Check Git status and inspect relevant notes, reports, and experiment instructions.
  Preserve existing work and do not infer completed milestones from the plan.
- Use the current checkout. Cloud tasks are already isolated; do not create a
  Git worktree unless the user explicitly requests one.
- Treat current user instructions as authoritative when they change the plan.

## Work across machines

- Keep these instructions portable: use repository-relative paths in documentation
  and scripts, and inspect OS, GPU/VRAM, driver, and tool availability as needed.
- Git shares committed source, documentation, and experiment specifications.
  Python environments, model caches, credentials, datasets, and running services
  need separate setup on each node; do not assume they were synchronized.
- Before GPU work, check the actual device and framework/runtime compatibility.
  On a CPU-only node, complete useful CPU work and identify GPU checks as unrun.
- Use isolated dependency environments and pin versions for runnable experiments.
  Record setup and execution commands in each experiment's README.
- Keep secrets, model weights, virtual environments, and large traces out of Git.
  Check ignore rules before generating artifacts inside the checkout.
- Do not automatically pull, reset, or overwrite local work to synchronize nodes.
  Report local versus committed/pushed state accurately at handoff.

## Learning and experiments

- Prioritize model execution, GPU measurement, batching, scheduling, and KV-cache
  management. Use vLLM as the initial serving runtime unless the task calls for another.
- Start with small, understandable examples and small models. Establish correctness
  before optimization; avoid adding unrelated frameworks or large dependencies.
- For experiments, document the question, prediction, configuration, commands,
  results, explanation, and limitations. Record hardware and software/model revisions.
- Distinguish CPU simulations, GPU measurements, and untested predictions. Report
  latency distributions, throughput, memory, and failure behavior when relevant.
- Keep durable explanations in notes/, executable studies in experiments/,
  measurement tools in benchmarks/, and results/handoffs in reports/. Create these
  directories when needed, not as empty scaffolding.
- After substantive study or experiment work, leave a brief dated report with
  completed work, reproducible commands, unresolved issues, and the next step so
  another session can continue without relying on chat history.
- Update this file when durable project preferences change. Keep transient machine
  state and detailed progress in experiment reports rather than this file.
