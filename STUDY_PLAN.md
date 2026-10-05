# From computer systems research to LLM inference

## Goal and assumptions

By graduation, be able to explain an inference request from arrival to GPU execution, diagnose its performance, modify a serving runtime, and evaluate a scheduling or memory-management idea against credible baselines. Aim for a reproducible project and one substantive upstream contribution; a publication is a possible outcome, not a requirement for a successful transition.

Your starting point is a systems PhD background, limited familiarity with PyTorch, transformers, and CUDA/Triton, and access to **one RTX 3060**, with cloud rentals possible. This draft assumes roughly 12 months and **8–10 hours per week**, alongside dissertation work; the time budget is a planning assumption, not a confirmed commitment.

The first three months deliberately build the missing ML/GPU foundation. Your RTX 3060 is the primary development machine. Cloud GPUs are optional for targeted validation later, not a prerequisite for starting.

**Recommended specialization: serving systems and inference performance, with depth in scheduling and KV-cache management.** This uses your existing strengths while leaving a natural route into distributed inference and RDMA. Learn enough model internals and kernel programming to reason across layers; you do not need to become a model-training researcher first.

## Use your existing expertise

| Your background | Direct application to inference | New concepts to learn |
| --- | --- | --- |
| Concurrency and synchronization | Continuous batching, asynchronous execution, cache ownership, cancellation | CUDA streams/events, CPU–GPU coordination, graph capture constraints |
| Distributed systems | Multi-worker execution, routing, backpressure, failure handling | Tensor/pipeline parallelism, collective communication, model and KV placement |
| RDMA | KV transfer and communication between inference workers | GPU memory registration, GPUDirect RDMA, topology and overlap; these depend on hardware support |
| Resource scheduling | Admission control, fairness, preemption, latency objectives | Prefill/decode interference, unknown output lengths, token-level scheduling |
| Systems research methods | Bottleneck analysis, controlled experiments, reproducibility | Token workload distributions, quality/performance tradeoffs, GPU timing |

Treat a GPU as a new execution and memory model to learn. CPU concurrency intuition helps, but GPU barriers, occupancy, memory coalescing, and asynchronous launches require their own mental model.

## Working rhythm and priorities

Spend about two hours reading, four to five hours implementing or measuring, and two hours analyzing and writing each week. Reserve remaining time for debugging. Finish each month with a short note containing one question, a result, an explanation, and the next decision.

**Core:** model execution, GPU measurement, one serving runtime, a reliable load generator, batching/KV management, and the capstone. **Optional:** a second runtime, multi-node RDMA experiments, custom high-performance attention kernels, and advanced decoding methods. If graduation work gets busy, reduce optional breadth and the number of experimental configurations.

Do not spend the year recreating a production inference engine. Build small mechanisms to understand them, then evaluate your main idea in an existing runtime.

## Months 1–3: understand and measure inference

### Month 1 — Model execution and inference economics

Start with PyTorch tensors, indexing, broadcasting, matrix multiplication, device placement, and dtypes. Refresh only the math you need: matrix shapes, dot products, softmax, and basic probability. Understand what training and autograd do, then use evaluation/inference mode for this study; a full training project is unnecessary.

Next learn the forward pass of a decoder-only transformer: embeddings, attention, positional information, normalization, MLP, logits, and sampling. Cover numerical tolerance and distinguish prefill from autoregressive decode and what the KV cache stores. Use diagrams and tiny tensors before reading optimized implementation code.

Write a tiny causal-attention example, then a transparent inference loop around a small pretrained decoder using PyTorch and Hugging Face Transformers. Start with a model around 0.5B parameters, such as Qwen2.5-0.5B, after checking its model card, license, and installed-library compatibility. Compare generation with and without KV caching using greedy decoding and fixed inputs. Check logits within a declared numerical tolerance before drawing performance conclusions. Implementing an entire transformer from scratch is optional.

Derive rough weight, activation, and KV memory costs. For a conventional decoder, an approximate per-request KV footprint is:

`2 × layers × cached_tokens × KV_heads × head_dimension × bytes_per_element`

The factor of two represents keys and values. This excludes allocator overhead and architecture-specific differences; grouped-query attention uses KV heads, not the total query-head count. Explain how these costs change with context length, concurrency, and precision.

**Exit evidence:** a correctness-checked inference example and a note predicting which workload shapes are compute- or bandwidth-limited. Prefill is often more compute-intensive and low-batch decode often bandwidth-limited; verify this instead of treating it as universal.

**Progress check:** do not compress this foundation just because the systems concepts feel familiar. If tensor shapes or cached decoding are still unclear at month end, use the first two weeks of month 2 to finish them and shorten the kernel exercise.

### Month 2 — GPU execution and profiling

Learn the CUDA execution hierarchy, coalescing, registers/shared memory, occupancy, reductions, streams, events, and host/device synchronization. Understand arithmetic intensity and a simple roofline model. Learn to read a timeline before trying to optimize a kernel.

Start with vector addition to learn launches and indexing, then implement one small operator, such as RMSNorm or softmax, in Triton; use a small CUDA exercise to understand the lower-level execution model if needed. Compare to PyTorch across several shapes and verify numerical correctness. Profile end-to-end execution with PyTorch Profiler; use Nsight Systems/Compute when available and permitted by the host. Read FlashAttention for its memory-traffic idea after learning basic attention; implementing it is outside the core plan.

**Exit evidence:** one correctness-checked kernel experiment and a trace explaining a bottleneck. Report warm-up, timing method, transfer inclusion, and shape-specific results. Be able to explain why improving one kernel may barely affect request latency.

### Month 3 — Establish a serving baseline

Choose **vLLM as the primary runtime** and pin a working release or commit compatible with the RTX 3060 and its installed driver. Run a small supported model and trace the path through request handling, scheduling, KV allocation, model execution, and response streaming. Begin with the same approximately 0.5B model used in month 1; try approximately 1–1.5B parameters once measured memory headroom permits. Set a modest context limit and low concurrency explicitly instead of inheriting a model's maximum context. Use SGLang later as a comparison only if it helps answer a specific question.

Measure isolated prefill/decode and concurrent serving. Sweep a small set of prompt lengths, generated lengths, and offered loads. Record time to first token (TTFT), inter-token latency (ITL), end-to-end latency, throughput, memory usage, and request failures. Identify saturation and the difference between queueing time and execution time.

**Exit evidence:** a reproducible baseline with latency distributions and throughput curves, plus a source-code map tied to the pinned runtime revision. Another person should be able to rerun it from the instructions.

## Months 4–6: turn systems knowledge into runtime expertise

### Month 4 — Batching and scheduling

Study iteration-level/continuous batching, chunked prefill, admission control, and scheduling under latency objectives. Explain why long prefills can interfere with ongoing decodes and why increasing throughput can worsen tail latency.

Compare supported runtime configurations on mixed short/long requests. Build a small discrete-event simulator to reason about FIFO, length-aware heuristics, and fairness. Explicitly distinguish known prompt lengths from output lengths that are unknown when a request arrives; use perfect output-length knowledge only as a labeled oracle baseline.

**Exit evidence:** a measured tradeoff between throughput and latency, a simulator checked against at least some real measurements, and a concrete scheduling question worth pursuing.

### Month 5 — KV-cache management

Study PagedAttention, allocation/block size, prefix sharing, fragmentation, eviction, recomputation, and preemption. Follow the runtime's cache lifetime and synchronization rules. Learn how cache pressure feeds back into scheduling.

Measure workloads with different prefix-sharing rates and context lengths. Vary supported cache or batching settings and explain observed stalls, recomputation, or rejected requests. Keep model, token workload, and available memory comparable across configurations.

**Exit evidence:** a cache-pressure experiment that connects memory behavior to user-visible latency. Identify one failure mode that your capstone might address.

### Month 6 — Parallel and distributed inference

Learn tensor, pipeline, and data parallelism, collective communication, topology, and prefill/decode disaggregation. Cover expert parallelism and MoE routing at a conceptual level; deep implementation work is optional.

Estimate communication volume and compute time before running distributed experiments. With two suitable GPUs, compare a model/configuration across supported parallel layouts and explain the communication cost. Without them, analyze runtime traces and build a calibrated analytical model; label distributed performance predictions as unvalidated.

For the RDMA extension, first verify NIC/GPU support, topology, permissions, and available transport. A fast network alone does not establish GPUDirect RDMA support. Compare KV transfer cost with the benefit of separating prefill and decode, including queueing and transfer overlap.

**Exit evidence:** a placement/communication design note supported by measurements where hardware allows. A multi-node deployment is optional, not a prerequisite for the main capstone.

## Months 7–9: one substantial capstone

**Working question:** Can a scheduling policy that uses queue state and KV-memory pressure improve the rate of requests meeting latency objectives under mixed prompt lengths and bursty arrivals?

This is a study direction, not a novelty claim. Existing runtimes already implement sophisticated policies; inspect the pinned baseline and relevant literature before claiming an improvement.

### Month 7 — Define the experiment

Write a two-page proposal with a falsifiable hypothesis. Choose one mechanism, such as adapting the prefill token budget using decode backlog and KV occupancy. Explain what information is available online and the overhead of collecting it.

Define per-class TTFT and decoding-latency objectives. Define **goodput** precisely, for example successful requests per second meeting both objectives. Report rejected, cancelled, and timed-out requests against offered load so that aggressive admission control cannot masquerade as improved service.

Use the stock runtime and a tuned supported configuration as baselines. Choose a compact workload matrix spanning prompt length, output length, arrival process, and cache reuse. Hold out some workloads for evaluation. Select a second model or model size if resources permit.

**Exit evidence:** proposal, baseline results, success/failure criteria, and an experiment budget. Freeze the core scope here.

### Month 8 — Implement and evaluate

Implement the smallest runtime change that tests the hypothesis. Check request completion, output integrity, streaming, cancellation, cache ownership, and starvation behavior. Measure policy overhead and compare repeated runs under the same workload and memory budget.

Add ablations that isolate the mechanism: fixed versus adaptive token budget, with and without memory-pressure feedback, for example. Include workloads where the policy should not help and overload cases where its behavior matters.

**Exit evidence:** a working prototype, correctness evidence, and reproducible comparisons. A well-explained negative result is useful; do not switch projects just to find a favorable chart.

### Month 9 — Explain and communicate the result

Investigate anomalies, test sensitivity to workload mix, and bound the conclusions. Tie performance changes to queueing, memory, and execution behavior. Write a research-style report with the problem, mechanism, baselines, experimental method, limitations, and results.

Turn a self-contained improvement or bug fix into an upstream contribution when appropriate. A useful benchmark, documentation correction grounded in investigation, or regression test can also be valuable. You control submission quality; upstream acceptance is not guaranteed.

**Exit evidence:** a reproducible project, a concise technical talk, and one contribution ready for review. Broader publication work is optional and depends on novelty and time.

## Months 10–12: consolidate for graduation and hiring

### Month 10 — Learn adjacent techniques selectively

Survey quantization, speculative decoding, prefix caching, CUDA graphs, and MoE serving. Pick one to evaluate in enough depth to explain its workload-dependent benefits and costs. Approximate techniques need a task-quality check as well as speed measurements; speculative decoding needs acceptance-rate and draft-model overhead analysis.

**Exit evidence:** a short decision memo: when the technique helps, when it hurts, and how it interacts with your scheduler. Do not start another large project.

### Month 11 — Make the work legible to employers

Package the capstone with a clear README, pinned environment, small runnable example, workload description, result-generation commands, and limitations. Keep a 10-minute systems explanation and a deeper technical presentation ready.

Practice designing a multi-tenant inference service: capacity planning, routing, batching, cache placement, admission control, fairness, observability, and failure behavior. Practice diagnosing a trace and estimating whether a proposed optimization is compute-, memory-, or network-limited. Retain regular systems coding practice if your target roles require interviews.

Map your evidence to roles: inference/serving engineer, ML systems engineer, performance engineer, or inference-focused research engineer. By months 6–7, already review relevant job descriptions and adjust emphasis; do not wait until the final month to discover requirements or begin applications.

### Month 12 — Graduation buffer and final synthesis

Reserve this month for dissertation demands, interview preparation, and unfinished validation. Reproduce the main result from a clean documented setup when resources allow. Write down which conclusions generalize and which depend on hardware, model, runtime version, or workload.

**Exit evidence:** one polished project you can defend deeply, an honest account of limitations, and a clear connection from your PhD work to inference infrastructure.

## First four weeks

| Week | Work | Concrete checkpoint |
| --- | --- | --- |
| 1 | Inventory RTX 3060 VRAM and driver with `nvidia-smi`; set up an isolated Python environment and a compatible PyTorch build; practice tensor shapes, broadcasting, matmul, and CPU/GPU placement | Run and explain a matrix multiplication on the GPU, including a correctness comparison with CPU |
| 2 | Learn tokenization, causal masking, attention, and transformer blocks; run a small pretrained decoder | Annotated tensor-shape walkthrough and a generated response |
| 3 | Write a generation loop and compare cached/uncached decoding; calculate weight/KV memory | Correctness comparison and a memory estimate |
| 4 | Measure prefill and decode at a few short lengths; distinguish warm-up from steady state | First experiment report with commands, results, discrepancies, and limitations |

Keep the first month's code small enough that you can explain every tensor operation. Ask why a measurement behaves as it does before optimizing it.

## Reading sequence

Read to answer the current month's question. For the first six weeks, prioritize tutorials and working code; do not impose a paper quota. Afterwards, aim for one core paper every one or two weeks. For each systems paper, record its bottleneck, mechanism, assumptions, evaluation, and one condition under which its advantage might disappear.

| Stage | Resource | What to extract |
| --- | --- | --- |
| Model basics | [PyTorch basics](https://pytorch.org/tutorials/beginner/basics/intro.html), selected [Hugging Face LLM course](https://huggingface.co/learn/llm-course/chapter1/1) chapters | Tensor execution, tokenization, transformer inference; skip a full training curriculum initially |
| GPU fundamentals | [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/), [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) | Memory hierarchy, execution model, synchronization, kernel measurement |
| Attention | [FlashAttention](https://arxiv.org/abs/2205.14135) | IO-aware algorithm design and the difference between mathematical and memory complexity |
| Serving and memory | [Orca](https://www.usenix.org/conference/osdi22/presentation/yu), [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | Iteration-level scheduling, KV allocation, workload assumptions |
| Runtime implementation | [vLLM documentation](https://docs.vllm.ai/en/latest/), selected [SGLang documentation](https://docs.sglang.ai/) | Source paths and behavior for a pinned release; papers are not complete descriptions of current code |
| Scheduling | [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Chunked prefill and the throughput/latency tradeoff |
| Disaggregation | [DistServe](https://arxiv.org/abs/2401.09647), [Splitwise](https://arxiv.org/abs/2311.18677) | Conditions under which phase separation pays for communication and placement costs |
| Prefix reuse | [SGLang](https://arxiv.org/abs/2312.07104) | Reuse and scheduling for structured/repeated prompts |
| Measurement | [MLPerf Inference](https://arxiv.org/abs/1911.02549) and the chosen runtime's benchmark documentation | Scenario definitions, latency constraints, reproducibility; you need not run the full benchmark |

These are foundational starting points, not a claim to cover the newest work. Before the capstone, review recent systems proceedings and runtime changes relevant to your specific hypothesis. External resource pages could not be fetched from the onboarding environment because its network policy returned HTTP 403; their live contents were not verified during drafting.

## Experimental standards

- Record hardware, available memory, driver/toolchain, package/runtime revision, model/tokenizer revision, dtype, parallelism, scheduler settings, seeds, and workload generation. Respect model licensing/access requirements and keep credentials out of Git.
- Separate cold start, warm-up, and steady state. Use GPU events or appropriate synchronization for kernel timing; host timing of asynchronous launches alone is misleading. Measure user-visible serving latency at the client.
- Use open-loop offered-load experiments to expose queueing, plus closed-loop concurrency sweeps when useful. Ensure the client is not the bottleneck. State arrival distributions, run duration, sample counts, and treatment of unfinished requests.
- Define TTFT and ITL boundaries explicitly. Aggregate tokens/second can hide long waits and uneven service. Report percentiles and per-class results; do not present a p99 from too few observations as reliable.
- Repeat measurements and show variability. Keep model quality, generated lengths, warm-cache conditions, and resource budgets comparable. Label synthetic workloads, simulated results, and hardware measurements separately.
- Separate correctness from performance. Check output behavior, request lifecycle, memory safety/ownership, and fairness before claiming a scheduler improvement.

## Hardware and time adaptations

### Your RTX 3060: local first

Confirm the actual VRAM rather than assuming 12 GB: RTX 3060 desktop and laptop configurations differ. Begin with a roughly 0.5B model in FP16, a context limit around 1,024–2,048 tokens, and one concurrent request. Increase one dimension at a time after measuring headroom. Treat these as starting configurations to validate, not guaranteed runtime memory requirements.

FP16 weights alone use approximately 1 GB for 0.5B parameters, 3 GB for 1.5B, and 14 GB for 7B, in decimal units. KV cache, activations, temporary buffers, runtime allocations, and display use need additional space. A 7B model in FP16 is therefore not a sensible starting target on your card. Quantization can reduce weight memory, but adds implementation and quality questions; defer it until you understand the baseline.

Prefer a native Linux setup for serving-runtime and profiling compatibility. Check the selected runtime's current OS, GPU, Python, PyTorch/CUDA, and driver requirements before installation; CUDA toolkit and driver versions are distinct. Use separate Python environments for introductory PyTorch exercises and the serving runtime so their package constraints do not conflict. Some profiling capabilities require host permissions, so use available tools and record limitations.

One RTX 3060 can support meaningful studies of batching, prefill/decode interference, and memory pressure with small models. Constrain context/concurrency and choose workloads that expose these effects. Conclusions about this consumer GPU and model do not establish results for datacenter GPUs or large models.

### When cloud rental earns its cost

- **Months 1–5:** use the local GPU. Rent only if a diagnosed compatibility or capacity limitation blocks a required experiment.
- **Month 6:** optionally rent a suitable two-GPU machine for one parallelism experiment. For RDMA, verify that the provider actually exposes the required multi-node fabric and permissions; two GPUs in one machine do not test RDMA.
- **Months 8–9:** consider one larger-memory/datacenter GPU to test whether the capstone result persists on a second platform. First prepare the environment specification, scripts, input workloads, and result collection locally.
- Before each rental, define the question, experiment matrix, storage/transfer requirements, cost ceiling, and shutdown procedure. Check on-demand pricing and quota at booking time; do not leave a rented machine running for reading or analysis. Save results before stopping ephemeral instances.

| Available resources | Practical path | Evidence you should not claim |
| --- | --- | --- |
| CPU only | Model semantics, tiny cached decoding, runtime reading, simulator, analytical memory/communication models | GPU kernel speedups or production serving performance |
| Occasional rented/shared GPU | Develop locally; batch a preplanned experiment matrix into short sessions; save all metadata/results | Continuous access or multi-node behavior you did not measure |
| One GPU | Full core plan with a model that leaves sufficient KV headroom; reduce model size before sacrificing experimental validity | Multi-GPU scaling or RDMA benefits |
| Multiple GPUs / RDMA cluster | Core plan plus measured collectives, topology, or KV-transfer extension | Generalization beyond tested hardware without supporting evidence |

With only 5–6 hours per week, preserve months 1–5 and the capstone, make month 6 conceptual, and skip the month-10 implementation. With 12+ hours, deepen profiling and reproducibility before adding projects. If GPU access remains unavailable, finish the CPU work and simulator, but treat measured serving-performance readiness as pending access.

## How to use this repository

Create directories when there is real content for them:

```text
notes/          # Concepts, paper critiques, source-code maps
experiments/    # Small runnable studies, each with commands and dependencies
benchmarks/     # Workloads, load generation, measurement and plotting scripts
reports/        # Monthly conclusions and compact result tables
capstone/       # Proposal, implementation pointers, evaluation and final report
```

For each experiment, document **question → prediction → configuration → commands → result → explanation → limitations**. Store small anonymized result files where useful; keep model weights, credentials, large traces, and caches out of Git. Pin dependencies per executable experiment once its requirements are known, rather than installing a speculative stack for the entire year.

At the end of each month, decide whether to move forward, repeat one unresolved experiment, or cut an optional topic. The completion criterion is being able to explain and reproduce the result, not merely having read the material.
