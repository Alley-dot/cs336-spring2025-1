# cs336-spring2025-1

Stanford CS336 Spring 2025 assignments collected in one place for personal study.

## Assignment overview

This monorepo collects five CS336 assignments. The summary below is meant to answer two questions quickly:

1. What does each assignment actually cover?
2. Is the assignment README enough to serve as a local starting point?

The content summaries are based on each assignment's `README.md`, test files, and adapter interfaces.

| Assignment | Main topics | README verdict |
| --- | --- | --- |
| `assignment1-basics` | BPE tokenizer training and inference, core neural-network ops, Transformer LM building blocks, training utilities, optimizer and checkpointing | Good starting point |
| `assignment2-systems` | FlashAttention, Triton kernels, DDP, FSDP, sharded optimizer, distributed training systems work | Mostly complete starting point |
| `assignment3-scaling` | Scaling-law experiments around a small Transformer LM, model loading/generation, and compute or parameter tradeoff analysis | Incomplete starting point |
| `assignment4-data` | Web text extraction, language ID, PII masking, toxicity and quality filtering, Gopher rules, exact and MinHash deduplication | Partial starting point |
| `assignment5-alignment` | SFT data prep, response scoring, entropy and log-probs, policy-gradient training, GRPO, DPO, evaluation parsing, optional RLHF/safety work | Mostly complete starting point |

## Assignment details

### `assignment1-basics`

- Implements the foundations needed for a small language model stack.
- Covers tokenization end to end: training a BPE tokenizer, building a tokenizer from vocab and merges, matching expected tokenization behavior, and handling special tokens.
- Covers core model components such as linear layers, embeddings, SiLU, SwiGLU, RMSNorm, RoPE, scaled dot-product attention, multi-head self-attention, Transformer blocks, and a full Transformer language model.
- Covers training utilities including batch sampling, softmax, cross-entropy, gradient clipping, AdamW, cosine learning-rate scheduling, and checkpoint save/load.
- The README is the clearest in the repo: it explains environment setup, tests, adapter wiring, and dataset download.

### `assignment2-systems`

- Focuses on systems and efficiency work for Transformer training.
- Includes implementing FlashAttention in pure PyTorch autograd first, then again with Triton kernels.
- Includes distributed training wrappers for DDP and FSDP, including gradient synchronization and parameter gathering behavior.
- Includes optimizer-state sharding via a sharded optimizer.
- The README explains the dependency on the provided `cs336-basics` package and the intended repo layout, so it is usable as a start, but it gives less day-to-day workflow guidance than Assignment 1.

### `assignment3-scaling`

- Centers on scaling-law style experimentation rather than a large test-driven implementation scaffold.
- The repo currently exposes a small Transformer LM implementation with forward, generation, and checkpoint-loading utilities.
- The presence of `data/isoflops_curves.json` and the assignment title suggest the main work is analyzing model, data, and compute tradeoffs under scaling constraints.
- Compared with the other assignments, the README is very thin: it mentions `uv`, but not the intended workflow, experiments, or expected outputs.
- This is the weakest standalone starting point if you want to use the monorepo without prior course context.

### `assignment4-data`

- Focuses on building a data-processing and data-quality pipeline for language-model training.
- Includes HTML text extraction, language identification, masking of emails, phone numbers, and IP addresses, plus content filtering for NSFW or toxic speech.
- Includes document-quality classification and rule-based Gopher-style quality filtering.
- Includes deduplication with both exact line deduplication and fuzzy near-duplicate detection using MinHash.
- The README explains the intended package structure, but it does not clearly document the local workflow for assets, tests, or implementation order.

### `assignment5-alignment`

- Focuses on post-training and preference optimization for language models.
- Covers SFT-oriented preprocessing such as tokenizing prompt-response pairs, masking response tokens, packing SFT datasets, and iterating over batches.
- Covers log-prob and entropy computation, masked reductions, SFT microbatch loss computation, and policy-gradient style losses.
- Covers GRPO and DPO specifically, including reward normalization, clipped objectives, and per-instance DPO loss.
- Covers simple evaluation parsing utilities for MMLU and GSM8K outputs.
- The optional supplement extends this assignment into safety alignment and RLHF-style work.
- The README is usable for local startup because it explains dependency installation and test execution, but it is still lighter on structure than Assignment 1.

## Takeaways

- `assignment1-basics` is the most self-contained starting point and the best documented assignment in this monorepo.
- `assignment2-systems` and `assignment5-alignment` are both substantial and well-scoped, but their READMEs assume more course familiarity.
- `assignment4-data` has a clear technical scope, but its README would benefit from more operational detail.
- `assignment3-scaling` is the least documented and would need the most README expansion to work well as an independent personal-study project.

## UV environment tutorial

This repository works well as a set of isolated `uv` projects. You only need to install `uv` once, but you should treat each assignment directory as its own environment.

### Why this works

- Each assignment directory has its own `pyproject.toml` and lockfile.
- Python version requirements differ across assignments, so a single shared environment is not a good fit.
- Some assignments depend on large or specialized packages such as `flash-attn`, `vllm`, or local package paths like `cs336-basics`.

### Recommended workflow

1. Install `uv` once on your machine.
2. Enter the assignment directory you want to work on.
3. Create or refresh that assignment's environment with `uv sync`.
4. Run everything through `uv run ...` so commands always use the correct environment.
5. Switch assignments by changing directories, not by manually reusing one shared virtualenv.

### General command pattern

```sh
cd assignment1-basics
uv sync
uv run pytest
```

If you need the interpreter path for an editor or notebook setup:

```sh
uv run which python
```

## Per-assignment workflow

The tutorials below summarize, for each assignment:

1. how to start the environment,
2. how to do the assignment work, and
3. how to verify correctness.

When the original assignment depends on Stanford-only infrastructure, that is called out explicitly.

### `assignment1-basics`

#### Start the environment

```sh
cd assignment1-basics
uv sync
uv run pytest
```

- The assignment README states that `uv run <python_file_path>` is the standard way to run code.
- The initial test run is expected to fail with `NotImplementedError` until you connect your implementation to `tests/adapters.py`.
- The README also provides the data download commands for TinyStories and OpenWebText.

#### How to do the assignment

- Write substantive code in `assignment1-basics/cs336_basics/`.
- Treat `assignment1-basics/tests/adapters.py` as glue code only: the handout explicitly says adapters should just call into your implementation.
- Follow the handout's progression:
  - build a byte-level BPE tokenizer,
  - implement tokenizer encode/decode,
  - implement Transformer LM components from scratch,
  - implement softmax, cross-entropy, AdamW, cosine LR schedule, gradient clipping, batching, and checkpointing,
  - train on TinyStories, then run larger experiments on OpenWebText.
- Download the datasets before the training parts of the assignment:

```sh
cd assignment1-basics
mkdir -p data
cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```

#### How to verify correctness

- For full unit coverage:

```sh
cd assignment1-basics
uv run pytest
```

- For targeted checks while iterating, the test suite is naturally divided by component:
  - `tests/test_train_bpe.py`
  - `tests/test_tokenizer.py`
  - `tests/test_model.py`
  - `tests/test_nn_utils.py`
  - `tests/test_optimizer.py`
  - `tests/test_serialization.py`
  - `tests/test_data.py`
- The handout's end-to-end correctness signal is stronger than unit tests alone: after passing tests, you should be able to train a tokenizer, encode the datasets, train a small LM, and generate text with reasonable fluency on TinyStories.

### `assignment2-systems`

#### Start the environment

```sh
cd assignment2-systems
uv sync
uv run python
```

- The README recommends verifying that the bundled `cs336-basics` package is importable from the assignment environment.
- If you want to swap in your own Assignment 1 code, update the local path dependency in `assignment2-systems/pyproject.toml`.

#### How to do the assignment

- Write your systems code under `assignment2-systems/cs336_systems/`.
- The handout says this assignment covers:
  - benchmarking and profiling,
  - activation checkpointing,
  - FlashAttention 2 in both PyTorch and Triton,
  - DDP,
  - optimizer state sharding,
  - FSDP,
  - parallelism analysis.
- Use `cs336-basics/` as the reference Assignment 1 implementation for profiling and systems work.
- The tests connect through `assignment2-systems/tests/adapters.py`, just like in Assignment 1.

#### How to verify correctness

- Run the full test suite:

```sh
cd assignment2-systems
uv run pytest -v ./tests
```

- Or iterate by subsystem:
  - `tests/test_attention.py` for FlashAttention
  - `tests/test_ddp.py` for DDP
  - `tests/test_fsdp.py` for FSDP
  - `tests/test_sharded_optimizer.py` for optimizer sharding
- Beyond unit tests, the handout expects benchmark and profiling scripts that measure:
  - forward/backward/optimizer timings,
  - Nsight traces,
  - memory profiles,
  - communication overhead in distributed training.
- So the practical correctness bar is: pass tests and produce reasonable profiling/benchmarking outputs for the requested model scales.

### `assignment3-scaling`

#### Start the environment

```sh
cd assignment3-scaling
uv sync
uv run which python
```

- The README is intentionally minimal and mostly says to use `uv run <command>`.
- If you need extra packages for your own scripts or notebooks, the README explicitly allows `uv add <package>`.

#### How to do the assignment

- This assignment is more of an experimentation and analysis project than a test-driven implementation scaffold.
- The handout says the workflow is:
  - reproduce IsoFLOPs-style scaling law fitting using `data/isoflops_curves.json`,
  - fit scaling laws for optimal model size and dataset size as functions of compute,
  - query the provided training API to explore configurations under a fixed FLOPs budget,
  - predict the compute-optimal model and hyperparameters for a `1e19` FLOPs budget.
- The provided `cs336_scaling/model.py` is a reference model implementation for this assignment's setting.
- This assignment depends on course infrastructure more than the others:
  - the training API requires Stanford network access or VPN,
  - the API key is t·ied to the original course setup,
  - the main outputs are scripts, plots, and a writeup rather than unit-tested package code.

#### How to verify correctness

- There is no provided unit test suite in this repo for Assignment 3.
- According to the handout, correctness is verified by successfully doing the following:
  - reproducing the IsoFLOPs fitting procedure on `data/isoflops_curves.json`,
  - generating plots for extrapolated optimal model size and dataset size,
  - querying the training API without exceeding the `2e18` FLOPs fitting budget,
  - reporting a predicted optimal model size, hyperparameters, and predicted loss for `1e19` FLOPs.
- In practice, your self-checklist should be:
  - your scripts run under `uv run ...`,
  - your API queries work,
  - your plots and fitted curves are reproducible,
  - your budget accounting is correct.

### `assignment4-data`

#### Start the environment

```sh
cd assignment4-data
uv sync
uv run pytest -v ./tests
```

- The outer assignment environment contains the data-processing code and tests.
- The repo also includes `assignment4-data/cs336-basics/`, which is the training code used later once you have produced filtered data.

#### How to do the assignment

- Write your data-pipeline code in `assignment4-data/cs336_data/`.
- The handout organizes the work in roughly this order:
  - inspect Common Crawl WARC/WET examples,
  - convert HTML to text,
  - perform language identification,
  - mask PII such as emails, phone numbers, and IPs,
  - filter harmful content,
  - apply rule-based quality filters,
  - train a quality classifier,
  - run exact and MinHash-based deduplication,
  - filter a large set of WET files,
  - tokenize the filtered data with GPT-2,
  - train a GPT-2-small-shaped model on the filtered corpus.
- For local asset bootstrap, this repo includes:

```sh
cd assignment4-data
./get_assets.sh
```

- The handout also relies on course-local resources for some experiments, including:
  - Common Crawl sample files on the Together cluster,
  - fastText language ID and quality resources,
  - a 5,000-file WET subset on the cluster,
  - Paloma validation data on the cluster.
- If you are using this repo outside the course environment, you will need to replace those with your own local copies.

#### How to verify correctness

- Primitive-by-primitive verification comes from the tests:
  - `tests/test_extract.py`
  - `tests/test_langid.py`
  - `tests/test_pii.py`
  - `tests/test_toxicity.py`
  - `tests/test_quality.py`
  - `tests/test_deduplication.py`
- For a full local check:

```sh
cd assignment4-data
uv run pytest -v ./tests
```

- The end-to-end verification path from the handout is:
  1. run your filtering pipeline on WET files,
  2. tokenize the resulting corpus with GPT-2,
  3. edit the training config in `assignment4-data/cs336-basics/configs/experiment/your_data.yaml`,
  4. train the model.
- The handout's training command is:

```sh
cd assignment4-data/cs336-basics
uv run torchrun --standalone --nproc_per_node=2 scripts/train.py --config-name=experiment/your_data
```

- If you enable checkpointing, the handout also suggests sampling from a saved model:

```sh
cd assignment4-data/cs336-basics
uv run python scripts/generate_with_gpt2_tok.py --model_path output/your_data/step_N
```

- So the final correctness signal is not just passing the unit tests, but also successfully producing filtered data, tokenizing it, and training the provided GPT-2-small setup on it.

### `assignment5-alignment`

#### Start the environment

```sh
cd assignment5-alignment
uv sync --no-install-package flash-attn
uv sync
uv run pytest
```

- The README explicitly calls out the two-step install because `flash-attn` is finicky.
- This assignment has the heaviest environment in the repo, so it especially benefits from staying isolated from the others.

#### How to do the assignment

- Write your code in `assignment5-alignment/cs336_alignment/`.
- The handout says the mandatory assignment covers:
  - a zero-shot baseline on MATH,
  - supervised finetuning on reasoning traces,
  - expert iteration,
  - GRPO.
- The optional part extends into preference alignment and RLHF-style work.
- The handout expects you to use:
  - Qwen 2.5 Math 1.5B as the base model,
  - MATH train/validation data from the course paths,
  - vLLM for generation,
  - HuggingFace models/tokenizers for forward passes and training logic.
- Important course-infra dependencies called out in the docs:
  - MATH is provided on the cluster at `/data/a5-alignment/MATH`,
  - pre-downloaded models are provided under `/data/a5-alignment/models/...`,
  - the handout recommends two GPUs for experiments, usually one for the policy and one for vLLM.

#### How to verify correctness

- The handout explicitly says that the mandatory tests are the SFT and GRPO ones.
- Run those first:

```sh
cd assignment5-alignment
uv run pytest tests/test_sft.py tests/test_grpo.py
```

- Additional tests in this repo correspond to helper functionality or non-mandatory parts:
  - `tests/test_data.py`
  - `tests/test_metrics.py`
  - `tests/test_dpo.py`
- For a broader local check, run:

```sh
cd assignment5-alignment
uv run pytest
```

- The end-to-end correctness path from the handout is experimental rather than purely unit-test based:
  - reproduce the zero-shot MATH baseline,
  - run SFT and see validation accuracy improve,
  - run expert iteration and GRPO,
  - log rewards, entropy, rollout behavior, and validation curves.
- For this assignment, the strongest correctness signal is therefore a combination of:
  - passing the mandatory tests,
  - successfully training and evaluating on MATH,
  - and observing sensible validation metrics and rollouts.

## Practical notes

- If you are studying this repo outside the original course infrastructure, `assignment3-scaling`, `assignment4-data`, and `assignment5-alignment` depend the most on Stanford-only datasets, APIs, or cluster paths.
- `assignment1-basics` is the easiest assignment to run fully as a standalone personal-study project.
- `assignment2-systems` is also a good standalone project if you have the appropriate GPU setup for Triton and distributed experiments.
