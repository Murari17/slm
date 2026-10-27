# A 152M-parameter GPT trained from scratch

A small GPT-style language model built end to end: data pipeline, model, multi-round training on a single Colab GPU, post-training quantization to ~104 MB, and a fast local runner for a MacBook.

It is a learning project. The model writes fluent English and plausible-looking code, but it gets facts, arithmetic and program logic wrong, which is expected at this size (see [Limitations](#limitations)).

## Model

| | |
|---|---|
| Parameters | ~151.9M (embedding shared with the output layer) |
| Layers / heads / width | 16 / 12 / 768 |
| Context length | 1024 tokens |
| Tokenizer | GPT-2 BPE via `tiktoken` (vocab padded to 50,304) |
| Architecture | Decoder-only transformer with RoPE positions, RMSNorm, GELU MLP (4x), tied embeddings, no biases, causal flash attention (PyTorch SDPA) |

Parameter count is fixed by this config. Training on more tokens changes the weights, not the model size.

## Training

Training ran in stages, each resuming from the previous checkpoint.

**Stage 1: base run (web text)**
- **Data:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`), about 3B tokens tokenized into a single `uint16` file. The last 10M tokens are held out for validation.
- **Setup:** bf16 autocast, `torch.compile`, AdamW (betas 0.9/0.95, weight decay 0.1), peak LR 6e-4 with 200 warmup steps and cosine decay to 6e-5, gradient clipping at 1.0, 524,288 tokens per step, 5,702 steps.
- **Hardware:** one Google Colab G4 GPU (96 GB), about 3.8 hours.

**Stage 2: continued rounds (web + code + math)**
- Each round trains on **1B new tokens**, never reusing documents from earlier rounds, then saves a checkpoint and updates `state.json` on Drive.
- **Mix (by tokens, exact):** 50% web ([FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)), 35% code ([codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean)), 15% math ([FineMath-4+](https://huggingface.co/datasets/HuggingFaceTB/finemath)).
- **Learning rate:** peak LR decays by 2/3 each round with a floor of 1e-4, cosine to 10% of peak, 50 warmup steps.
- **Validation:** held-out sets for web, code and math, reported separately every 250 steps.
- **Resumable:** the cell saves to Drive every 250 steps. Re-run the same cell after a disconnect and it continues from the last checkpoint.
- **Throughput:** about 228k tokens/s on a G4 (micro-batch 64 x 8 accumulation, 524,288 tokens per step), roughly 73 minutes per 1B-token round. The model is compute-bound at this point, so larger batches don't help further.

### Results

| Checkpoint | Web val loss | Code val loss | Math val loss |
|---|---|---|---|
| Stage 1 final (fp32) | 3.17 | n/a | n/a |
| Before round 5 (4 rounds done) | 3.15 | 1.08 | 2.30 |

Code and math sets are held-out documents from the start of each stream and are never trained on.

## Quantization

The trained weights are compressed after training with plain round-to-nearest quantization (no retraining) and measured on the web validation set. The table was measured on the stage 1 model:

| Format | Val loss | vs fp32 | Approx. size |
|---|---|---|---|
| fp32 original | 3.1740 | n/a | ~600 MB |
| int8 per-row | 3.1743 | +0.0003 | ~150 MB |
| int6 per-row | 3.1788 | +0.0048 | ~114 MB |
| **blocks int5 (groups of 64) + embedding int6** | **3.1839** | **+0.0099** | **~104 MB** |
| blocks int4 (groups of 64) + embedding int8 | 3.2140 | +0.040 | ~100 MB |
| int3 per-row | 4.0662 | +0.89 | ~60 MB |

The exported file uses the int5 g64 + int6 embedding recipe: five-bit and six-bit values are packed into bytes with one 16-bit scale per group. Weights are unpacked to full precision when the model runs, so this reduces file size, not memory use or speed. Small models tolerate low-bit quantization worse than large ones, which is why int4 costs noticeably more here.

## Files

| File | Purpose |
|---|---|
| `train_gpt_150m.ipynb` | Colab notebook: data download and tokenization, model, stage 1 training, quantization experiments, int5 export, and the early continued-training steps |
| `continue_training.py` | The "universal" continue-training cell for stage 2: one cell, one round per run, mixed web/code/math data, auto-resume, int5 export after each round |
| `run_gpt150m.py` | Standalone runner: loads the packed file and generates text with a KV cache (Apple GPU via MPS, CUDA, or CPU) |

The trained weights (`gpt150m_int5g64.npz`, ~104 MB) and the tokenized data are not stored in this repo because of GitHub's file size limit. Download the weights from: _add your link here (Hugging Face Hub or a GitHub Release)_.

## Run it locally

```bash
pip install torch numpy tiktoken
python run_gpt150m.py gpt150m_int5g64.npz "The Roman Empire was founded when"
```

Useful options: `--tokens 300`, `--temp 0.5`, `--top_k 50`, `--rep 1.1` (repetition penalty). Leave out the prompt for interactive mode. The first run needs internet once so `tiktoken` can download its tokenizer file.

To confirm the KV cache gives the same output as a full forward pass:

```bash
python run_gpt150m.py gpt150m_int5g64.npz --check
```

## Reproduce the training

Checkpoints, data and `state.json` live under `My Drive/gpt150m/`, so a Google Drive mount is needed.

**Stage 1**
1. Open `train_gpt_150m.ipynb` in Google Colab and select a GPU runtime.
2. Run the setup, tokenization, model and training cells in order. Tokenization runs once (set `TARGET_TOKENS` lower for a quick test).

**Stage 2**
1. Paste `continue_training.py` into a Colab cell and run it. Each run is one 1B-token round.
2. Set `MICRO_BS, GRAD_ACCUM` to suit your GPU. Their product must stay 512 (524,288 tokens per step):

| GPU | `MICRO_BS, GRAD_ACCUM` |
|---|---|
| G4 (96 GB) | 64, 8 |
| A100 (40 GB) | 32, 16 |
| L4 (24 GB) | 16, 32 |

3. Re-run the same cell for the next round. After a disconnect, run it again and it resumes.

Speed numbers above are from the G4; other GPUs will be slower.

## Evaluation

Beyond validation loss, the model is checked with a fixed set of 20 prompts (web, code, math, coherence) at temperature 0.7, top-k 50 and a fixed seed, so rounds can be compared directly. Each output is scored for repetition (share of repeated 4-grams) and, for code, whether it parses.

Observations from the latest check:
- **Fluency:** grammatical, on-topic text for every prompt.
- **Code:** correct structure (indentation, docstrings, imports) but wrong logic, for example a `fibonacci` that returns `2*(n-1)`.
- **Math:** none of 4 arithmetic and calculus prompts answered correctly.
- **Repetition:** 4 of 20 prompts looped without a repetition penalty.

Standard benchmarks (HellaSwag, ARC, PIQA) have not been run yet.

## Limitations

- **Facts are unreliable.** The model writes plausible text but mixes up names, dates and events.
- **Code and math are weak.** The model has seen code and math data, but at 152M parameters it imitates the format rather than reasoning. Expect syntactically plausible code that does not do what its name says.
- **It's a base model.** It continues text rather than following instructions, so it works best with an article-style prompt, not a question.
- **Repetition** can appear in longer outputs. The runner includes a repetition penalty to reduce it.
- **Short context:** 1024 tokens. When the cache fills, the runner keeps the newest half of the text.
- **Tokenizer:** GPT-2 BPE is not tuned for code (whitespace-heavy indentation costs many tokens).

## Possible next steps

- Run standard benchmarks and track them per round
- Instruction fine-tuning (SFT) on chat data
- A larger v2 model with a code-friendly tokenizer and a longer context
- Smarter quantization (GPTQ or AWQ-style) to make int4 usable
