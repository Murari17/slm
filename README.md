# A 152M-parameter GPT trained from scratch

A small GPT-style language model built end to end: data pipeline, model, training on a single Colab GPU, post-training quantization to ~104 MB, and a fast local runner for a MacBook.

It's a learning project. The model writes fluent English but gets facts wrong, which is expected at this size (see [Limitations](#limitations)).

## Model

| | |
|---|---|
| Parameters | ~151.9M (embedding shared with output layer) |
| Layers / heads / width | 16 / 12 / 768 |
| Context length | 1024 tokens |
| Tokenizer | GPT-2 BPE via `tiktoken` (vocab padded to 50,304) |
| Architecture | Decoder-only transformer with RoPE positions, RMSNorm, GELU MLP (4x), tied embeddings, no biases, causal flash attention (PyTorch SDPA) |

## Training

- **Data:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`), about 3B tokens tokenized into a single `uint16` file. The last 10M tokens are held out for validation. A further ~1B tokens were added in a continued run with a lower learning rate.
- **Hardware:** one Google Colab G4 GPU (96 GB), about 221k tokens/s, roughly 3.8 hours for the main run.
- **Setup:** bf16 autocast, `torch.compile`, AdamW (betas 0.9/0.95, weight decay 0.1), peak LR 6e-4 with 200 warmup steps and cosine decay to 6e-5, gradient clipping at 1.0, 524,288 tokens per step (32 x 1024 x 16 accumulation), 5,702 steps.
- **Checkpointing:** saved to Google Drive every 250 steps, and the notebook resumes automatically after a disconnect.

Validation loss on the held-out tokens is **3.17** for the final model (measured in the quantization tests below).

## Quantization

The trained weights were compressed after training with plain round-to-nearest quantization (no retraining) and measured on the same validation set:

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
| `train_gpt_150m.ipynb` | Colab notebook: data download and tokenization, model definition, training loop, sampling |
| `export_int5_colab_cell.py` | Paste into a Colab cell after training to export the packed model and verify it by reloading |
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

1. Open `train_gpt_150m.ipynb` in Google Colab and select a GPU runtime (training was done on a G4; an A100 or L4 also works, with smaller batches on less memory).
2. Run the cells in order. Cell 2 tokenizes the data once (set `TARGET_TOKENS` lower for a quick test), and cell 4 trains.
3. After training, run `export_int5_colab_cell.py` in a new cell to create the packed model file.

Checkpoints and data are stored under `My Drive/gpt150m/`, so a Drive mount is needed.

## Limitations

- **Facts are unreliable.** The model writes plausible text but mixes up names, dates, and events. For example, it continued a Roman Empire prompt with invented details about Byzantine emperors.
- **It's a base model.** It continues text rather than following instructions, so it works best with an article-style prompt, not a question.
- **Repetition** can appear in longer outputs. The runner includes a repetition penalty to reduce it.
- **Short context:** 1024 tokens. When the cache fills, the runner keeps the newest half of the text.
- **English web text only** (educational pages), with little code or math.

## Possible next steps

- Instruction fine-tuning (SFT) on chat data
- A larger v2 model with a code-friendly tokenizer and a longer context
- Smarter quantization (GPTQ or AWQ-style) to make int4 usable
