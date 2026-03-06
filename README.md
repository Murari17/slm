# SMALL LANGUAGE MODEL

This project builds a local QA system from one or more chemistry PDFs.
It uses your local tokenizer, local model, and local files only.
No external LLM API is used.

## Overview

The pipeline does the following:

1. Reads all PDFs from `data/raw/`
2. Extracts text into `data/extracted/`
3. Cleans and merges text into `data/cleaned/chemistry_clean.txt`
4. Splits text into sentences (`data/cleaned/sentences.txt`)
5. Builds QA-style training data (`dataset/training_data.txt`)
6. Trains a SentencePiece tokenizer (`tokenizer.model`, `tokenizer.vocab`)
7. Trains a Transformer language model (`model.pth`)
8. Runs interactive Q&A from terminal (`generate.py`)

## Project Files

- `run_pipeline.py`: full end-to-end runner
- `textextraction.py`: extract all PDFs from `data/raw/`
- `clean_text.py`: clean and merge extracted text
- `sentence_split.py`: split cleaned text to sentences
- `create_dataset.py`: create QA-style training samples
- `train_tokenizer.py`: train SentencePiece tokenizer
- `model_def.py`: Transformer architecture
- `train.py`: train model and save `model.pth`
- `generate.py`: ask questions in terminal
- `requirements.txt`: Python dependencies

## Prerequisites

- Python 3.11+ (tested with 3.13)
- Windows PowerShell

## Setup

From project root (`c:\Users\murar\Desktop\fyp`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Add Your PDFs

Put any number of source PDFs into:

`data/raw/`

Example files:

- `data/raw/unit1.pdf`
- `data/raw/unit2.pdf`
- `data/raw/engineering_chemistry_notes.pdf`

## Fastest Way to Run Everything

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --epochs 30
```

This executes extraction, cleaning, sentence split, dataset generation, tokenizer training, and model training.

## Train and Test (Recommended Flow)

1. Add/update PDFs in `data/raw/`
2. Run full pipeline:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --epochs 30
```

3. Test model interactively:

```powershell
.\.venv\Scripts\python.exe generate.py
```

4. Ask questions when prompted:

```text
Ask: What is hardness of water?
```

## Manual Step-by-Step Commands

Use this if you want full control over each stage:

```powershell
.\.venv\Scripts\python.exe textextraction.py
.\.venv\Scripts\python.exe clean_text.py
.\.venv\Scripts\python.exe sentence_split.py
.\.venv\Scripts\python.exe create_dataset.py
.\.venv\Scripts\python.exe train_tokenizer.py
$env:EPOCHS='30'; .\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe generate.py
```

## Useful Pipeline Options

```powershell
# preprocess + tokenizer only (no model training)
.\.venv\Scripts\python.exe run_pipeline.py --skip-train

# preprocess + training, reuse existing tokenizer
.\.venv\Scripts\python.exe run_pipeline.py --skip-tokenizer --epochs 30

# longer training
.\.venv\Scripts\python.exe run_pipeline.py --epochs 50
```

## Generated Outputs

After running pipeline, these files are expected:

- `data/extracted/*.txt`
- `data/cleaned/chemistry_clean.txt`
- `data/cleaned/sentences.txt`
- `dataset/training_data.txt`
- `tokenizer.model`
- `tokenizer.vocab`
- `model.pth`

## Notes on Answer Quality

- Better data quality and coverage improve answers more than epochs alone.
- Adding more relevant PDFs is the biggest boost.
- You can increase epochs (`--epochs 50` or more) for better fitting.
- `generate.py` also includes a retrieval fallback from `data/cleaned/sentences.txt` when generated output is weak.

## Troubleshooting

### `ModuleNotFoundError`

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### `spm_train` / tokenizer errors with vocab size

`train_tokenizer.py` already uses `hard_vocab_limit=False`, so it adapts to smaller corpora.

### PyTorch NumPy warning

```powershell
.\.venv\Scripts\python.exe -m pip install numpy
```

### Answers are weak or unrelated

1. Add more relevant PDFs to `data/raw/`
2. Re-run full pipeline
3. Increase epochs
4. Ask clearer, topic-specific questions

## Reproducibility Tips

- Always run commands from project root.
- Prefer `.\.venv\Scripts\python.exe ...` for consistent environment usage.
- Re-run full pipeline whenever you add or remove PDFs.
