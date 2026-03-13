# SMALL LANGUAGE MODEL

## Documentation PDF

Project documentation (PDF):

https://drive.google.com/file/d/10cIpMwRrJhPXKvwFGhlByFj74Fl-Kubm/view?usp=sharing

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

From project root:

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

## Train and Test (Basic CPU Mode)

1. Add/update PDFs in `data/raw/`
2. Run full pipeline:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --epochs 12
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
$env:EPOCHS='12'; .\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe generate.py
```

Notes:

- Training and inference are CPU-only by design.
- The project uses basic generation with a simple retrieval fallback.
- If you have a smaller machine, reduce epochs further (for example `--epochs 8`).

## Generated Outputs

After running pipeline, these files are expected:

- `data/extracted/*.txt`
- `data/cleaned/corpus_clean.txt`
- `data/cleaned/sentences.txt`
- `dataset/training_data.txt`
- `tokenizer.model`
- `tokenizer.vocab`
- `model.pth`

## Contribution Workflow

### 1) Fork and Clone

1. Fork this repository on GitHub.
2. Clone your fork locally.
3. Create a feature branch:

```powershell
git checkout -b add-new-pdfs
```

### 2) Add New PDF Data

1. Add your new PDF files to `data/raw/`.
2. Use meaningful file names (for example: `physics.pdf`).

### 3) Rebuild Dataset and (Optionally) Retrain

Run preprocessing and tokenizer rebuild:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --skip-train
```

If you also want to test model quality locally:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --epochs 12
```

### 4) Validate Locally

Run question testing:

```powershell
.\.venv\Scripts\python.exe generate.py
```

Test at least 3-5 questions and note improvements.

### 5) Commit and Push

```powershell
git add .
git commit -m "Add new PDFs and regenerate training dataset"
git push origin add-new-pdfs
```
