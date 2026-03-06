# SMALL LANGUAGE MODEL

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

## Train and Test

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

## Generated Outputs

After running pipeline, these files are expected:

- `data/extracted/*.txt`
- `data/cleaned/chemistry_clean.txt`
- `data/cleaned/sentences.txt`
- `dataset/training_data.txt`
- `tokenizer.model`
- `tokenizer.vocab`
- `model.pth`

## Contribution Workflow (Add More PDFs and Improve Model)

Use this workflow when multiple people want to contribute new PDFs and improve the dataset/model together.

### 1) Fork and Clone

1. Fork this repository on GitHub.
2. Clone your fork locally.
3. Create a feature branch:

```powershell
git checkout -b add-new-chemistry-pdfs
```

### 2) Add New PDF Data

1. Add your new PDF files to `data/raw/`.
2. Use meaningful file names (for example: `unit3_electrochemistry.pdf`).
3. Only add PDFs you are allowed to share (respect copyright and license).

### 3) Rebuild Dataset and (Optionally) Retrain

Run preprocessing and tokenizer rebuild:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --skip-train
```

If you also want to test model quality locally:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --epochs 30
```

### 4) Validate Locally

Run question testing:

```powershell
.\.venv\Scripts\python.exe generate.py
```

Test at least 3-5 chemistry questions and note improvements.

### 5) Commit and Push

```powershell
git add .
git commit -m "Add new PDFs and regenerate training dataset"
git push origin add-new-chemistry-pdfs
```

### 6) Open Pull Request

Open a PR to `main` with:

1. What PDFs were added
2. Source/license information of those PDFs
3. What commands were run
4. Sample questions and answers before/after (if available)

### Recommended Team Policy

1. Prefer committing new source PDFs and regenerated dataset files.
2. Avoid frequent commits of `model.pth` unless publishing a tagged model version.
3. Retrain the official model after merged PRs to keep one consistent model artifact.
