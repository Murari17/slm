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
