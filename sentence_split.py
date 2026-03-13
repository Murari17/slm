from pathlib import Path
import re

import nltk
from nltk.tokenize import sent_tokenize


def ensure_tokenizer_resources() -> None:
    # Fully offline mode: only check local resources; do not download anything.
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except Exception:
            pass


base_dir = Path(__file__).resolve().parent
input_path = base_dir / "data" / "cleaned" / "corpus_clean.txt"
output_path = base_dir / "data" / "cleaned" / "sentences.txt"

ensure_tokenizer_resources()

text = input_path.read_text(encoding="utf-8")
try:
    sentences = sent_tokenize(text)
except LookupError:
    # If punkt resources are unavailable, do a simple punctuation-based split.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in parts if s and s.strip()]

output_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")