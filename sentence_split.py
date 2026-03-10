from pathlib import Path
import re

import nltk
from nltk.tokenize import sent_tokenize


def ensure_tokenizer_resources() -> None:
    # Newer NLTK versions may require both resources for sentence tokenization.
    # Also guard against corrupt cached zips (BadZipFile) and SSL errors.
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        token_name = resource.split("/")[1]
        try:
            nltk.data.find(resource)
        except Exception:
            try:
                nltk.download(token_name, quiet=True)
            except Exception:
                pass


base_dir = Path(__file__).resolve().parent
input_path = base_dir / "data" / "cleaned" / "chemistry_clean.txt"
output_path = base_dir / "data" / "cleaned" / "sentences.txt"

ensure_tokenizer_resources()

text = input_path.read_text(encoding="utf-8")
try:
    sentences = sent_tokenize(text)
except LookupError:
    # Offline fallback when NLTK resources cannot be downloaded.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in parts if s and s.strip()]

output_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")