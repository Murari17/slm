import re
from pathlib import Path


def clean_text(raw: str) -> str:
    text = raw
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2022", " ").replace("\u27a2", " ")
    text = text.replace("\u2047", " ")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"Page\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bModule\s*[-:]?\s*\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[•➢]+\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


base_dir = Path(__file__).resolve().parent
extracted_dir = base_dir / "data" / "extracted"
cleaned_dir = base_dir / "data" / "cleaned"
output_path = cleaned_dir / "corpus_clean.txt"

text_files = sorted(extracted_dir.glob("*.txt"))
if not text_files:
    raise FileNotFoundError(f"No extracted text files found in: {extracted_dir}")

cleaned_chunks: list[str] = []
for path in text_files:
    raw = path.read_text(encoding="utf-8")
    cleaned = clean_text(raw)
    if cleaned:
        cleaned_chunks.append(cleaned)

combined_text = "\n".join(cleaned_chunks)

cleaned_dir.mkdir(parents=True, exist_ok=True)
output_path.write_text(combined_text, encoding="utf-8")

print(f"Cleaning done for {len(text_files)} extracted file(s).")
print(f"Saved cleaned corpus: {output_path}")