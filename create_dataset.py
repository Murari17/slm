from pathlib import Path
import re


base_dir = Path(__file__).resolve().parent
input_path = base_dir / "data" / "cleaned" / "sentences.txt"
output_dir = base_dir / "dataset"
output_path = output_dir / "training_data.txt"

with input_path.open("r", encoding="utf-8") as f:
    sentences = f.readlines()

dataset = []


def normalize_question(text: str) -> str:
    text = text.strip().rstrip(".?!")
    return text


def build_question_from_sentence(sentence: str) -> str:
    # If the sentence looks like a definition, turn it into a direct question.
    m = re.match(r"^([A-Za-z][A-Za-z0-9\-\s]{2,60}?)\s+(is|are)\s+", sentence)
    if m:
        subject = normalize_question(m.group(1))
        verb = m.group(2)
        if verb == "is":
            return f"What is {subject}?"
        return f"What are {subject}?"

    # Otherwise build a short "explain this" style prompt.
    words = sentence.split()
    snippet = " ".join(words[:12]).strip()
    snippet = normalize_question(snippet)
    return f"Explain: {snippet}?"


def build_question_variants(sentence: str) -> list[str]:
    primary = build_question_from_sentence(sentence).strip()
    short = normalize_question(" ".join(sentence.split()[:10]))
    fallback = f"Explain: {short}?"
    if primary.lower() == fallback.lower():
        return [primary]
    return [primary, fallback]


for s in sentences:
    s = s.strip()

    if len(s) < 20:
        continue

    for question in build_question_variants(s):
        dataset.append(f"Ask: {question}\nAnswer: {s}\n")

output_dir.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as f:
    for d in dataset:
        f.write(d + "\n")

print("Dataset created!")