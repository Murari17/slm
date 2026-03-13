from pathlib import Path
import re

import sentencepiece as spm
import torch

from model_def import TransformerModel


STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "in", "to", "for", "and", "or", "on", "with", "by", "do", "you", "mean",
}


def normalize_text(text: str) -> str:
    text = text.replace("\u2047", " ")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def extract_subject(question: str) -> tuple[str, str]:
    q = normalize_text(question.lower().rstrip("?.!"))
    if q.startswith("what is "):
        return q[len("what is "):].strip(), "is"
    if q.startswith("what are "):
        return q[len("what are "):].strip(), "are"
    return "", ""


def keyword_score(question: str, sentence: str, subject: str = "", verb: str = "") -> int:
    q = [w for w in words(question) if w not in STOPWORDS]
    if not q:
        return 0

    s_lower = sentence.lower()
    s_words = set(words(sentence))
    overlap = sum(1 for w in q if w in s_words)
    score = overlap * 10 - min(len(sentence), 220) // 40

    if subject:
        if subject in s_lower:
            score += 25
        if f"{subject} {verb}" in s_lower:
            score += 25
        if s_lower.startswith(f"{subject} {verb}") or s_lower.startswith(f"the {subject} {verb}"):
            score += 20

    if "ask:" in s_lower or "answer:" in s_lower:
        score -= 30

    return score


def best_retrieval(question: str, sentences: list[str]) -> tuple[str, int]:
    subject, verb = extract_subject(question)
    best = ""
    best_score = -10**9
    for sentence in sentences:
        candidate = normalize_text(sentence)
        if len(candidate) < 20:
            continue
        score = keyword_score(question, candidate, subject=subject, verb=verb)
        if score > best_score:
            best = candidate
            best_score = score
    return best, best_score


def weak_answer(question: str, answer: str) -> bool:
    answer = normalize_text(answer)
    if not answer or len(answer) < 24:
        return True
    if "Ask:" in answer or "Answer:" in answer:
        return True
    q_keywords = {w for w in words(question) if w not in STOPWORDS}
    if not q_keywords:
        return False
    a_words = set(words(answer))
    return len(q_keywords.intersection(a_words)) == 0


def generate_answer(
    model: TransformerModel,
    sp: spm.SentencePieceProcessor,
    question: str,
    max_new_tokens: int = 72,
    temperature: float = 0.65,
    top_k: int = 12,
) -> str:
    prompt = f"Ask: {question}\nAnswer:"
    input_ids = sp.encode(prompt)
    generated_ids: list[int] = []
    eos_id = sp.eos_id()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = input_ids + generated_ids
            if len(context) > model.max_seq_len:
                context = context[-model.max_seq_len:]

            x = torch.tensor(context, dtype=torch.long).unsqueeze(0)
            logits = model(x)[0, -1] / max(temperature, 0.1)

            k = min(top_k, logits.size(0))
            values, indices = torch.topk(logits, k)
            probs = torch.softmax(values, dim=-1)
            next_id = int(indices[torch.multinomial(probs, num_samples=1)].item())

            if next_id == eos_id:
                break

            generated_ids.append(next_id)
            text = sp.decode(generated_ids)
            if "\nAsk:" in text or "\nAnswer:" in text:
                break

    answer = normalize_text(sp.decode(generated_ids))
    for marker in ("\nAsk:", "Ask:", "\nAnswer:", "Answer:"):
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()

    if not answer:
        return "I need more training data to answer this clearly."
    return answer


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    tokenizer_path = base_dir / "tokenizer.model"
    model_path = base_dir / "model.pth"
    sentences_path = base_dir / "data" / "cleaned" / "sentences.txt"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))

    state_dict = torch.load(model_path, map_location="cpu")

    # Auto-detect architecture from checkpoint so any saved model loads correctly.
    embed_size = int(state_dict["embedding.weight"].shape[1])
    num_layers = sum(1 for k in state_dict if k.startswith("encoder.layers.") and k.endswith(".norm1.weight"))
    num_heads = 8 if embed_size >= 256 else 4
    max_seq_len = int(state_dict["pos_embedding.weight"].shape[0]) if "pos_embedding.weight" in state_dict else 512

    model = TransformerModel(
        vocab_size=sp.get_piece_size(),
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )
    model.load_state_dict(state_dict)
    model.eval()

    sentences: list[str] = []
    if sentences_path.exists():
        sentences = [line.strip() for line in sentences_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    question = input("Ask: ").strip()
    generated = generate_answer(model, sp, question)

    if sentences:
        retrieved, retrieved_score = best_retrieval(question, sentences)
        subject, verb = extract_subject(question)
        generated_score = keyword_score(question, generated, subject=subject, verb=verb)

        if subject and retrieved and subject not in retrieved.lower():
            retrieved_score = -10**9
            retrieved = ""

        if weak_answer(question, generated):
            if retrieved and retrieved_score >= 20:
                generated = retrieved
            else:
                generated = "I could not find a clear answer in the loaded notes."
        elif retrieved and retrieved_score >= 45 and generated_score + 8 < retrieved_score:
            generated = retrieved

    print(normalize_text(generated))


if __name__ == "__main__":
    main()

