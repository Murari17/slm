from pathlib import Path
import os
import re
import sys


def ensure_project_venv() -> None:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / ".venv" / "bin" / "python",  # macOS/Linux
        base_dir / ".venv" / "Scripts" / "python.exe",  # Windows
    ]

    current_python = Path(sys.executable).resolve()
    for candidate in candidates:
        if candidate.exists() and current_python != candidate.resolve():
            os.execv(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]])


ensure_project_venv()

import sentencepiece as spm
import torch

from model_def import TransformerModel


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def retrieve_best_sentence(question: str, sentences: list[str]) -> str:
    q_words = set(tokenize_words(question))
    if not q_words:
        return ""

    stopwords = {
        "what", "is", "are", "the", "a", "an", "of", "in", "to", "for", "and", "or", "on", "with", "by",
    }
    key_words = {w for w in q_words if w not in stopwords} or q_words

    # Prefer candidates containing all key words when possible.
    strict_candidates = [s for s in sentences if all(k in s.lower() for k in key_words)]
    candidates = strict_candidates if strict_candidates else sentences

    best_sentence = ""
    best_score = 0

    for sentence in candidates:
        s_words = tokenize_words(sentence)
        if not s_words:
            continue

        overlap = sum(1 for w in s_words if w in key_words)
        if overlap == 0:
            continue

        s_text = sentence.lower()
        bonus = 0
        if "hardness of water is" in s_text:
            bonus += 8
        if "hardness" in s_text:
            bonus += 2
        if question.lower().startswith("what is") and " is " in s_text:
            bonus += 2

        score = overlap + bonus
        if score > best_score or (score == best_score and best_sentence and len(sentence) < len(best_sentence)):
            best_score = score
            best_sentence = sentence

    best_sentence = best_sentence.strip()
    for marker in ["Hardness:", "hardness:", "Causes:", "causes:"]:
        idx = best_sentence.find(marker)
        if idx != -1:
            best_sentence = best_sentence[idx:]
            break

    return best_sentence


def looks_weak_answer(question: str, answer: str) -> bool:
    if not answer or len(answer) < 30:
        return True

    bad_markers = ["ask:", "answer:", "\u2047", "nps"]
    answer_l = answer.lower()
    if any(marker in answer_l for marker in bad_markers):
        return True

    q_words = set(tokenize_words(question))
    a_words = set(tokenize_words(answer))
    stopwords = {
        "what", "is", "are", "the", "a", "an", "of", "in", "to", "for", "and", "or", "on", "with", "by",
    }
    q_key_words = {w for w in q_words if w not in stopwords} or q_words

    return bool(q_key_words and len(q_key_words.intersection(a_words)) == 0)


def generate_answer(
    model: TransformerModel,
    sp: spm.SentencePieceProcessor,
    question: str,
    max_new_tokens: int = 80,
    temperature: float = 0.6,
    top_k: int = 10,
    repetition_penalty: float = 1.15,
) -> str:
    prompt_text = f"Ask: {question}\nAnswer:"
    input_ids = sp.encode(prompt_text)
    generated_ids: list[int] = []
    eos_id = sp.eos_id()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_ids = input_ids + generated_ids
            if len(context_ids) > model.max_seq_len:
                context_ids = context_ids[-model.max_seq_len:]

            x = torch.tensor(context_ids, dtype=torch.long).unsqueeze(0)
            logits = model(x)[0, -1]

            if repetition_penalty > 1.0 and generated_ids:
                for token_id in set(generated_ids[-50:]):
                    logits[token_id] = logits[token_id] / repetition_penalty

            if temperature > 0:
                logits = logits / temperature

            if top_k > 0:
                k = min(top_k, logits.size(0))
                values, indices = torch.topk(logits, k)
                probs = torch.softmax(values, dim=-1)
                sample_idx = torch.multinomial(probs, num_samples=1)
                next_id = int(indices[sample_idx].item())
            else:
                next_id = int(torch.argmax(logits).item())

            if next_id == eos_id:
                break

            generated_ids.append(next_id)

            partial = sp.decode(generated_ids)
            if "Ask:" in partial or "\nAsk:" in partial:
                break
            if "\nAnswer:" in partial:
                break

    answer = sp.decode(generated_ids).strip()
    for marker in ["\nAsk:", "Ask:", "\nAnswer:", "Answer:"]:
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()

    if "\n" in answer:
        answer = answer.split("\n", 1)[0].strip()

    if not answer:
        answer = "I need more training data to answer that clearly."

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
    max_seq_len = 256
    if "pos_embedding.weight" in state_dict:
        max_seq_len = int(state_dict["pos_embedding.weight"].shape[0])

    model = TransformerModel(vocab_size=sp.get_piece_size(), max_seq_len=max_seq_len)
    model.load_state_dict(state_dict)
    model.eval()

    prompt = input("Ask: ")
    generated = generate_answer(model, sp, prompt)

    if sentences_path.exists():
        with sentences_path.open("r", encoding="utf-8") as file:
            sentences = [line.strip() for line in file if line.strip()]

        if looks_weak_answer(prompt, generated):
            retrieved = retrieve_best_sentence(prompt, sentences)
            if retrieved:
                generated = retrieved

    print(generated)


if __name__ == "__main__":
    main()
