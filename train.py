import os
from pathlib import Path

import sentencepiece as spm
import torch

from model_def import TransformerModel


def build_batches(data: torch.Tensor, seq_len: int, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    stride = seq_len * batch_size
    for i in range(0, len(data) - stride, stride):
        batch_x: list[torch.Tensor] = []
        batch_y: list[torch.Tensor] = []
        for j in range(batch_size):
            start = i + j * seq_len
            end = start + seq_len + 1
            if end > len(data):
                break
            chunk = data[start:end]
            batch_x.append(chunk[:-1])
            batch_y.append(chunk[1:])
        if batch_x:
            batches.append((torch.stack(batch_x), torch.stack(batch_y)))
    return batches


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    tokenizer_path = base_dir / "tokenizer.model"
    dataset_path = base_dir / "dataset" / "training_data.txt"
    model_out_path = base_dir / "model.pth"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))

    text = dataset_path.read_text(encoding="utf-8")
    records = [r.strip() for r in text.split("\n\n") if r.strip()]
    eos_id = sp.eos_id()

    tokens: list[int] = []
    for record in records:
        tokens.extend(sp.encode(record))
        if eos_id != -1:
            tokens.append(eos_id)

    data = torch.tensor(tokens, dtype=torch.long)
    if len(data) < 200:
        raise ValueError("Training data is too small. Add more source text and run the pipeline again.")

    vocab_size = sp.get_piece_size()
    seq_len = 64
    batch_size = 16
    epochs = int(os.getenv("EPOCHS", "12"))

    device = torch.device("cpu")
    print(f"Training on CPU | vocab: {vocab_size} | tokens: {len(data)}")

    model = TransformerModel(vocab_size=vocab_size, max_seq_len=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_batches = build_batches(data, seq_len=seq_len, batch_size=batch_size)

    if not train_batches:
        raise ValueError("Could not create training batches. Increase dataset size or lower sequence length.")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_batches:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_batches)
        print(f"Epoch {epoch + 1:>3}/{epochs} | train: {avg_train_loss:.4f}")

    torch.save(model.state_dict(), model_out_path)
    print(f"Training complete. Saved model to: {model_out_path}")


if __name__ == "__main__":
    main()