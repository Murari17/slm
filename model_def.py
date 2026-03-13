import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: [batch, seq_len]
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)

        # Standard causal mask: each token can only see itself and earlier tokens.
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
        h = self.encoder(h, mask=mask)
        return self.fc(h)
