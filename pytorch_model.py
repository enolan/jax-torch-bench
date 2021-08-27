import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional

from data import Enwik9Loader
from utils import EWMA


class LM(nn.Module):
    def __init__(self, seq_len, d_model, n_layers, num_heads, ff_dim):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        self.register_parameter(
            "positional_encoding", nn.Parameter(torch.empty(seq_len, d_model))
        )
        nn.init.normal_(self.positional_encoding)

        self.byte_embedding = nn.Embedding(num_embeddings=256, embedding_dim=d_model)
        t_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=t_layer, num_layers=n_layers
        )
        self.prob_decoder = nn.Linear(in_features=d_model, out_features=256)

    def forward(self, text_batch):
        batch_size = text_batch.shape[0]
        # Shift input right so causality isn't violated
        embeddings = self.byte_embedding(text_batch.int())
        embeddings = torch.cat(
            [
                torch.zeros(batch_size, 1, self.d_model, device=text_batch.device),
                embeddings[:, :-1, :],
            ],
            axis=1,
        )
        embeddings = nn.Dropout(p=0.1)(embeddings + self.positional_encoding)
        output_probabilities = self.prob_decoder(
            self.transformer(
                embeddings,
                mask=nn.Transformer.generate_square_subsequent_mask(
                    None, self.seq_len
                ).to(embeddings),
            )
        )
        return output_probabilities

    def sample(self, prompt: str, top_p: float = 0.95) -> str:
        assert len(prompt) < self.seq_len
        prompt_bytes = prompt.encode("utf-8")
        input = torch.cat(
            [
                torch.tensor(
                    np.frombuffer(prompt_bytes, dtype=np.uint8), device="cuda"
                )[None, :],
                torch.zeros(
                    1,
                    self.seq_len - len(prompt_bytes),
                    dtype=torch.uint8,
                    device="cuda",
                ),
            ],
            dim=1,
        )
        for i in range(len(prompt_bytes), self.seq_len):
            out_probs = self(input)[0][i]
            sorted_probs, sorted_indices = out_probs.sort(descending=True)
            cumulative_probs = sorted_probs.softmax(0).cumsum(0)
            sorted_indices_to_remove = cumulative_probs > top_p
            inverse_permutation = torch.empty_like(sorted_indices)
            inverse_permutation[sorted_indices] = torch.arange(
                sorted_indices.shape[0], device="cuda"
            )
            indices_to_remove = sorted_indices_to_remove[inverse_permutation].nonzero(
                as_tuple=True
            )
            filtered_probs = out_probs.clone()
            filtered_probs[indices_to_remove] = -1e30
            # Always keep most likely token, in case it has > top_p probability on its own.
            filtered_probs[sorted_indices[0]] = out_probs[sorted_indices[0]]
            out_token = filtered_probs.softmax(0).multinomial(1)
            input[0, i] = out_token
        return bytes(input[0]).decode("utf-8")


def compute_loss(lm, batch):
    probs = lm(batch)
    probs = probs.reshape(-1, 256)
    targets = batch.reshape(-1).long()
    return nn.CrossEntropyLoss()(probs, targets)


optimizer: Optional[torch.optim.Optimizer] = None


def train_loop(lm: LM, batch_size: int, seq_len: int) -> None:
    global optimizer
    optimizer = (
        torch.optim.Adam(lm.parameters(), 1e-4) if optimizer is None else optimizer
    )
    ewma = EWMA(0.99)
    try:
        for epoch in itertools.count():
            with tqdm(
                list(enumerate(Enwik9Loader(batch_size, seq_len))), leave=False
            ) as pbar:
                for i, batch in pbar:
                    loss = compute_loss(lm, torch.tensor(batch, device="cuda"))
                    smoothed_loss = ewma.update_ewma(loss)
                    pbar.set_postfix(
                        loss=f"{loss:.4f}", smoothed_loss=f"{smoothed_loss:.4f}"
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if i % 1000 == 0:
                        print(
                            f"At iter {i}, loss {loss:.4f}, smoothed loss {smoothed_loss:.4f}"
                        )
                print(
                    f"Epoch {epoch} completed, loss {loss:.4f}, smoothed loss {smoothed_loss:.4f}"
                )
    except KeyboardInterrupt:
        pass
