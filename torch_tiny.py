import torch
import torch.nn as nn


class TransformerTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=256, embedding_dim=512)
        self.mha = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.linear_1 = nn.Linear(in_features=512, out_features=3072)
        self.linear_2 = nn.Linear(in_features=3072, out_features=512)
        self.prob_decoder = nn.Linear(in_features=512, out_features=10)

    def forward(self, input):
        input_embed = self.embedding(input)
        out_block_1 = self.mha(query=input_embed, key=input_embed, value=input_embed)[0]
        in_block_2 = input_embed + out_block_1
        out_block_2 = self.linear_2(torch.relu(self.linear_1(in_block_2)))
        return self.prob_decoder(in_block_2 + out_block_2)


def setup_bench(batch_size: int = 32):
    tt = TransformerTiny().cuda()
    return (
        torch.randint(0, 256, (batch_size, 256), device="cuda"),
        torch.ones(batch_size, 256, dtype=int, device="cuda"),
        tt,
        torch.optim.Adam(tt.parameters(), lr=1e-3),
    )


def compute_loss(tt, inputs, targets):
    model_out = tt(inputs).reshape(-1, 10)
    targets = targets.reshape(-1)
    return torch.nn.CrossEntropyLoss()(model_out, targets)


def compute_grad(tt, inputs, targets):
    tt.zero_grad()
    loss = compute_loss(tt, inputs, targets)
    loss.backward()


def apply_grad(tt, optimizer, inputs, targets):
    compute_grad(tt, inputs, targets)
    optimizer.step()


# compute_loss 1.71 ms ± 2.87 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# compute_grad 4 ms ± 18 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# batch_size 512 seq_len 128:
# %timeit compute_loss(tt,inputs, targets)
# 42.6 ms ± 92.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# %timeit compute_grad(tt, inputs, targets)
# 115 ms ± 299 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# + feedforward batch size 256, seq_len 256:

# compute_loss(tt, inputs, targets)
# 84.8 ms ± 236 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# compute_grad(tt, inputs, targets)
# 232 ms ± 767 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# apply_grad(tt, opt, inputs, targets)
# 225 ms ± 913 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# + residuals and an embedding layer

# compute_loss(tt, inputs, targets)
# 86.8 ms ± 397 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# compute_grad(tt, inputs, targets)
# 242 ms ± 1.33 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# apply_grad(tt, opt, inputs, targets)
# 243 ms ± 1.89 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

