import haiku as hk
from haiku import LayerNorm, Linear, MultiHeadAttention
import itertools
import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy.typing as npt
import optax
from tqdm import trange, tqdm
from typing import Union

from data import get_train_data
from utils import ConfigurationError, ShapeError

SEQ_LEN = 20
D_MODEL = 2048

seqs, vocab, tokenizer = get_train_data(SEQ_LEN)
seqs = jnp.array(seqs)


class TransformerLayer(hk.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        init_scale: float,
        mask: jnp.ndarray,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ConfigurationError("d_model must be divisible by num_heads")

        self.dropout = dropout
        self.mask = mask
        self.mha = MultiHeadAttention(
            num_heads, d_model // num_heads, w_init_scale=init_scale
        )
        self.layer_norm_1 = LayerNorm(
            axis=2, create_scale=True, create_offset=True, name="layer_norm_1"
        )
        self.linear_1 = Linear(output_size=ff_dim, name="linear_1")
        self.linear_2 = Linear(output_size=d_model, name="linear_2")
        self.layer_norm_2 = LayerNorm(
            axis=2, create_scale=True, create_offset=True, name="layer_norm_2"
        )

    def __call__(self, embeds):
        # TODO add type annotations once I figure out the deal with adding Jax
        # arrays:
        # https://stackoverflow.com/questions/68884215/why-does-mypy-think-adding-two-jax-arrays-returns-a-numpy-array
        out_block_1 = self.layer_norm_1(
            hk.dropout(
                hk.next_rng_key(),
                self.dropout,
                self.mha(embeds, embeds, embeds, mask=self.mask),
            )
        )
        in_block_2 = embeds + out_block_1
        out_block_2 = self.layer_norm_2(
            hk.dropout(
                hk.next_rng_key(),
                self.dropout,
                self.linear_2(nn.relu(self.linear_1(in_block_2))),
            )
        )
        return in_block_2 + out_block_2


class LM(hk.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.max_len = max_len

        causal_mask = jnp.tril(jnp.ones([max_len, max_len]))
        # wait should this be lower or upper triangular? I'm cribbing from a
        # Haiku example which uses lower...
        # brief experiment says this is right: changing a later token doesn't
        # change the output of an earlier one, but the opposite does change it.

        self.word_embedding = hk.Embed(vocab_size, embed_dim=d_model)
        self.transformer_layers = hk.Sequential(
            [
                TransformerLayer(
                    d_model, num_heads, ff_dim, dropout, 2.0 / n_layers, causal_mask
                )
                for _ in range(n_layers)
            ]
        )
        self.prob_decoder = hk.Linear(output_size=vocab_size)

    def __call__(self, text):
        "Run the model, returning unnormalized log probabilities."
        if len(text.shape) != 2 or text.shape[1] != self.max_len:
            raise ShapeError(
                f"input text shape should be [batch, {self.max_len}] with dtype int. Got {text.shape}"
            )
        print(f"text {text.shape}")
        input = self.word_embedding(text)
        print(f"input {input.shape}")
        # Shift input right so causality isn't violated
        input = jnp.concatenate(
            [jnp.zeros([text.shape[0], 1, self.d_model]), input[:, :-1, :]], axis=1
        )
        input = input + hk.get_parameter(
            "positional_encoding",
            [self.max_len, self.d_model],
            init=hk.initializers.TruncatedNormal(),
        )
        return self.prob_decoder(
            self.transformer_layers(hk.dropout(hk.next_rng_key(), self.dropout, input))
        )


def compute_loss(params, eval_model, text, vocab_size, rng):
    model_out = eval_model(rng=rng, params=params, text=text)
    one_hots = nn.one_hot(text, vocab_size)
    losses = optax.softmax_cross_entropy(model_out, one_hots)
    return losses.mean()


def setup_model(rng):
    def f():
        model = LM(
            vocab_size=len(vocab),
            n_layers=8,
            d_model=128,
            num_heads=8,
            ff_dim=2048,
            dropout=0.1,
            max_len=SEQ_LEN,
        )

        def eval_model(text):
            return model(text)

        def sample_from_model(prompt):
            tokens = jnp.array(vocab(tokenizer(prompt)), dtype=int)[None, :]
            prompt_tokens = tokens.shape[1]
            tokens = jnp.concatenate(
                [tokens, jnp.zeros([1, SEQ_LEN - prompt_tokens], dtype=int)], axis=1
            )
            chosen_tokens = []
            for i in range(prompt_tokens, SEQ_LEN):
                unnorm_log_probs = model(text=tokens)[0, i, :]
                chosen_token = jax.random.categorical(
                    hk.next_rng_key(), unnorm_log_probs
                )
                chosen_tokens.append(chosen_token)
                tokens = tokens.at[0, i].set(chosen_token)
            return chosen_tokens

        def init(text):
            return eval_model(text)

        return init, (eval_model, sample_from_model)

    f = hk.multi_transform(f)
    return (
        f.init(text=jnp.zeros([1, SEQ_LEN], dtype=int), rng=rng),
        f.apply[0],
        f.apply[1],
    )


def setup_optimizer(params):
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    return optimizer, opt_state


def run_train_step(eval_model, optimizer, opt_state, params, rng, text):
    loss, grad = jax.value_and_grad(
        lambda p: compute_loss(p, eval_model, text=text, vocab_size=len(vocab), rng=rng)
    )(params)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def train_loop(eval_model, optimizer, opt_state, params, batch_size, rng, n_iters=None):
    n_iters = seqs.shape[0] // batch_size if n_iters is None else n_iters

    fast_train_step = jax.jit(
        run_train_step, static_argnames=["eval_model", "optimizer"]
    )
    # warm with dummy iter
    print("JITting...")
    fast_train_step(
        eval_model,
        optimizer,
        opt_state,
        params,
        rng,
        text=jnp.zeros([batch_size, SEQ_LEN], dtype=int),
    )

    try:
        for epoch in itertools.count():
            with trange(n_iters, leave=False) as pbar:
                for i in pbar:
                    rng, rng2 = jax.random.split(rng)

                    batch = seqs[i * batch_size : (i + 1) * batch_size, :]
                    params, opt_state, loss = fast_train_step(
                        eval_model, optimizer, opt_state, params, rng2, text=batch
                    )
                    pbar.set_postfix(loss=f"{loss:.4f}")
            print(f"After epoch {epoch}, loss {loss:.4f}")
    except KeyboardInterrupt:
        return params, opt_state
