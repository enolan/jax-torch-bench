import itertools
import flax.linen as fnn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from tqdm import trange, tqdm
from typing import Union

from data import get_train_data
from utils import ConfigurationError, ShapeError

SEQ_LEN = 128
D_MODEL = 768

seqs, vocab, tokenizer = get_train_data(SEQ_LEN)
seqs = jnp.array(seqs)


class TransformerLayer(fnn.Module):
    d_model: int
    num_heads: int
    ff_dim: int
    dropout: float

    def setup(self):
        # if d_model % num_heads != 0:
        #     raise ConfigurationError("d_model must be divisible by num_heads")

        self.mha = fnn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout,
            deterministic=False,
        )
        self.layer_norm_1 = fnn.LayerNorm()
        self.linear_1 = fnn.Dense(features=self.ff_dim)
        self.linear_2 = fnn.Dense(features=self.d_model)
        self.layer_norm_2 = fnn.LayerNorm()
        self.dropout_layer = fnn.Dropout(self.dropout, deterministic=False)

    def __call__(
        self, embeds: npt.NDArray[np.float32], mask: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        # "correct" type annotations for jax DeviceArrays are numpy ndarrays :<
        out_block_1 = self.layer_norm_1(self.mha(embeds, mask=mask))
        in_block_2 = embeds + out_block_1
        out_block_2 = self.layer_norm_2(
            self.dropout_layer(self.linear_2(jnn.relu(self.linear_1(in_block_2))))
        )
        return in_block_2 + out_block_2


class LM(fnn.Module):
    vocab_size: int
    n_layers: int
    d_model: int
    num_heads: int
    ff_dim: int
    dropout: float
    max_len: int

    def setup(self):
        self.word_embedding = fnn.Embed(
            num_embeddings=self.vocab_size, features=self.d_model
        )
        self.transformer_layers = [
            TransformerLayer(self.d_model, self.num_heads, self.ff_dim, self.dropout)
            for _ in range(self.n_layers)
        ]
        self.prob_decoder = fnn.Dense(features=self.vocab_size)
        self.positional_encoding = self.param(
            "positional_encoding",
            jnn.initializers.lecun_normal(),
            (self.max_len, self.d_model),
        )
        self.dropout_layer = fnn.Dropout(self.dropout, deterministic=False)

    def __call__(self, text):
        "Run the model, returning unnormalized log probabilities."
        if (
            len(text.shape) != 2
            or text.shape[1] != self.max_len
            or text.dtype != jnp.int32
        ):
            raise ShapeError(
                f"input text shape should be [batch, {self.max_len}] with dtype int. Got {text.shape}, {text.dtype}"
            )
        input = self.word_embedding(text)
        mask = fnn.attention.make_causal_mask(text)
        # Shift input right so causality isn't violated
        input = jnp.concatenate(
            [jnp.zeros([text.shape[0], 1, self.d_model]), input[:, :-1, :]], axis=1
        )
        input = input + self.positional_encoding
        input = self.dropout_layer(input)
        for tl in self.transformer_layers:
            input = tl(input, mask=mask)
        return self.prob_decoder(input)

    def sample(self, prompt: str, top_p: float = 0.95) -> str:
        tokens = jnp.array(vocab(tokenizer(prompt)), dtype=int)[None, :]
        prompt_tokens = tokens.shape[1]
        tokens = jnp.concatenate(
            [tokens, jnp.zeros([1, SEQ_LEN - prompt_tokens], dtype=int)], axis=1
        )
        rng = self.make_rng("token_sampling")
        chosen_tokens = []
        predict = jax.jit(self.__call__)
        for i in range(prompt_tokens, SEQ_LEN):
            unnorm_log_probs = predict(text=tokens)[0, i, :]
            sorted_indices = jnp.argsort(unnorm_log_probs)[::-1]
            cumulative_probs = jnp.cumsum(jnn.softmax(unnorm_log_probs[sorted_indices]))
            sorted_indices_to_remove = cumulative_probs > top_p
            inverse_permutation = np.empty_like(sorted_indices)
            inverse_permutation[sorted_indices] = np.arange(sorted_indices.size)
            indices_to_remove = np.nonzero(
                sorted_indices_to_remove[inverse_permutation]
            )
            unnorm_log_probs = unnorm_log_probs.at[indices_to_remove].add(-1e30)
            rng, rng2 = jax.random.split(rng)
            chosen_token = jax.random.categorical(rng2, unnorm_log_probs)
            chosen_tokens.append(chosen_token)
            tokens = tokens.at[0, i].set(chosen_token)
        return " ".join([vocab.lookup_token(tok) for tok in chosen_tokens])


def compute_loss(params, model, text, rng):
    model_out = model.apply(params, text=text, rngs={"dropout": rng})
    one_hots = jnn.one_hot(text, len(vocab))
    losses = optax.softmax_cross_entropy(model_out, one_hots)
    return losses.mean()


def setup_model(rng):
    model = LM(
        vocab_size=len(vocab),
        n_layers=8,
        d_model=D_MODEL,
        num_heads=12,
        ff_dim=3072,
        dropout=0.1,
        max_len=SEQ_LEN,
    )

    rng_p, rng_d = jax.random.split(rng)
    params = model.init(
        {"params": rng_p, "dropout": rng_d}, jnp.zeros([1, SEQ_LEN], dtype=jnp.int32)
    )
    return params, model


def setup_optimizer(params):
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    return optimizer, opt_state


def run_train_step(model, optimizer, opt_state, params, text, rng):
    loss, grad = jax.value_and_grad(
        lambda p: compute_loss(p, model, text=text, rng=rng)
    )(params)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def train_loop(
    model, optimizer, opt_state, params, batch_size, rng=None, n_epochs=None
):
    fast_train_step = jax.jit(
        run_train_step, static_argnames=["model", "optimizer"], donate_argnums=[2, 3]
    )
    # warm with dummy iter
    print("JITting...")
    params, opt_state, _ = fast_train_step(
        model,
        optimizer,
        opt_state,
        params,
        text=jnp.zeros([batch_size, SEQ_LEN], dtype=int),
        rng=rng,
    )

    try:
        for epoch in itertools.count():
            with trange(seqs.shape[0] // batch_size, leave=False) as pbar:
                for i in pbar:
                    batch = seqs[i * batch_size : (i + 1) * batch_size, :]
                    rng, rng2 = jax.random.split(rng)
                    params, opt_state, loss = fast_train_step(
                        model, optimizer, opt_state, params, text=batch, rng=rng2
                    )
                    pbar.set_postfix(loss=f"{loss:.4f}")
                print(f"After epoch {epoch}, loss {loss:.4f}")
                if epoch + 1 == n_epochs:
                    return params, opt_state
    except KeyboardInterrupt:
        return params, opt_state
    return params, opt_state


def setup_all():
    params, model = setup_model(jax.random.PRNGKey(11))
    optimizer, opt_state = setup_optimizer(params)
    return params, model, optimizer, opt_state
