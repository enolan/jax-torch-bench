import itertools
import flax.linen as fnn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
import pickle
import random
from tqdm import trange, tqdm
from typing import Union

from config import ModelConfig
from data import Enwik9Loader
from utils import ConfigurationError, EWMA, ShapeError


class TransformerLayer(fnn.Module):
    d_model: int
    num_heads: int
    ff_dim: int
    dropout: float

    def setup(self):
        self.mha = fnn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            # dropout in the attention matrix was introduced in
            # https://arxiv.org/abs/1907.11065, it's *not* the normal thing
            # from Attention is All You Need.
            dropout_rate=0,
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
        in_block_2 = embeds + self.dropout_layer(out_block_1)
        out_block_2 = self.layer_norm_2(
            self.dropout_layer(self.linear_2(jnn.relu(self.linear_1(in_block_2))))
        )
        return in_block_2 + out_block_2


class LM(fnn.Module):
    cfg: ModelConfig

    def setup(self):
        self.byte_embedding = fnn.Embed(num_embeddings=256, features=self.cfg.d_model)
        self.transformer_layers = [
            TransformerLayer(
                self.cfg.d_model, self.cfg.num_heads, self.cfg.ff_dim, self.cfg.dropout
            )
            for _ in range(self.cfg.n_layers)
        ]
        self.prob_decoder = fnn.Dense(features=256)
        self.positional_encoding = self.param(
            "positional_encoding",
            jnn.initializers.lecun_normal(),
            (self.cfg.seq_len, self.cfg.d_model),
        )
        self.dropout_layer = fnn.Dropout(self.cfg.dropout, deterministic=False)

    def __call__(self, text):
        "Run the model, returning unnormalized log probabilities."
        if (
            len(text.shape) != 1
            or text.shape[0] != self.cfg.seq_len
            or text.dtype != jnp.uint8
        ):
            raise ShapeError(
                f"input text shape should be [{self.cfg.seq_len}] with dtype uint8. Got {text.shape}, {text.dtype}"
            )
        input = self.byte_embedding(text)
        mask = fnn.attention.make_causal_mask(text)
        # Shift input right so causality isn't violated
        input = jnp.concatenate(
            [jnp.zeros([1, self.cfg.d_model]), input[:-1, :]], axis=0
        )
        input = input + self.positional_encoding
        input = self.dropout_layer(input)
        for tl in self.transformer_layers:
            input = tl(input, mask=mask)
        return self.prob_decoder(input)

    def sample(self, prompt: str, top_p: float = 0.95) -> str:
        bytes_in = jnp.array(np.frombuffer(prompt.encode("utf-8"), dtype=np.uint8))
        prompt_tokens = bytes_in.shape[0]
        tokens = jnp.concatenate(
            [bytes_in, jnp.zeros([self.cfg.seq_len - prompt_tokens], dtype=jnp.uint8)],
            axis=0,
        )
        rng = self.make_rng("token_sampling")
        chosen_tokens = []
        # On my machine with Python 3.9, jax.jit(self.__call__) works, but it's
        # broken on Colab with Python 3.7.
        predict = jax.jit(lambda text: self(text))
        for i in range(prompt_tokens, self.cfg.seq_len):
            unnorm_log_probs = predict(text=tokens)[i, :]
            sorted_indices = jnp.argsort(unnorm_log_probs)[::-1]
            cumulative_probs = jnp.cumsum(jnn.softmax(unnorm_log_probs[sorted_indices]))
            sorted_indices_to_remove = cumulative_probs > top_p
            inverse_permutation = np.empty_like(sorted_indices)
            inverse_permutation[sorted_indices] = np.arange(sorted_indices.size)
            indices_to_remove = np.nonzero(
                sorted_indices_to_remove[inverse_permutation]
            )
            filtered_log_probs = unnorm_log_probs.at[indices_to_remove].set(-1e30)
            # Always preserve the most likely token, in case it has > top_p
            # probability on its own.
            filtered_log_probs = filtered_log_probs.at[sorted_indices[0]].set(
                unnorm_log_probs[sorted_indices[0]]
            )
            rng, rng2 = jax.random.split(rng)
            chosen_token = jax.random.categorical(rng2, filtered_log_probs)
            chosen_tokens.append(chosen_token.item())
            tokens = tokens.at[i].set(chosen_token)
        return prompt + bytes(chosen_tokens).decode("utf-8")


def compute_loss(params, model: LM, text, rng):
    model_out = model.apply(params, text=text, rngs={"dropout": rng})
    one_hots = jnn.one_hot(text, 256)
    loss = optax.softmax_cross_entropy(model_out, one_hots)
    return loss


def setup_model(rng, cfg: ModelConfig):
    model = LM(cfg)

    rng_p, rng_d = jax.random.split(rng)
    params = model.init(
        {"params": rng_p, "dropout": rng_d}, jnp.zeros([cfg.seq_len], dtype=jnp.uint8)
    )
    return params, model


def setup_optimizer(params, cfg: ModelConfig):
    optimizer = optax.adam(cfg.learning_rate)
    opt_state = optimizer.init(params)
    return optimizer, opt_state


def train_loop(
    model: LM, optimizer, opt_state, params, cfg: ModelConfig, datapath: str, rng=None
):
    rng = (
        rng
        if rng is not None
        else jax.random.PRNGKey(random.randrange(-(2 ** 63), 2 ** 63))
    )

    def run_train_step(opt_state, params, text_batch, rng):
        rng, rng2 = jax.random.split(rng)
        loss, grad = jax.value_and_grad(
            lambda p: jax.vmap(
                lambda text: compute_loss(p, model, text=text, rng=rng),
                in_axes=0,
                out_axes=0,
            )(text_batch).mean()
        )(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, rng2

    fast_train_step = jax.jit(run_train_step, donate_argnums=[0, 1, 3])
    # warm with dummy iter
    print("JITting...", end="", flush=True)
    params, opt_state, loss, rng = fast_train_step(
        opt_state,
        params,
        jnp.zeros([cfg.batch_size, cfg.seq_len], dtype=jnp.uint8),
        rng,
    )
    print(" done.")

    ewma = EWMA(smoothing_factor=0.99)
    loss = None
    try:
        for epoch in itertools.count():
            with tqdm(
                list(Enwik9Loader(cfg.batch_size, cfg.seq_len, datapath)), leave=False
            ) as pbar:
                for idx, batch in enumerate(pbar):
                    batch = jnp.array(batch)
                    if loss is not None:
                        smoothed_loss = ewma.update_ewma(loss)
                        pbar.set_postfix(
                            loss=f"{loss:.4f}", smoothed_loss=f"{smoothed_loss:.4f}"
                        )
                        if idx % 1000 == 0:
                            print(f"At step {idx}, smoothed loss {smoothed_loss:.4f}")
                            save_model(params, opt_state, "jax_model.pkl")
                    params, opt_state, loss, rng = fast_train_step(
                        opt_state, params, batch, rng
                    )
            print(
                f"Epoch {epoch} complete, loss {loss:.4f}, smoothed loss {smoothed_loss:.4f}"
            )
            break
    except KeyboardInterrupt:
        return params, opt_state
    return params, opt_state


def setup_all(cfg: ModelConfig, rng=None):
    rng = (
        rng
        if rng is not None
        else jax.random.PRNGKey(random.randrange(-(2 ** 63), 2 ** 63))
    )

    params, model = setup_model(rng, cfg)
    optimizer, opt_state = setup_optimizer(params, cfg)

    def sample(params, prompt):
        rng_d, rng_s = jax.random.split(
            jax.random.PRNGKey(random.randrange(-(2 ** 63), 2 ** 63))
        )
        return model.apply(
            params,
            prompt,
            rngs={"dropout": rng_d, "token_sampling": rng_s},
            method=LM.sample,
        )

    return params, model, optimizer, opt_state, sample


def save_model(params, opt_state, name):
    with open(name, "wb") as f:
        pickle.dump((params, opt_state), f)


def load_model(name):
    with open(name, "rb") as f:
        return pickle.load(f)
