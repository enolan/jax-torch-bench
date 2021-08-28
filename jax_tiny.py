import flax.linen as fnn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import optax

d_model = 512


class TransformerLayer(fnn.Module):
    def setup(self):
        self.mha = fnn.SelfAttention(num_heads=8, qkv_features=d_model)
        self.layer_norm_1 = fnn.LayerNorm()
        self.linear_1 = fnn.Dense(features=3072)
        self.linear_2 = fnn.Dense(features=d_model)
        self.layer_norm_2 = fnn.LayerNorm()
        self.dropout_layer = fnn.Dropout(0.1, deterministic=False)

    def __call__(self, embeds, mask):
        out_block_1 = self.layer_norm_1(self.mha(embeds, mask=mask))
        in_block_2 = embeds + self.dropout_layer(out_block_1)
        out_block_2 = self.layer_norm_2(
            self.dropout_layer(self.linear_2(jnn.relu(self.linear_1(in_block_2))))
        )
        return in_block_2 + out_block_2


class TransformerTiny(fnn.Module):
    def setup(self):

        self.embedding = fnn.Embed(num_embeddings=256, features=d_model)
        self.positional_encoding = self.param(
            "positional_encoding", jnn.initializers.lecun_normal(), (256, d_model)
        )
        self.transformer_layers = [TransformerLayer() for _ in range(8)]
        self.prob_decoder = fnn.Dense(features=256)
        self.dropout_layer = fnn.Dropout(0.1, deterministic=False)

    def __call__(self, input):
        input_embed = self.embedding(input) + self.positional_encoding
        input_embed = self.dropout_layer(
            jnp.concatenate([jnp.zeros([1, d_model]), input_embed[:-1, :]], axis=0)
        )
        mask = fnn.attention.make_causal_mask(input)
        for tl in self.transformer_layers:
            input_embed = tl(input_embed, mask=mask)
        return self.prob_decoder(input_embed)


def setup_bench(batch_size: int = 32):
    rng = jax.random.PRNGKey(11)
    rng_params, rng_input = jax.random.split(rng)
    tt = TransformerTiny()
    params = tt.init(
        {"params": rng_params, "dropout": jax.random.PRNGKey(0)},
        jnp.zeros([256], dtype=jnp.uint8),
    )
    eval_fn = lambda params, rng, inputs: jax.vmap(
        lambda input: tt.apply(params, input, rngs={"dropout": rng})
    )(inputs)
    eval_fn_fast = jax.jit(eval_fn)

    def compute_loss(params, rng, inputs):
        model_out = eval_fn_fast(params, rng, inputs)
        one_hots = jnn.one_hot(inputs, 256)
        return optax.softmax_cross_entropy(model_out, one_hots).mean()

    eval_and_grad_fn = lambda params, rng, batch: jax.value_and_grad(
        lambda p: compute_loss(p, rng, batch)
    )(params)
    eval_and_grad_fast = jax.jit(eval_and_grad_fn)

    eval_fn_fast(
        params, jax.random.PRNGKey(0), jnp.zeros([batch_size, 256], dtype=jnp.uint8)
    )
    eval_and_grad_fast(
        params, jax.random.PRNGKey(0), jnp.zeros([batch_size, 256], dtype=jnp.uint8)
    )

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    def grad_update(params, opt_state, rng, inputs):
        loss, grad = eval_and_grad_fast(params, rng, inputs)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    grad_update_fast = jax.jit(grad_update)
    grad_update_fast(
        params,
        opt_state,
        jax.random.PRNGKey(0),
        jnp.zeros([batch_size, 256], dtype=jnp.uint8),
    )

    return (
        params,
        jax.random.randint(rng_input, [batch_size, 256], 0, 256, dtype=jnp.uint8),
        eval_fn_fast,
        eval_and_grad_fast,
        opt_state,
        grad_update_fast,
    )


# attn + probability decoder only:

# eval_fn 1.67ms ± 4.58 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# eval_and_grad_fn 4.02 ms ± 9.12 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# batch_size 512:

# eval_fn(params, inputs).block_until_ready()
# 24.8 ms ± 94.5 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# jax.tree_map(lambda x: x.block_until_ready(), eval_and_grad_fn(params, inputs, targets))
# 78.5 ms ± 93.5 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# + feedforward network (no residuals), same large batch size

# eval_fn(params, inputs).block_until_ready()
# 70.6 ms ± 161 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# jax.tree_map(lambda x: x.block_until_ready(), eval_and_grad_fn(params, inputs, targets))
# 215 ms ± 660 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

# jax.tree_map(lambda x: x.block_until_ready(), update_grad(params, opt_state, inputs, targets))
# 215 ms ± 1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# seq_len 256, batch size 256:

# eval_fn(params, inputs).block_until_ready()
# 76.1 ms ± 123 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# jax.tree_map(lambda x: x.block_until_ready(), eval_and_grad_fn(params, inputs, targets))
# 229 ms ± 4.55 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# jax.tree_map(lambda x: x.block_until_ready(), update_grad(params, opt_state, inputs, targets))
# 227 ms ± 1.36 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# With embedding and residuals:

# eval_fn(params, inputs).block_until_ready()
# 76.4 ms ± 207 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# jax.tree_map(lambda x: x.block_until_ready(), eval_and_grad_fn(params, inputs, targets))
# 240 ms ± 1.22 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# jax.tree_map(lambda x: x.block_until_ready(), update_grad(params, opt_state, inputs, targets))
# 241 ms ± 1.26 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# + right shift of inputs:

# eval_fn(params, inputs).block_until_ready()
# 76.8 ms ± 143 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# jax.tree_map(lambda x: x.block_until_ready(), eval_and_grad_fn(params, inputs, targets))
# 243 ms ± 2.32 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# jax.tree_map(lambda x: x.block_until_ready(), update_grad(params, opt_state, inputs, targets))
# 242 ms ± 1.09 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# + causal masking:

# eval_fn(params, inputs).block_until_ready()
# 79.3 ms ± 172 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# jax.tree_map(lambda x: x.block_until_ready(), eval_and_grad_fn(params, inputs, targets))
# 244 ms ± 3.87 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# jax.tree_map(lambda x: x.block_until_ready(), update_grad(params, opt_state, inputs, targets))
# 244 ms ± 1.54 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# + change output dim to 256 for a sketch at autoregressive

# eval_fn(params, inputs).block_until_ready()
# 81 ms ± 139 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# jax.tree_map(lambda x: x.block_until_ready(), eval_and_grad_fn(params, inputs, targets))
# 249 ms ± 1.25 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# jax.tree_map(lambda x: x.block_until_ready(), update_grad(params, opt_state, inputs, targets))
# 249 ms ± 1.42 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# just make it actually autoregressive:

# eval_fn(params, inputs).block_until_ready()
# 81.2 ms ± 152 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# jax.tree_map(lambda x: x.block_until_ready(), eval_and_grad_fn(params, inputs))
# 250 ms ± 947 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

# jax.tree_map(lambda x: x.block_until_ready(), update_grad(params, opt_state, inputs))
# 251 ms ± 1.64 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# add dropout:

# eval_fn(params, rng, inputs).block_until_ready()
# 81.5 ms ± 294 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# jax.tree_map(lambda x: x.block_until_ready(), eval_and_grad_fn(params, rng, inputs))
# 252 ms ± 1.45 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# jax.tree_map(lambda x: x.block_until_ready(), update_grad(params, opt_state, rng, inputs))
# 252 ms ± 1.86 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# add layer norm:

# eval_fn(params, rng, inputs).block_until_ready()
# 81.7 ms ± 230 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# jax.tree_map(lambda x: x.block_until_ready(), eval_and_grad_fn(params, rng, inputs))
# 264 ms ± 1.38 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# jax.tree_map(lambda x: x.block_until_ready(), update_grad(params, opt_state, rng, inputs))
# 264 ms ± 1.28 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# add positional encoding:

# eval_fn(params, rng, inputs).block_until_ready()
# 81.7 ms ± 159 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# jax.tree_map(lambda x: x.block_until_ready(), eval_and_grad_fn(params, rng, inputs))
# 265 ms ± 972 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

# jax.tree_map(lambda x: x.block_until_ready(), update_grad(params, opt_state, rng, inputs))
# 265 ms ± 1.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# list of 1 inner layer:

# eval_fn(params, rng, inputs).block_until_ready()
# 81.8 ms ± 182 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# jax.tree_map(lambda x: x.block_until_ready(), eval_and_grad_fn(params, rng, inputs))
# 265 ms ± 1.53 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# jax.tree_map(lambda x: x.block_until_ready(), update_grad(params, opt_state, rng, inputs))
# 265 ms ± 1.66 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# batch size down to 32:

# eval_fn(params, rng, inputs).block_until_ready()
# 10.6 ms ± 17.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# jax.tree_map(lambda x: x.block_until_ready(), eval_and_grad_fn(params, rng, inputs))
# 32.5 ms ± 135 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# jax.tree_map(lambda x: x.block_until_ready(), update_grad(params, opt_state, rng, inputs))
# 32.9 ms ± 126 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# list of 8 inner layers:

# eval_fn(params, rng, inputs).block_until_ready()
# 81.3 ms ± 204 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# jax.tree_map(lambda x: x.block_until_ready(), eval_and_grad_fn(params, rng, inputs))
# 249 ms ± 1.51 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# jax.tree_map(lambda x: x.block_until_ready(), update_grad(params, opt_state, rng, inputs))
# 252 ms ± 1.24 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
