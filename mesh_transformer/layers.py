import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from mesh_transformer.util import f_psum, g_psum, maybe_shard, head_print
from jax.experimental import PartitionSpec as P
from jax.experimental.maps import thread_resources


class ReplicatedLayerNorm(hk.Module):
    def __init__(self, offset=True):
        super().__init__()
        self.offset = offset

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        variance = jnp.var(inputs, axis=-1, keepdims=True)

        param_shape = inputs.shape[-1:]
        scale = hk.get_parameter("scale", param_shape, inputs.dtype, init=jnp.ones)
        scale = jax.lax.all_gather(scale, "shard")[0]

        offset = hk.get_parameter("offset", param_shape, inputs.dtype, init=jnp.zeros)
        offset = jax.lax.all_gather(offset, "shard")[0]

        scale = jnp.broadcast_to(scale, inputs.shape)
        offset = jnp.broadcast_to(offset, inputs.shape)
        mean = jnp.broadcast_to(mean, inputs.shape)

        inv = scale * jax.lax.rsqrt(variance + 1e-5)
        if self.offset:
            return inv * (inputs - mean) + offset
        else:
            return inv * (inputs - mean)


class ReplicatedDoubleLayerNorm(hk.Module):
    def __init__(self, offset=True):
        super().__init__(name="replicated_layer_norm")
        self.offset = offset

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        variance = jnp.var(inputs, axis=-1, keepdims=True)

        param_shape = inputs.shape[-1:]
        scale = hk.get_parameter("scale", param_shape, inputs.dtype, init=jnp.ones)
        scale = jax.lax.all_gather(scale, "shard")
        shards_per_checkpoint_shard = scale.shape[0] // 2
        mp_index = (jax.lax.axis_index('shard') // shards_per_checkpoint_shard) * shards_per_checkpoint_shard
        scale = scale[mp_index]

        offset = hk.get_parameter("offset", param_shape, inputs.dtype, init=jnp.zeros)
        offset = jax.lax.all_gather(offset, "shard")[mp_index]

        scale = jnp.broadcast_to(scale, inputs.shape)
        offset = jnp.broadcast_to(offset, inputs.shape)
        mean = jnp.broadcast_to(mean, inputs.shape)

        inv = scale * jax.lax.rsqrt(variance + 1e-5)
        if self.offset:
            return inv * (inputs - mean) + offset
        else:
            return inv * (inputs - mean)


class RMSNorm(hk.Module):
    def __init__(self, offset, elementwise):
        super().__init__()
        self.offset = offset
        self.elementwise = elementwise

    def __call__(self, x):
        param_shape = (x.shape[-1],) if self.elementwise else ()
        normed = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-5)

        scale = hk.get_parameter('scale', param_shape, init=hk.initializers.Constant(x.shape[-1] ** 0.5))
        scale = jax.lax.pmean(scale, "shard")
        normed = normed * scale

        if self.offset:
            offset = hk.get_parameter('offset', param_shape, init=jnp.zeros)
            offset = jax.lax.pmean(offset, "shard")
            normed = normed + offset

        return normed


def getnorm(type):
    if type == "layernorm":
        return ReplicatedLayerNorm()
    if type == "layernorm-desync":
        return hk.LayerNorm(-1, True, True)
    elif type == "layernorm-nobias":
        return ReplicatedLayerNorm(offset=False)
    elif type == "doublelayernorm":
        return ReplicatedDoubleLayerNorm()
    elif type == "doublelayernorm-nobias":
        return ReplicatedDoubleLayerNorm(offset=False)
    elif type == "rmsnorm":
        return RMSNorm(False, True)
    elif type == "scalenorm":
        return RMSNorm(False, False)
    elif type == "rmsnorm-bias":
        return RMSNorm(True, True)
    elif type == "scalenorm-bias":
        return RMSNorm(True, False)
    else:
        raise Exception("Not implemented")

def getactfn(type):
    if type in ("gelu_new", "gelu_python"):  # tanh approximation of GELU from https://arxiv.org/pdf/1606.08415.pdf section 2
        return lambda x: jax.nn.gelu(x, approximate=True)
    elif type == "gelu":  # error function formula for GELU from https://arxiv.org/pdf/1606.08415.pdf section 2
        return lambda x: jax.nn.gelu(x, approximate=False)
    elif type == "quick_gelu":  # sigmoid approximation of GELU from https://arxiv.org/pdf/1606.08415.pdf section 2
        return lambda x: x * jax.nn.sigmoid(1.702 * x)
    elif type == "gelu_fast":  # another approximation of GELU proposed by the authors of the same paper
        return lambda x: x * 0.5 * (1.0 + jnp.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
    elif type == "gelu_10":
        return lambda x: jnp.clip(jax.nn.gelu(x, approximate=True), min=-10, max=10)
    elif type == "mish":
        return lambda x: x * jnp.tanh(jax.nn.softplus(x))
    elif type in ("silu", "swish"):
        return jax.nn.silu
    elif type == "relu":
        return jax.nn.relu
    elif type == "sigmoid":
        return jax.nn.sigmoid
    elif type == "tanh":
        return jnp.tanh
    elif type == "linear":
        return lambda x: x
    else:
        raise Exception("Not implemented")


class RelativePositionEmbs(hk.Module):
    @staticmethod
    def _relative_position_bucket(relative_position,
                                  num_buckets=32,
                                  max_distance=128):
        ret = 0
        n = -relative_position
        n = np.maximum(n, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = (n < max_exact)
        val_if_large = max_exact + (
                np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps) /
                np.log(max_distance / max_exact) *
                (num_buckets - max_exact)).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)
        ret += np.where(is_small, n, val_if_large)
        return ret

    def __call__(self, qlen, klen, heads, num_buckets):
        """Produce relative position embedding attention biases.
        Returns:
          output: `(heads, q_len, k_len)` attention bias
        """
        context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
        memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(relative_position)
        relative_attention_bias = hk.get_parameter('rel_embedding', [heads, num_buckets],
                                                   init=hk.initializers.TruncatedNormal(stddev=0.02))
        # Instead of using a slow gather, we create a leading-dimension one-hot
        # array from rp_bucket and use it to perform the gather-equivalent via a
        # contraction, i.e.:
        # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
        # This is equivalent to relative_attention_bias[:, rp_bucket]
        bcast_iota = jax.lax.broadcasted_iota(jnp.int32, (num_buckets, 1, 1), 0)
        rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, Ellipsis] == bcast_iota).astype(
            relative_attention_bias.dtype)
        # --> shape (qlen, klen, num_heads)
        values = jax.lax.dot_general(
            relative_attention_bias,
            rp_bucket_one_hot,
            (
                ((1,), (0,)),  # rhs, lhs contracting dims
                ((), ())))  # no batched dims
        return values


class TransposingLinear(hk.Module):
    def __init__(self, input_size, output_size, with_bias=True, w_init=None, b_init=None, name=None):
        if name is None:
            name = "linear"
        super().__init__(name=name)
        self.input_size = input_size
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros

    def __call__(self, inputs: jnp.ndarray, *, precision=None, transpose_weights=False) -> jnp.ndarray:
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1. / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

        if transpose_weights:
            w = w.T
        out = jnp.dot(inputs, w, precision=precision)

        if self.with_bias:
            b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        return out


class AllReduceLinear(hk.Module):
    def __init__(self, output_size, with_bias=True, w_init=None, b_init=None, name=None, all_reduce=False, shards=None):
        if all_reduce and shards is None:
            raise ValueError("Shards must be specified if `all_reduce` is true")
        if name is None:
            name = "linear"
        super().__init__(name=name)
        self.input_size = None
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros
        self.all_reduce = all_reduce
        self.shards = shards

    def __call__(self, inputs: jnp.ndarray, *, precision=None) -> jnp.ndarray:
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1. / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

        out = jnp.dot(inputs, w, precision=precision)

        if self.all_reduce:
            out = g_psum(out)

        if self.with_bias:
            b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
            if self.all_reduce:
                b *= self.shards
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        return out


def fixed_pos_embedding(x, seq_dim=0, shift=0, neox=False):
    dim = x.shape[-1]
    inv_freq = 1. / (10000 ** (np.arange(0, dim, 2) / dim))

    sinusoid_inp = np.einsum('i , j -> i j', np.arange(shift, x.shape[seq_dim] + shift), inv_freq)
    if neox:
        sinusoid_inp = np.concatenate((sinusoid_inp, sinusoid_inp), axis=-1)

    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]

    x = jnp.stack((-x2, x1), axis=-1)

    return x.reshape(*x.shape[:-2], -1)


def rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(x, sincos, neox=False):
    sin, cos = map(lambda t: (t if neox else t.repeat(2, axis=-1))[-x.shape[0]:, None, :], sincos)
    return (x * cos) + ((rotate_half if neox else rotate_every_two)(x) * sin)


def rotate_every_two_v2(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]

    x = jnp.stack((-x2, x1), axis=-1)

    return x.reshape(*x.shape[:-2], -1)


def apply_rotary_pos_emb_v2(x, sincos):
    sin, cos = map(lambda t: t.repeat(2, axis=-1)[-x.shape[-3]:, None, :], sincos)
    return (x * cos) + (rotate_every_two_v2(x) * sin)


def create_alibi_tensor(heads: int, heads_per_shard: int, k_length: int):
    slopes = (2 ** (-(2 ** -(jnp.log2(heads) - 3)))) ** (1 + heads_per_shard*jax.lax.axis_index("shard") + jnp.arange(heads_per_shard))  # shape: (heads_per_shard,)
    tensor = jnp.arange(k_length)  # shape: (k_length,)
    tensor = jnp.outer(slopes, tensor)  # shape: (heads_per_shard, k_length)
    tensor = tensor[:, jnp.newaxis, :]  # shape: (heads_per_shard, 1, k_length)
    return tensor


class EmbeddingShard(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        in_dim = config["n_vocab"] + config.get("n_vocab_padding", 0)
        out_dim = config["d_model"]
        shards = config["cores_per_replica"]
        self.compat = config.get("compat", "j")
        self.pe_shift = config.get("pe_shift", 2 if self.compat in ("fairseq_lm",) else 0)

        assert in_dim % shards == 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_dim_per_shard = in_dim // shards
        self.out_dim_per_shard = out_dim // shards
        self.post_embed = config["pe"] in ("fairseq_sinusoidal", "sinusoidal")
        self.has_sqrt_embed_scale = self.compat in ("fairseq_lm",)

        self.d_embed = config.get("d_embed", self.out_dim)

        self.embedding_dropout = config.get("embedding_dropout", 0.0)

        if config["pe"] == "fixed":
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            self.positional_embeddings = hk.get_parameter('pos_embs', [config["seq"], self.out_dim_per_shard], init=embed_init)
        elif config["pe"] == "sinusoidal":  # Sinusoidal positional embedding exactly as described in section 3.5 of https://arxiv.org/pdf/1706.03762.pdf
            assert out_dim % 2 == 0
            sincos = fixed_pos_embedding(jnp.empty((config["seq"], out_dim)), shift=self.pe_shift)
            self.positional_embeddings = jnp.stack(sincos, axis=2).reshape((config["seq"], -1))
        elif config["pe"] == "fairseq_sinusoidal":  # A slightly incorrect version of sinusoidal positional embedding used by fairseq
            assert out_dim % 2 == 0
            sincos = fixed_pos_embedding(jnp.empty((config["seq"], out_dim)), shift=self.pe_shift)
            self.positional_embeddings = jnp.concatenate(sincos, axis=-1)
        else:
            self.positional_embeddings = None

        self.proj = TransposingLinear(self.in_dim_per_shard, self.d_embed, w_init=hk.initializers.TruncatedNormal(stddev=1 / np.sqrt(in_dim)), with_bias=self.compat not in ("neo", "fairseq_lm", "neox", "opt", "bloom"))

        if self.compat == "bloom":
            self.norm = getnorm(config["norm"])

        if self.d_embed != self.out_dim:
            self.project_in = jnp.concatenate(jax.lax.all_gather(hk.get_parameter("project_in", [self.d_embed, self.out_dim_per_shard], init=hk.initializers.TruncatedNormal(stddev=1 / np.sqrt(self.d_embed))), "shard"), axis=-1)
        else:
            self.project_in = None

    def __call__(self, x, dtype=jnp.bfloat16, pe_length=0, soft_embeddings=None):
        pe_length = jnp.int32(pe_length)
        shard_start_index = jax.lax.axis_index('shard') * self.in_dim_per_shard

        input_onehot = jax.nn.one_hot(x - shard_start_index, self.in_dim_per_shard)
        proj_out = self.proj(input_onehot)

        mask = jnp.broadcast_to((x < self.in_dim)[:, jnp.newaxis], proj_out.shape)
        proj_out = jnp.where(mask, proj_out, 0)

        if soft_embeddings is not None:
            assert soft_embeddings.ndim == 2
            assert soft_embeddings.shape[1] == getattr(self, "d_embed", self.out_dim)

            soft_shard_start_index = self.in_dim + jax.lax.axis_index('shard') * soft_embeddings.shape[0]

            input_soft_onehot = jax.nn.one_hot(x - soft_shard_start_index, soft_embeddings.shape[0])
            proj_out += jnp.dot(input_soft_onehot, soft_embeddings)

        if self.has_sqrt_embed_scale:
            proj_out *= jnp.sqrt(self.out_dim).astype(proj_out.dtype)

        if self.project_in is not None:
            proj_out @= self.project_in

        if not self.post_embed and self.positional_embeddings is not None:
            shard_roll_index = jnp.int32(jax.lax.axis_index('shard') * self.out_dim_per_shard)
            pos_embed = jnp.pad(self.positional_embeddings, ((0, 0), (0, self.out_dim - self.out_dim_per_shard)))
            pos_embed = jnp.roll(pos_embed, shard_roll_index, axis=1)
            pos_embed = jnp.roll(pos_embed, -pe_length - self.pe_shift, axis=0)[-proj_out.shape[0]:]
            proj_out += pos_embed

        proj_out = g_psum(proj_out)

        if self.post_embed:
            pos_embed = self.positional_embeddings
            pos_embed = jnp.roll(pos_embed, -pe_length, axis=0)[-proj_out.shape[0]:]
            proj_out += pos_embed

        if 0.0 < self.embedding_dropout <= 1.0:
            proj_out = hk.dropout(hk.next_rng_key(), self.embedding_dropout, proj_out)

        if self.compat == "bloom":
            proj_out = f_psum(proj_out)
            proj_out = self.norm(proj_out)

        return proj_out


class EmbeddingShardV2(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        in_dim = config["n_vocab"]
        out_dim = config["d_model"]
        shards = config["cores_per_replica"]

        assert in_dim % shards == 0

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.proj = hk.Linear(self.out_dim, w_init=hk.initializers.TruncatedNormal(stddev=1 / np.sqrt(in_dim)))

    def __call__(self, x, dtype=jnp.bfloat16):
        input_onehot = jax.nn.one_hot(x, self.in_dim)
        input_onehot = maybe_shard(input_onehot, P("dp", None, "mp"))

        proj_out = self.proj(input_onehot)

        return proj_out


# We actually combine the FF and dense in one layer (i.e. compute in parallel) to minimize all reduces
class TransformerLayerShard(hk.Module):
    def __init__(self, config, name=None, init_scale=1., attention_type="global"):
        super().__init__(name=name)
        heads = config["n_heads"]
        dim = config["d_model"]
        shards = config["cores_per_replica"]
        norm = getnorm(config["norm"])
        self.is_rotary = config["pe"] in ("rotary", "neox_rotary")
        self.is_neox_rotary = config["pe"] == "neox_rotary"
        self.attention_type = attention_type
        self.local_attention_window = config.get("local_attention_window", 256)
        self.compat = config.get("compat", "j")
        self.pe_shift = config.get("pe_shift", 2 if self.compat in ("fairseq_lm",) else 0)
        self.activation_fn = getactfn(config.get("activation", "relu" if self.compat in ("opt",) else "gelu" if self.compat in ("fairseq_lm",) else "gelu_fast" if self.compat in ("neox", "bloom") else "gelu_new"))
        self.neox_gpt_j_residual = self.compat == "neox" and config.get("neox_gpt_j_residual", True)
        self.use_combined_qkv = config.get("combined_qkv", self.compat in ("neox", "bloom"))
        self.early_all_reduce = self.compat == "neox" and not self.neox_gpt_j_residual
        self.do_layer_norm_before = config.get("do_layer_norm_before", True)

        self.attention_dropout = config.get("attention_dropout", 0.0)
        self.residual_dropout = config.get("residual_dropout", 0.0)

        assert dim % heads == 0
        assert heads % shards == 0
        assert attention_type in ("global", "local")

        if config["pe"] == "alibi":
            # For the sake of simplicity, if ALiBi is enabled, we require the number of attention heads to be a power of two
            assert (heads & (heads - 1)) == 0

        self.dim = dim
        self.dim_per_head = dim // heads
        self.heads_per_shard = heads // shards
        self.dim_per_shard = dim // shards
        self.pe_rotary_dims = int(config["pe_rotary_pct"] * self.dim_per_head) if "pe_rotary_pct" in config and 0 <= config["pe_rotary_pct"] <= 1 else config.get("pe_rotary_dims", self.dim_per_head)

        self.norm = norm
        if self.compat != "j":
            self.norm_2 = getnorm(config["norm"])

        if self.use_combined_qkv:
            self.qkv = hk.Linear(self.dim_per_shard * 3, with_bias=self.compat in ("fairseq_lm", "neox", "opt", "bloom"), name="combined_qkv")
        else:
            self.q = hk.Linear(self.dim_per_shard, with_bias=self.compat in ("fairseq_lm", "neox", "opt", "bloom"), name="linear")
            self.v = hk.Linear(self.dim_per_shard, with_bias=self.compat in ("fairseq_lm", "neox", "opt", "bloom"), name="linear_1")
            self.k = hk.Linear(self.dim_per_shard, with_bias=self.compat in ("fairseq_lm", "neox", "opt", "bloom"), name="linear_2")

        self.o = AllReduceLinear(self.dim, with_bias=self.compat in ("neo", "fairseq_lm", "neox", "opt", "bloom"),
                                 w_init=hk.initializers.TruncatedNormal(stddev=init_scale / np.sqrt(self.dim)),
                                 name="linear_3",
                                 all_reduce=self.early_all_reduce, shards=shards)

        self.dense_proj = hk.Linear(self.dim_per_shard * 4, name="linear_4")
        self.dense_proj_o = AllReduceLinear(self.dim,
                                            w_init=hk.initializers.TruncatedNormal(stddev=init_scale / np.sqrt(self.dim)),
                                            name="linear_5",
                                            all_reduce=self.early_all_reduce, shards=shards)

    def self_attn(self, q, v, k, attn_bias):
        if self.is_rotary:
            k_rot = k[:, :, :self.pe_rotary_dims]
            k_pass = k[:, :, self.pe_rotary_dims:]

            q_rot = q[:, :, :self.pe_rotary_dims]
            q_pass = q[:, :, self.pe_rotary_dims:]

            sincos = fixed_pos_embedding(k_rot, shift=self.pe_shift, neox=self.is_neox_rotary)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, neox=self.is_neox_rotary)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, neox=self.is_neox_rotary)

            k = jnp.concatenate([k_rot, k_pass], axis=-1)
            q = jnp.concatenate([q_rot, q_pass], axis=-1)

        attention_logits = jnp.einsum("thd,Thd->htT", q, k)

        if self.compat not in ("neo", "fairseq_lm", "opt"):
            sqrt_key_size = np.sqrt(self.dim_per_head).astype(k.dtype)
            attention_logits = attention_logits / sqrt_key_size

        attention_logits += attn_bias

        attention_weights = jax.nn.softmax(attention_logits)
        if 0.0 < self.attention_dropout <= 1.0:
            attention_weights = hk.dropout(hk.next_rng_key(), self.attention_dropout, attention_weights)
        attention_vec = jnp.einsum("htT,Thd->thd", attention_weights, v).reshape((-1, self.dim_per_shard))

        attention_out = self.o(attention_vec)
        if 0.0 < self.residual_dropout <= 1.0:
            attention_out = hk.dropout(hk.next_rng_key(), self.residual_dropout, attention_out)
        return attention_out

    def ff(self, x):
        dense_proj = self.dense_proj(x)
        dense_proj = self.activation_fn(dense_proj)
        return self.dense_proj_o(dense_proj)

    def qvk_proj(self, x):
        if self.use_combined_qkv:
            m = self.qkv(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head * 3))
            q, k, v = jnp.split(m, 3, axis=-1)
        else:
            q = self.q(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))
            v = self.v(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))
            k = self.k(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))

        if self.compat in ("fairseq_lm", "opt"):
            q /= jnp.sqrt(self.dim_per_head).astype(q.dtype)

        return q, v, k

    def neo_ff(self, x):
        if self.do_layer_norm_before:
            x = f_psum(x)
            x = self.norm_2(x)
        dense_out = self.ff(x)
        if not self.early_all_reduce:
            dense_out = g_psum(dense_out)
        return dense_out

    def __call__(self, x, attn_bias):
        x_original = x
        if self.do_layer_norm_before:
            x = f_psum(x)
            x = self.norm(x)

        q, v, k = self.qvk_proj(x)

        seq_len = x.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        if self.attention_type == "local":
            causal_mask -= np.tril(causal_mask, -self.local_attention_window)

        bias = -1e10 * (1. - causal_mask)
        bias += attn_bias

        attn_out = self.self_attn(q, v, k, bias)
        if not self.neox_gpt_j_residual and self.compat != "j":
            out = attn_out
        elif self.neox_gpt_j_residual:
            x2 = x_original
            if self.do_layer_norm_before:
                x2 = f_psum(x2)
                x2 = self.norm_2(x2)
            dense_out = self.ff(x2)
            out = attn_out + dense_out
        else:
            dense_out = self.ff(x)
            out = attn_out + dense_out

        if not self.early_all_reduce:
            out = g_psum(out)
        return out

    # iterate the decoding process by a single token
    def decode_once(self, decode_state, x, attn_bias):
        x_original = x
        if self.do_layer_norm_before:
            x = f_psum(x)
            x = self.norm(x)

        assert x.shape[0] == 1

        q, v, k = self.qvk_proj(x)

        # add new kv to end
        v = jnp.concatenate((decode_state["v"], v), axis=0)[1:]
        k = jnp.concatenate((decode_state["k"], k), axis=0)[1:]

        tokens_decoded = decode_state["tokens_decoded"] + 1
        length = v.shape[0]

        if self.attention_type == "local":
            masked_tokens = length - jnp.minimum(tokens_decoded, self.local_attention_window)
        else:
            masked_tokens = length - tokens_decoded

        attention_mask = jnp.arange(0, length) < masked_tokens
        bias = (-1e10 * attention_mask)
        bias += attn_bias

        attn_out = self.self_attn(q, v, k, bias)
        if not self.neox_gpt_j_residual and self.compat != "j":
            out = attn_out
        elif self.neox_gpt_j_residual:
            x2 = x_original
            if self.do_layer_norm_before:
                x2 = f_psum(x2)
                x2 = self.norm_2(x2)
            dense_out = self.ff(x2)
            out = attn_out + dense_out
        else:
            dense_out = self.ff(x)
            out = attn_out + dense_out

        if not self.early_all_reduce:
            out = g_psum(out)
        return out, {
            "tokens_decoded": tokens_decoded,
            "k": k,
            "v": v
        }

    # take in right aligned context tokens and generate an initial state
    def get_init_decode_state(self, x, given_length, attn_bias):
        x_original = x
        if self.do_layer_norm_before:
            x = f_psum(x)
            x = self.norm(x)

        q, v, k = self.qvk_proj(x)

        full_length = x.shape[0]
        masked_tokens = full_length - given_length

        seq_len = x.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        if self.attention_type == "local":
            causal_mask -= np.tril(causal_mask, -self.local_attention_window)

        bias = -1e10 * (1. - causal_mask)  # regular AR masking
        bias -= 1e10 * (jnp.arange(0, full_length) < masked_tokens)  # mask out zero tokens before context starts
        bias += attn_bias  # finally add attn bias for rpe

        attn_out = self.self_attn(q, v, k, bias)
        if not self.neox_gpt_j_residual and self.compat != "j":
            out = attn_out
        elif self.neox_gpt_j_residual:
            x2 = x_original
            if self.do_layer_norm_before:
                x2 = f_psum(x2)
                x2 = self.norm_2(x2)
            dense_out = self.ff(x2)
            out = attn_out + dense_out
        else:
            dense_out = self.ff(x)
            out = attn_out + dense_out

        if not self.early_all_reduce:
            out = g_psum(out)
        return out, {"k": k, "v": v, "tokens_decoded": given_length.astype(jnp.uint32)}


# This new class combines the input and output projection into one matmul for better efficiency
class TransformerLayerShardV2(hk.Module):
    def __init__(self, config, name=None, init_scale=1.):
        super().__init__(name=name)
        self.dim = config["d_model"]
        self.n_head = config["n_heads"]
        self.d_head = config["d_head"]
        self.d_rotary = config["pe_rotary_dims"]
        self.mp_num = thread_resources.env.shape['mp']

        self.norm = hk.LayerNorm(-1, True, True)
        self.input_proj = hk.Linear(self.d_head * self.n_head * 3 + self.dim * 8)
        self.output_proj = hk.Linear(self.dim,
                                     w_init=hk.initializers.TruncatedNormal(stddev=init_scale / jnp.sqrt(self.dim)))

    def self_attn(self, q, v, k, attn_bias):
        k_rot = k[:, :, :, :self.d_rotary]
        k_pass = k[:, :, :, self.d_rotary:]

        q_rot = q[:, :, :, :self.d_rotary]
        q_pass = q[:, :, :, self.d_rotary:]

        sincos = fixed_pos_embedding(k_rot, seq_dim=1)
        q_rot = apply_rotary_pos_emb_v2(q_rot, sincos)
        k_rot = apply_rotary_pos_emb_v2(k_rot, sincos)
        q_rot = maybe_shard(q_rot, P("dp", None, "mp", None))
        k_rot = maybe_shard(k_rot, P("dp", None, "mp", None))

        k = jnp.concatenate([k_rot, k_pass], axis=-1)
        q = jnp.concatenate([q_rot, q_pass], axis=-1)

        k = maybe_shard(k, P("dp", None, "mp", None))
        q = maybe_shard(q, P("dp", None, "mp", None))

        attention_logits = jnp.einsum("bthd,bThd->bhtT", q, k)

        attention_logits = maybe_shard(attention_logits, P("dp", "mp", None, None))

        sqrt_key_size = np.sqrt(self.d_head).astype(k.dtype)
        attention_logits = attention_logits / sqrt_key_size

        attention_logits += attn_bias
        attention_logits = maybe_shard(attention_logits, P("dp", "mp", None, None))

        attention_weights = jax.nn.softmax(attention_logits)
        attention_weights = maybe_shard(attention_weights, P("dp", "mp", None, None))

        attention_vec = jnp.einsum("bhtT,bThd->bthd", attention_weights, v)

        attention_vec = maybe_shard(attention_vec, P("dp", None, "mp", None))
        sharded_attn_vec = attention_vec.reshape(attention_vec.shape[:2] + (self.mp_num, self.n_head//self.mp_num, -1))
        sharded_attn_vec = maybe_shard(sharded_attn_vec, P("dp", None, "mp", None, None))

        attention_vec = attention_vec.reshape(sharded_attn_vec.shape[:2] + (self.mp_num, -1))
        return maybe_shard(attention_vec, P("dp", None, "mp", None))

    # input: [batch, seq, dim]
    # output: [batch, seq, n_head, d_head]
    def head_split(self, x):
        reshaped = x.reshape(x.shape[:-1] + (self.n_head//self.mp_num, self.d_head))
        reshaped = reshaped.reshape(x.shape[:-2] + (-1, ) + x.shape[-1:])

        # return reshaped
        return maybe_shard(reshaped, P("dp", None, "mp", None))

    def input(self, x):
        # [batch, seq, dim]
        projected = self.input_proj(x)

        # [batch, seq, mp, dim//mp]
        projected = maybe_shard(projected, P("dp", None, "mp"))
        mp_split = jnp.reshape(projected, projected.shape[:-1] + (self.mp_num, -1))
        mp_split = maybe_shard(mp_split, P("dp", None, "mp", None))

        local_dim = self.d_head * self.n_head // self.mp_num

        q, v, k, ff = jnp.split(mp_split, [local_dim, local_dim * 2, local_dim * 3], axis=-1)

        q = self.head_split(q)
        v = self.head_split(v)
        k = self.head_split(k)

        return q, v, k, ff

    def output(self, *x):
        out = jnp.concatenate(x, axis=-1)
        out = maybe_shard(out, P("dp", None, "mp", None))

        out = out.reshape(x[0].shape[:-2] + (-1,))
        out_shard = maybe_shard(out, P("dp", None, "mp"))

        return self.output_proj(out_shard)

    def __call__(self, x, attn_bias):

        x = self.norm(x)

        q, v, k, ff = self.input(x)

        # head_print("x.shape", x.shape)
        # head_print("attn_bias.shape", attn_bias.shape)

        seq_len = x.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))[None, :, :]
        bias = -1e10 * (1. - causal_mask)

        # head_print("bias.shape", bias.shape)

        bias += attn_bias

        attn_out = self.self_attn(q, v, k, bias)
        ff_out = self.glu(ff)

        return self.output(attn_out, ff_out)

    # [batch, seq, mp, dim*2//mp]
    def glu(self, x):
        out, gate = jnp.split(x, 2, axis=-1)

        return out * jax.nn.gelu(gate)

    # iterate the decoding process by a single token
    def decode_once(self, decode_state, x, attn_bias):
        x = self.norm(x)

        assert x.shape[0] == 1

        q, v, k, ff = self.input(x)

        # add new kv to end
        v = jnp.concatenate((decode_state["v"], v), axis=1)[1:]
        k = jnp.concatenate((decode_state["k"], k), axis=1)[1:]

        tokens_decoded = decode_state["tokens_decoded"] + 1
        length = v.shape[1]

        masked_tokens = length - tokens_decoded

        attention_mask = jnp.arange(0, length) < masked_tokens
        bias = (-1e10 * attention_mask)
        bias += attn_bias

        attn_out = self.self_attn(q, v, k, bias)
        ff_out = self.glu(ff)

        return self.output(attn_out, ff_out), {
            "tokens_decoded": tokens_decoded,
            "k": k,
            "v": v
        }

    # take in right aligned context tokens and generate an initial state
    def get_init_decode_state(self, x, given_length, attn_bias):
        x = self.norm(x)

        q, v, k, ff = self.input(x)

        full_length = x.shape[1]
        masked_tokens = full_length - given_length

        causal_mask = np.tril(np.ones((full_length, full_length)))

        bias = -1e10 * (1. - causal_mask)  # regular AR masking
        bias -= 1e10 * (jnp.arange(0, full_length) < masked_tokens)  # mask out zero tokens before context starts
        bias += attn_bias  # finally add attn bias for rpe

        attn_out = self.self_attn(q, v, k, bias)
        ff_out = self.glu(ff)

        return self.output(attn_out, ff_out), {
            "tokens_decoded": given_length.astype(jnp.uint32),
            "k": k,
            "v": v,
        }


class ProjectionShard(hk.Module):
    def __init__(self, config, name=None, embedding_shard=None):
        super().__init__(name=name)
        self.out_dim_unpadded = config["n_vocab"]
        out_dim = self.out_dim_unpadded + config.get("n_vocab_padding", 0)
        shards = config["cores_per_replica"]
        self.compat = config.get("compat", "j")
        self.do_layer_norm_before = config.get("do_layer_norm_before", True)
        if self.do_layer_norm_before or self.compat != "opt":
            norm = getnorm(config["norm"])

        assert out_dim % shards == 0

        self.in_dim = config["d_model"]
        self.dim = out_dim
        self.dim_per_shard = out_dim // shards

        if self.do_layer_norm_before or self.compat != "opt":
            self.norm = norm

        self.d_embed = config.get("d_embed", self.in_dim)
        assert self.d_embed % shards == 0

        if self.compat in ("neo", "fairseq_lm", "opt", "bloom"):
            self.proj = embedding_shard.proj
        else:
            self.proj = TransposingLinear(config["d_model"], self.dim_per_shard, with_bias=self.compat not in ("neo", "fairseq_lm", "neox", "opt", "bloom"))

        if self.d_embed != self.in_dim:
            self.project_out = jnp.concatenate(jax.lax.all_gather(hk.get_parameter("project_out", [self.in_dim, self.d_embed // shards], init=hk.initializers.TruncatedNormal(stddev=1 / np.sqrt(self.in_dim))), "shard"), axis=-1)
        else:
            self.project_out = None

    def __call__(self, x):
        if self.do_layer_norm_before or self.compat != "opt":
            x = self.norm(x)
        if self.project_out is not None:
            x @= self.project_out
        proj = self.proj(x, transpose_weights=self.compat in ("neo", "fairseq_lm", "opt", "bloom"))

        all_proj = jax.lax.all_gather(proj, 'shard')

        return hk.Flatten()(jnp.transpose(all_proj, (1, 0, 2)))[:, :self.out_dim_unpadded]

    def loss(self, x, targets, z_loss=1):
        if self.do_layer_norm_before or self.compat != "opt":
            x = f_psum(x)
            x = self.norm(x)
        if self.project_out is not None:
            x @= self.project_out
        logits = self.proj(x, transpose_weights=self.compat in ("neo", "fairseq_lm", "opt", "bloom"))

        shard_start_index = jax.lax.axis_index('shard') * self.dim_per_shard

        vocab_mask = targets < self.out_dim_unpadded
        logit_mask = jnp.arange(self.dim_per_shard) + shard_start_index < self.out_dim_unpadded
        logits = jnp.where(logit_mask, logits, -1e9)

        global_max = jax.lax.pmax(jax.lax.stop_gradient(logits.max(-1, keepdims=True, initial=-jnp.inf, where=logit_mask)), "shard")
        logits -= jax.lax.stop_gradient(global_max)

        gt_onehot = jax.nn.one_hot(targets - shard_start_index, self.dim_per_shard)
        predicted_logits = jnp.sum(jnp.multiply(gt_onehot, logits), axis=-1)
        predicted_logits = g_psum(predicted_logits)

        exp_logits = jnp.exp(logits)

        sum_exp_logits = exp_logits.sum(axis=-1)
        sum_exp_logits = g_psum(sum_exp_logits)

        loss = jnp.log(sum_exp_logits) - predicted_logits

        loss += (1e-4 * jnp.square(jnp.log(sum_exp_logits)) * z_loss).sum() / vocab_mask.sum()

        correct = (0.0 == predicted_logits)

        return vocab_mask * loss, vocab_mask * correct


class Projection(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        out_dim = config["n_vocab"]

        self.dim = out_dim
        self.norm = hk.LayerNorm(-1, True, True)

        self.proj = hk.Linear(self.dim)

    def __call__(self, x):
        x = self.norm(x)
        return self.proj(x)

    def loss(self, x, targets, z_loss=1):
        x = self.norm(x)
        logits = self.proj(x)

        logits -= logits.max(-1, keepdims=True)

        gt_onehot = jax.nn.one_hot(targets, self.dim)
        predicted_logits = jnp.sum(jnp.multiply(gt_onehot, logits), axis=-1)
        exp_logits = jnp.exp(logits)

        sum_exp_logits = exp_logits.sum(axis=-1)

        loss = jnp.log(sum_exp_logits) - predicted_logits

        loss += (1e-4 * jnp.square(jnp.log(sum_exp_logits)) * z_loss).mean()
        correct = (0.0 == predicted_logits)
        return loss, correct
