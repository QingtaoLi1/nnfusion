import torch
import torch.nn as nn
import os
from custom_op import CustomOp


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float().to(device) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)   # [MaxSeqLen, Dim]
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)   # [MaxSeqLen, Dim]

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, v, kv_seq_len, position_ids, unsqueeze_dim=1):
        # v: [bs, num_attention_heads, kv_seq_len, head_size]
        # if kv_seq_len > self.max_seq_len_cached:
        #     self._set_cos_sin_cache(seq_len=kv_seq_len, device=v.device, dtype=v.dtype)

        cos = self.cos_cached[:kv_seq_len].to(dtype=v.dtype)
        sin = self.sin_cached[:kv_seq_len].to(dtype=v.dtype)

        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed


class FusedLlamaRotaryEmbeddingFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, kv_seq_len, position_ids, unsqueeze_dim, cos_cached, sin_cached):
        welder_arch = os.environ["WELDER_ARCH"] if "WELDER_ARCH" in os.environ else "A100"
        head_dim = q.shape[-1]
        fused_q_op = CustomOp(ir=f'''
m0[S, D] = input2[S, D] where S in {kv_seq_len};
m1[S, D] = m0[input1[S], D];
m2[S, D] = input3[S, D] where S in {kv_seq_len};
m3[S, D] = m2[input1[S], D];

m4[S, H, D] = (-input0[S, H, D + {head_dim // 2}]).when([D < {head_dim // 2}], input0[S, H, D - {head_dim // 2}]) where D in {head_dim};
output0[S, H, D] = input0[S, H, D] * m1[S, D] + m4[S, H, D] * m3[S, D];
''', input_orders={'input0': q, 'input1': position_ids, 'input2': cos_cached, 'input3': sin_cached}, device=device, arch=welder_arch)
        q_embed = fused_q_op([q, position_ids, cos_cached, sin_cached])

        fused_k_op = CustomOp(ir=f'''
m0[MaxS, D] = input2[MaxS, D] where MaxS in {kv_seq_len};
m1[S, D] = m0[input1[S], D];
m2[MaxS, D] = input3[MaxS, D] where MaxS in {kv_seq_len};
m3[S, D] = m2[input1[S], D];

m5[S, H, D] = (-input0[S, H, D + {head_dim // 2}]).when([D < {head_dim // 2}], input0[S, H, D - {head_dim // 2}]) where D in {head_dim};
output0[S, H, D] = input0[S, H, D] * m1[S, D] + m5[S, H, D] * m3[S, D];
''', input_orders={'input0': k, 'input1': position_ids, 'input2': cos_cached, 'input3': sin_cached}, device=device, arch=welder_arch)
        k_embed = fused_k_op([k, position_ids, cos_cached, sin_cached])

        ctx.save_for_backward(position_ids, cos_cached, sin_cached)
        ctx.kv_seq_len = kv_seq_len
        ctx.head_dim = head_dim
        return q_embed, k_embed
    
    @staticmethod
    def backward(ctx, dq_embed, dk_embed):
        welder_arch = os.environ["WELDER_ARCH"] if "WELDER_ARCH" in os.environ else "A100"
        position_ids, cos_cached, sin_cached = ctx.saved_tensors
        kv_seq_len = ctx.kv_seq_len
        head_dim = ctx.head_dim
        dq_op = CustomOp(ir=f'''
m0[MaxS, D] = input2[MaxS, D] where MaxS in {kv_seq_len};
m1[S, D] = m0[input1[S], D];
m2[MaxS, D] = input3[MaxS, D] where MaxS in {kv_seq_len};
m3[S, D] = m2[input1[S], D];

m10[S, H, D] = input0[S, H, D].cast(`float32`);
di0a[S, H, D] = m10[S, H, D] * m1[S, D];
dm4[S, H, D] = m10[S, H, D] * m3[S, D];
di0b[S, H, D] = dm4[S, H, D + {head_dim // 2}].when([D < {head_dim // 2}], (-dm4[S, H, D - {head_dim // 2}])) where D in {head_dim};
output0[S, H, D] = di0a[S, H, D] + di0b[S, H, D];
''', input_orders={'input0': dq_embed, 'input1': position_ids, 'input2': cos_cached, 'input3': sin_cached}, device=device, arch=welder_arch)
        dq = dq_op([dq_embed, position_ids, cos_cached, sin_cached])

        dk_op = CustomOp(ir=f'''
m0[MaxS, D] = input2[MaxS, D] where MaxS in {kv_seq_len};
m1[S, D] = m0[input1[S], D];
m2[MaxS, D] = input3[MaxS, D] where MaxS in {kv_seq_len};
m3[S, D] = m2[input1[S], D];

m11[S, H, D] = input0[S, H, D].cast(`float32`);
di1a[S, H, D] = m11[S, H, D] * m1[S, D];
dm5[S, H, D] = m11[S, H, D] * m3[S, D];
di1b[S, H, D] = dm5[S, H, D + {head_dim // 2}].when([D < {head_dim // 2}], (-dm5[S, H, D - {head_dim // 2}])) where D in {head_dim};
output0[S, H, D] = di1a[S, H, D] + di1b[S, H, D];
''', input_orders={'input0': dk_embed, 'input1': position_ids, 'input2': cos_cached, 'input3': sin_cached}, device=device, arch=welder_arch)
        dk = dk_op([dk_embed, position_ids, cos_cached, sin_cached])

        return dq, dk, None, None, None, None, None, None

class FusedLlamaRotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float().to(device) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, q, k, v, kv_seq_len, position_ids, unsqueeze_dim=1):
        return FusedLlamaRotaryEmbeddingFunc.apply(q, k, v, kv_seq_len, position_ids, unsqueeze_dim, self.cos_cached, self.sin_cached)
    
