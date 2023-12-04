import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from custom_op import CustomOp


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
        print ({'cos_cached': self.cos_cached.shape, 'sin_cached': self.sin_cached.shape})

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
        print ("WARNING!!! only support unsqueeze_dim = 1 !!!")
        print ("WARNING!!! only support head_dim % 2 == 0 !!!")
        print ("q\t: ", q.shape)                        # [Seq, Head, HeadDim]
        print ("k\t: ", k.shape)                        # [Seq, Head, HeadDim]
        print ("v\t: ", v.shape)                        # [Seq, Head, HeadDim]
        print ("kv_seq_len\t: ", kv_seq_len)            # scalar
        print ("position_ids\t: ", position_ids.shape)  # [Seq]
        print ("unsqueeze_dim\t: ", unsqueeze_dim)      # scalar
        print ("cos_cached\t: ", cos_cached.shape)      # [MaxSeqLen, Dim]
        print ("sin_cached\t: ", sin_cached.shape)      # [MaxSeqLen, Dim]

        head_dim = q.shape[-1]
        fused_op = CustomOp(ir=f'''
m0[MaxS, D] = input3[MaxS, D] where MaxS in {kv_seq_len};
m1[S, D] = m0[input2[S], D];
m2[MaxS, D] = input4[MaxS, D] where MaxS in {kv_seq_len};
m3[S, D] = m2[input2[S], D];
m4[S, H, D] = (-input0[S, H, D + {head_dim // 2}]).when([D < {head_dim // 2}], input0[S, H, D - {head_dim // 2}]) where D in {head_dim};
m5[S, H, D] = (-input1[S, H, D + {head_dim // 2}]).when([D < {head_dim // 2}], input1[S, H, D - {head_dim // 2}]) where D in {head_dim};
output0[S, H, D] = input0[S, H, D] * m1[S, D] + m4[S, H, D] * m3[S, D];
output1[S, H, D] = input1[S, H, D] * m1[S, D] + m5[S, H, D] * m3[S, D];
''', input_orders={'input0': q, 'input1': k, 'input2': position_ids, 'input3': cos_cached, 'input4': sin_cached}, device=device, arch="A100")
        q_embed, k_embed = fused_op([q, k, position_ids, cos_cached, sin_cached])
        print ("q_embed\t: ", q_embed.shape)

        ctx.save_for_backward(position_ids, cos_cached, sin_cached)
        ctx.kv_seq_len = kv_seq_len
        ctx.head_dim = head_dim
        return q_embed, k_embed
    
    @staticmethod
    def backward(ctx, dq_embed, dk_embed):
        position_ids, cos_cached, sin_cached = ctx.saved_tensors
        kv_seq_len = ctx.kv_seq_len
        head_dim = ctx.head_dim
        dqk_op = CustomOp(ir=f'''
m0[MaxS, D] = input3[MaxS, D] where MaxS in {kv_seq_len};
m1[S, D] = m0[input2[S], D];
m2[MaxS, D] = input4[MaxS, D] where MaxS in {kv_seq_len};
m3[S, D] = m2[input2[S], D];

m10[S, H, D] = input0[S, H, D].cast(`float32`);
m11[S, H, D] = input1[S, H, D].cast(`float32`);
di0a[S, H, D] = m10[S, H, D] * m1[S, D];
di1a[S, H, D] = m11[S, H, D] * m1[S, D];
dm4[S, H, D] = m10[S, H, D] * m3[S, D];
dm5[S, H, D] = m11[S, H, D] * m3[S, D];
di0b[S, H, D] = dm4[S, H, D + {head_dim // 2}].when([D < {head_dim // 2}], (-dm4[S, H, D - {head_dim // 2}])) where D in {head_dim};
di1b[S, H, D] = dm5[S, H, D + {head_dim // 2}].when([D < {head_dim // 2}], (-dm5[S, H, D - {head_dim // 2}])) where D in {head_dim};
output0[S, H, D] = di0a[S, H, D] + di0b[S, H, D];
output1[S, H, D] = di1a[S, H, D] + di1b[S, H, D];
''', input_orders={'input0': dq_embed, 'input1': dk_embed, 'input2': position_ids, 'input3': cos_cached, 'input4': sin_cached}, device=device, arch="A100")
        dq, dk = dqk_op([dq_embed, dk_embed, position_ids, cos_cached, sin_cached])

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
    

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_dtype(torch.float16)
    ref = LlamaRotaryEmbedding(64).to(device)
    fused = FusedLlamaRotaryEmbedding(64).to(device)
    
    q = torch.randn(2048, 8, 64, requires_grad=True, device=device)
    k = torch.randn(2048, 8, 64, requires_grad=True, device=device)
    v = torch.randn(2048, 8, 64, requires_grad=True, device=device)
    kv_seq_len = 2048
    position_ids = torch.arange(2048, dtype=torch.long, device=device)
    unsqueeze_dim = 1
    print ({'q': q.shape, 'k': k.shape, 'v': v.shape, 'kv_seq_len': kv_seq_len, 'position_ids': position_ids.shape, 'unsqueeze_dim': unsqueeze_dim})
    q_embed_ref, k_embed_ref = ref(q, k, v, kv_seq_len, position_ids, unsqueeze_dim)
    q_embed_fused, k_embed_fused = fused(q, k, v, kv_seq_len, position_ids, unsqueeze_dim)

    q_embed_grad = torch.ones_like(q_embed_fused, device=device)
    k_embed_grad = torch.ones_like(k_embed_fused, device=device)
    q_embed_fused.backward(q_embed_grad)
    k_embed_fused.backward(k_embed_grad)

    print (q_embed_ref[0][:10])
    print (q_embed_fused[0][:10])

    # start = time.time()
    # for i in range(100):
    #     y = layer.forward(x)
    #     y.backward(y_grad)
    #     #print(x, x.grad, layer.fc2.weight.grad, layer.fc2.bias.grad)
    # end = time.time()
    # print(end-start)
    
    



