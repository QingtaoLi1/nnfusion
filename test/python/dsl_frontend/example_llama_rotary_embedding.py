import torch
import torch.nn as nn
import os
from custom_op import CustomOp, KERNEL_CACHE
from test_utils import test_forward_time, test_backward_time


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
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)   # [1, MaxSeqLen, Dim]
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)   # [1, MaxSeqLen, Dim]
        print (f"t: {t.shape}")
        print (f"freqs: {freqs.shape}")
        print (f"emb: {emb.shape}")
        print (f"cos_cached: {self.cos_cached.shape}")
        print (f"sin_cached: {self.sin_cached.shape}")

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
        # print (f"position_ids: {position_ids.shape}")
        # print (q.shape)
        # print (cos.shape)
        # print (self._rotate_half(q).shape)
        # print (sin.shape)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed


class FusedLlamaRotaryEmbeddingFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, kv_seq_len, position_ids, unsqueeze_dim, cos_cached, sin_cached):
        welder_arch = os.environ["WELDER_ARCH"] if "WELDER_ARCH" in os.environ else "A100"
        head_dim = q.shape[-1]
        position_ids = position_ids.squeeze()

        fused_q_op = CustomOp(ir=f'''
m0[S, D] = input2[S, D] where S in {kv_seq_len};
m1[S, D] = m0[input1[S], D];
m2[S, D] = input3[S, D] where S in {kv_seq_len};
m3[S, D] = m2[input1[S], D];

m4[B, H, S, D] = (-input0[B, H, S, D + {head_dim // 2}]).when([D < {head_dim // 2}], input0[B, H, S, D - {head_dim // 2}]) where D in {head_dim};
output0[B, H, S, D] = input0[B, H, S, D] * m1[S, D] + m4[B, H, S, D] * m3[S, D];
''', input_orders={'input0': q, 'input1': position_ids, 'input2': cos_cached, 'input3': sin_cached}, device=device, arch=welder_arch)
        q_embed = fused_q_op([q, position_ids, cos_cached, sin_cached])

        fused_k_op = CustomOp(ir=f'''
m0[MaxS, D] = input2[MaxS, D] where MaxS in {kv_seq_len};
m1[S, D] = m0[input1[S], D];
m2[MaxS, D] = input3[MaxS, D] where MaxS in {kv_seq_len};
m3[S, D] = m2[input1[S], D];

m5[B, H, S, D] = (-input0[B, H, S, D + {head_dim // 2}]).when([D < {head_dim // 2}], input0[B, H, S, D - {head_dim // 2}]) where D in {head_dim};
output0[B, H, S, D] = input0[B, H, S, D] * m1[S, D] + m5[B, H, S, D] * m3[S, D];
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
m1[S, D] = m0[input1[S], D].cast(`float32`);
m2[MaxS, D] = input3[MaxS, D] where MaxS in {kv_seq_len};
m3[S, D] = m2[input1[S], D].cast(`float32`);

m10[B, H, S, D] = input0[B, H, S, D].cast(`float32`);
di0a[B, H, S, D] = m10[B, H, S, D] * m1[S, D];
dm4[B, H, S, D] = m10[B, H, S, D] * m3[S, D];
di0b[B, H, S, D] = dm4[B, H, S, D + {head_dim // 2}].when([D < {head_dim // 2}], (-dm4[B, H, S, D - {head_dim // 2}])) where D in {head_dim};
di0[B, H, S, D] = di0a[B, H, S, D] + di0b[B, H, S, D];
output0[B, H, S, D] = di0[B, H, S, D].cast(`float16`);
''', input_orders={'input0': dq_embed, 'input1': position_ids, 'input2': cos_cached, 'input3': sin_cached}, device=device, arch=welder_arch)
        dq = dq_op([dq_embed, position_ids, cos_cached, sin_cached])

        dk_op = CustomOp(ir=f'''
m0[MaxS, D] = input2[MaxS, D] where MaxS in {kv_seq_len};
m1[S, D] = m0[input1[S], D].cast(`float32`);
m2[MaxS, D] = input3[MaxS, D] where MaxS in {kv_seq_len};
m3[S, D] = m2[input1[S], D].cast(`float32`);

m11[B, H, S, D] = input0[B, H, S, D].cast(`float32`);
di1a[B, H, S, D] = m11[B, H, S, D] * m1[S, D];
dm5[B, H, S, D] = m11[B, H, S, D] * m3[S, D];
di1b[B, H, S, D] = dm5[B, H, S, D + {head_dim // 2}].when([D < {head_dim // 2}], (-dm5[B, H, S, D - {head_dim // 2}])) where D in {head_dim};
di1[B, H, S, D] = di1a[B, H, S, D] + di1b[B, H, S, D];
output0[B, H, S, D] = di1[B, H, S, D].cast(`float16`);
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
    

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_dtype(torch.float16)

    arches = ["A100", 'V100', 'A6000']
    batch_size = 1
    max_seq_lens = 1024
    seq_lens = [64, 128, 256, 512, 1024]
    q_kv_hidden_sizes = [(4096, 4096), (8192, 1024)]
    head_dim = 128
    for arch in arches:
        os.environ["WELDER_ARCH"] = arch
        print (f"============= {arch} ================")
        
        for seq_len in seq_lens:
            for (q_hidden_size, kv_hidden_size) in q_kv_hidden_sizes:
                q_num_head = q_hidden_size // head_dim
                kv_num_head = kv_hidden_size // head_dim
                unsqueeze_dim = 0

                ref = LlamaRotaryEmbedding(head_dim, max_position_embeddings=max_seq_lens).to(device)
                fused = FusedLlamaRotaryEmbedding(head_dim, max_position_embeddings=max_seq_lens).to(device)
                q = torch.randn(batch_size, q_num_head, seq_len, head_dim, requires_grad=True, device=device)
                k = torch.randn(batch_size, kv_num_head, seq_len, head_dim, requires_grad=True, device=device)
                v = torch.randn(batch_size, kv_num_head, seq_len, head_dim, requires_grad=True, device=device)
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device).view(1, seq_len)

                # print ("WARNING!!! only support head_dim % 2 == 0 !!!")
                # print ("q\t: ", q.shape)                        # [Seq, Head, HeadDim]
                # print ("k\t: ", k.shape)                        # [Seq, Head, HeadDim]
                # print ("v\t: ", v.shape)                        # [Seq, Head, HeadDim]
                # print ("kv_seq_len\t: ", seq_len)               # scalar
                # print ("position_ids\t: ", position_ids.shape)  # [Seq]
                # print ("unsqueeze_dim\t: ", unsqueeze_dim)      # scalar
                # print ("cos_cached\t: ", ref.cos_cached.shape)      # [MaxSeqLen, Dim]
                # print ("sin_cached\t: ", ref.sin_cached.shape)      # [MaxSeqLen, Dim]


                q2 = q.clone().detach().requires_grad_(True)
                k2 = k.clone().detach().requires_grad_(True)
                v2 = v.clone().detach().requires_grad_(True)
                position_ids2 = position_ids.clone().detach()

                # Run forward and backward
                q_embed_ref, k_embed_ref = ref(q, k, v, seq_len, position_ids, unsqueeze_dim)
                loss_ref = q_embed_ref.sum() + k_embed_ref.sum()
                loss_ref.backward()

                q_embed_fused, k_embed_fused = fused(q2, k2, v2, seq_len, position_ids2, unsqueeze_dim)
                loss_fused = q_embed_fused.sum() + k_embed_fused.sum()
                loss_fused.backward()

                # Check validity
                print (f"------ Vadility Check : ({seq_len}, {q_hidden_size}, {kv_hidden_size}) ------")
                # print (f"q_embed_ref    : {q_embed_ref.flatten()[:10]}")
                # print (f"q_embed_fused  : {q_embed_fused.flatten()[:10]}")
                # print (f"k_embed_ref    : {k_embed_ref.flatten()[:10]}")
                # print (f"k_embed_fused  : {k_embed_fused.flatten()[:10]}")
                # print (f"q_ref_grad     : {q.grad.flatten()[-10:]}")    # grad of first half all 1.0.
                # print (f"q_fused_grad   : {q2.grad.flatten()[-10:]}")
                # print (f"k_ref_grad     : {k.grad.flatten()[-10:]}")
                # print (f"k_fused_grad   : {k2.grad.flatten()[-10:]}")

                # assert (torch.allclose(q_embed_ref, q_embed_fused, atol=1e-2, rtol=1e-3))
                # assert (torch.allclose(k_embed_ref, k_embed_fused, atol=1e-2, rtol=1e-3))
                # assert (torch.allclose(q.grad, q2.grad, atol=1e-2, rtol=1e-3))
                # assert (torch.allclose(k.grad, k2.grad, atol=1e-2, rtol=1e-3))

                def check_all(q_embed_ref, q_embed_fused, k_embed_ref, k_embed_fused,
                            q_ref_grad, q_fused_grad, k_ref_grad, k_fused_grad,
                            atol, rtol):
                    error_code = 0
                    if (not torch.allclose(q_embed_ref, q_embed_fused, atol=atol, rtol=rtol)):
                        error_code += 1
                        print (f"atol={atol}, rtol={rtol}: Forward q failed.")
                    if (not torch.allclose(k_embed_ref, k_embed_fused, atol=atol, rtol=rtol)):
                        error_code += 2
                        print (f"atol={atol}, rtol={rtol}: Backward dq check failed.")
                    if (not torch.allclose(q_ref_grad, q_fused_grad, atol=atol, rtol=rtol)):
                        error_code += 4
                        print (f"atol={atol}, rtol={rtol}: Forward k check failed.")
                    if (not torch.allclose(k_ref_grad, k_fused_grad, atol=atol, rtol=rtol)):
                        error_code += 8
                        print (f"atol={atol}, rtol={rtol}: Backward dk check failed.")
                    if (error_code == 0):
                        print (f"atol={atol}, rtol={rtol}: Passed.")
                    return error_code

                error_code = check_all(q_embed_ref, q_embed_fused, k_embed_ref, k_embed_fused, q.grad, q2.grad, k.grad, k2.grad, atol=1e-2, rtol=1e-2)
                if error_code == 0:
                    error_code = check_all(q_embed_ref, q_embed_fused, k_embed_ref, k_embed_fused, q.grad, q2.grad, k.grad, k2.grad, atol=1e-2, rtol=1e-3)                

                # Check efficiency
                print ("------ Efficiency Check ------")
                repeat = 1000
                test_forward_time(repeat, ref, q, k, v, seq_len, position_ids, unsqueeze_dim)
                test_forward_time(repeat, fused, q2, k2, v2, seq_len, position_ids2, unsqueeze_dim)
                test_backward_time(repeat, ref, q, k, v, seq_len, position_ids, unsqueeze_dim)
                test_backward_time(repeat, fused, q2, k2, v2, seq_len, position_ids2, unsqueeze_dim)
                print ()

                del ref, fused, q, k, v, q2, k2, v2, q_embed_ref, k_embed_ref, q_embed_fused, k_embed_fused, loss_ref, loss_fused
                torch.cuda.empty_cache()
        
        from pathlib import Path
        home_path = os.environ["HOME"]
        Path(f"{home_path}/.kernel/{arch}/llama_rope/").mkdir(parents=True, exist_ok=True)
        open(f"{home_path}/.kernel/{arch}/__init__.py", 'a').close()
        open(f"{home_path}/.kernel/{arch}/llama_rope/__init__.py", 'a').close()
        exit_code = os.system(f"mv {home_path}/.kernel/*.json {home_path}/.kernel/{arch}/llama_rope/")
        print (f"(mv JSON) exit_code: {exit_code}")
        KERNEL_CACHE.clear()
        os.system("rm .antares*.out")
