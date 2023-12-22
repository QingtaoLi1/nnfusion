import os
import time
import torch
import torch.nn as nn
from fused_op import FusedLlamaRotaryEmbedding


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


def test_forward_time(repeat, module, *args):
    if isinstance(module, torch.autograd.Function):
        module = module.apply

    warmup = 100
    for i in range(warmup):
        y = module(*args)

    elapsed_time = 0
    for i in range(repeat):
        start = time.time()
        y = module(*args)
        end = time.time()
        elapsed_time += (end-start)
    print (f"{module} forward time: {elapsed_time/repeat*1000} ms.")


def test_backward_time(repeat, module, *args):
    if isinstance(module, torch.autograd.Function):
        module = module.apply

    warmup = 100
    for i in range(warmup):
        y = module(*args)
        if isinstance(y, tuple):
            loss = sum([yi.sum() for yi in y if isinstance(yi, torch.Tensor)])
        else:
            loss = y.sum()
        loss.backward()
        
    elapsed_time = 0
    for i in range(repeat):
        # module.zero_grad()
        y = module(*args)
        if isinstance(y, tuple):
            loss = sum([yi.sum() for yi in y if isinstance(yi, torch.Tensor)])
        else:
            loss = y.sum()
        start = time.time()
        loss.backward()
        end = time.time()
        elapsed_time += (end-start)
    print (f"{module} backward time: {elapsed_time/repeat*1000} ms.")


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.environ["WELDER_ARCH"] = "A100"
    torch.set_default_dtype(torch.float16)

    # Experiment setup
    max_seq_lens = 1024
    seq_lens = [64, 128, 256, 512, 1024]
    q_kv_hidden_sizes = [(4096, 4096), (8192, 1024)]
    head_dim = 128

    for seq_len in seq_lens:
        for (q_hidden_size, kv_hidden_size) in q_kv_hidden_sizes:
            q_num_head = q_hidden_size // head_dim
            kv_num_head = kv_hidden_size // head_dim
            unsqueeze_dim = 1

            ref = LlamaRotaryEmbedding(head_dim, max_position_embeddings=max_seq_lens).to(device)
            fused = FusedLlamaRotaryEmbedding(head_dim, max_position_embeddings=max_seq_lens).to(device)
            q = torch.randn(seq_len, q_num_head, head_dim, requires_grad=True, device=device)
            k = torch.randn(seq_len, kv_num_head, head_dim, requires_grad=True, device=device)
            v = torch.randn(seq_len, kv_num_head, head_dim, requires_grad=True, device=device)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)

            # print ("WARNING!!! only support unsqueeze_dim = 1 !!!")
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
    

