import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from custom_op import CustomOp


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class FusedLlamaRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weights, eps):
        seq_len, hidden_size = hidden_states.shape

        var_op = CustomOp(ir=f'''
m0[S, H] = input0[S, H].cast(`float32`);
m1[S, H] = m0[S, H].call(`pow`, [const(2.0).cast(`float32`)]);
m2[S] +=! m1[S, H];
output0[S] = m2[S] / const({hidden_size}).cast(`float32`);
''', input_orders={'input0': hidden_states}, device=device, arch="A100")
        var = var_op([hidden_states])
        
        fused_op = CustomOp(ir=f'''
m0[S, H] = input0[S, H].cast(`float32`);
m5[S, H] = m0[S, H] / (input2[S] + const({eps}).cast(`float32`)).call(`sqrt`);
output0[S, H] = m5[S, H].cast(`float16`) * input1[H];
''', input_orders={'input0': hidden_states, 'input1': weights, 'input2': var}, device=device, arch="A100")
        y = fused_op([hidden_states, weights, var])

        ctx.save_for_backward(hidden_states, weights)
        ctx.eps = eps
        return y
    
    @staticmethod
    def backward(ctx, dy):
        hidden_states, weights = ctx.saved_tensors
        eps = ctx.eps
        hidden_size = hidden_states.shape[-1]

        var_op = CustomOp(ir=f'''
m0[S, H] = input0[S, H].cast(`float32`);
m1[S, H] = m0[S, H].call(`pow`, [const(2.0).cast(`float32`)]);
m2[S] +=! m1[S, H];
output0[S] = m2[S] / const({hidden_size}).cast(`float32`);
''', input_orders={'input0': hidden_states}, device=device, arch="A100")
        var = var_op([hidden_states])

        m5_op = CustomOp(ir=f'''
m0[S, H] = input0[S, H].cast(`float32`);
output0[S, H] = m0[S, H] / (input1[S] + const({eps}).cast(`float32`)).call(`sqrt`);
''', input_orders={'input0': hidden_states, 'input1': var}, device=device, arch="A100")
        m5 = m5_op([hidden_states, var])
        
        dw_op = CustomOp(ir=f'''
output0[H] +=! input0[S, H].cast(`float32`) * input1[S, H].cast(`float16`);
''', input_orders={'input0': dy, 'input1': m5}, device=device, arch="A100")
        dw = dw_op([dy, m5])

        dvar_op = CustomOp(ir=f'''
dm5[S, H] = input0[S, H].cast(`float32`) * input1[H].cast(`float32`);
m0[S] +=! dm5[S, H] * input2[S, H].cast(`float32`);
output0[S] = m0[S] * const(-0.5).cast(`float32`) * (input3[S].cast(`float32`) + const({eps}).cast(`float32`)).call(`pow`, [const(-1.5).cast(`float32`)]);
''', input_orders={'input0': dy, 'input1': weights, 'input2': hidden_states, 'input3': var}, device=device, arch="A100")
        dvar = dvar_op([dy, weights, hidden_states, var])

        dx_op = CustomOp(ir=f'''
dm5[S, H] = input0[S, H].cast(`float32`) * input3[H].cast(`float32`);
dx_1[S, H] = dm5[S, H] / (input4[S].cast(`float32`) + const({eps}).cast(`float32`)).call(`sqrt`);
dx_2[S, H] = input1[S].cast(`float32`) * const(2.0 / {hidden_size}).cast(`float32`) * input2[S, H].cast(`float32`);
output0[S, H] = dx_1[S, H] + dx_2[S, H];
''', input_orders={'input0': dy, 'input1': dvar, 'input2': hidden_states, 'input3': weights, 'input4': var}, device=device, arch="A100")
        dx = dx_op([dy, dvar, hidden_states, weights, var])

        return dx, dw, None

class FusedLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def reset_parameters(self):
        # self.fc2.reset_parameters()
        pass

    def forward(self, hidden_states):
        return FusedLlamaRMSNormFunc.apply(hidden_states, self.weight, self.variance_epsilon)


def test_forward_time(repeat, module, *args):
    warmup = 100
    for i in range(warmup):
        y = module(*args)

    elapsed_time = 0
    for i in range(repeat):
        start = time.time()
        y = module(*args)
        end = time.time()
        elapsed_time += (end-start)
    print (f"{module} forward time: {elapsed_time/repeat} sec.")

def test_backward_time(repeat, module, *args):
    warmup = 100
    for i in range(warmup):
        y = module(*args)
        loss = y.sum()
        loss.backward()
        
    elapsed_time = 0
    for i in range(repeat):
        y = module(*args)
        loss = y.sum()
        start = time.time()
        loss.backward()
        end = time.time()
        elapsed_time += (end-start)
    print (f"{module} backward time: {elapsed_time/repeat} sec.")

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_dtype(torch.float16)

    # Experiment setup
    max_seq_len = 4096
    hidden_size = 8192
    x = torch.randn(max_seq_len, hidden_size, requires_grad=True, device=device)
    x2 = x.detach().clone().requires_grad_()
    ref = LlamaRMSNorm(hidden_size).to(device)
    fused = FusedLlamaRMSNorm(hidden_size).to(device)
    
    # Run forward and backward
    y_ref = ref(x)
    loss_ref = y_ref.sum()
    loss_ref.backward()

    y_fused = fused(x2)
    loss_fused = y_fused.sum()
    loss_fused.backward()

    # Check validity
    print ("------ Vadility Check ------")
    print (f"y_ref      : {y_ref[0][:10]}")
    print (f"y_fused    : {y_fused[0][:10]}")
    print (f"x_ref_grad : {x.grad[0][:10]}")
    print (f"x_fused_grad: {x2.grad[0][:10]}")
    print (f"w_ref_grad : {ref.weight.grad[:10]}")
    print (f"w_fused_grad: {fused.weight.grad[:10]}")
    assert (torch.allclose(y_ref, y_fused, atol=1e-2, rtol=1e-3))
    assert (torch.allclose(x.grad, x2.grad, atol=1e-2, rtol=1e-3))
    assert (torch.allclose(ref.weight.grad, fused.weight.grad, atol=1e-2, rtol=1e-3))

    # Check efficiency
    print ("------ Efficiency Check ------")
    repeat = 1000
    test_forward_time(repeat, ref, x)
    test_forward_time(repeat, fused, x2)
    test_backward_time(repeat, ref, x)
    test_backward_time(repeat, fused, x2)

