import torch
import torch.nn as nn
import os
from .custom_op import CustomOp


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
        welder_arch = os.environ["WELDER_ARCH"] if "WELDER_ARCH" in os.environ else "A100"
        seq_len, hidden_size = hidden_states.shape

        var_op = CustomOp(ir=f'''
m0[S, H] = input0[S, H].cast(`float32`);
m1[S, H] = m0[S, H].call(`pow`, [const(2.0).cast(`float32`)]);
m2[S] +=! m1[S, H];
output0[S] = m2[S] / const({hidden_size}).cast(`float32`);
''', input_orders={'input0': hidden_states}, device=device, arch=welder_arch)
        var = var_op([hidden_states])
        
        fused_op = CustomOp(ir=f'''
m0[S, H] = input0[S, H].cast(`float32`);
m5[S, H] = m0[S, H] / (input2[S] + const({eps}).cast(`float32`)).call(`sqrt`);
output0[S, H] = m5[S, H].cast(`float16`) * input1[H];
''', input_orders={'input0': hidden_states, 'input1': weights, 'input2': var}, device=device, arch=welder_arch)
        y = fused_op([hidden_states, weights, var])

        ctx.save_for_backward(hidden_states, weights, var)
        ctx.eps = eps
        return y
    
    @staticmethod
    def backward(ctx, dy):
        welder_arch = os.environ["WELDER_ARCH"] if "WELDER_ARCH" in os.environ else "A100"
        hidden_states, weights, var = ctx.saved_tensors
        eps = ctx.eps
        hidden_size = hidden_states.shape[-1]

        dw_op = CustomOp(ir=f'''
m0[S, H] = input1[S, H].cast(`float32`);
m5[S, H] = m0[S, H] / (input2[S] + const({eps}).cast(`float32`)).call(`sqrt`);
output0[H] +=! input0[S, H].cast(`float32`) * m5[S, H].cast(`float16`);
''', input_orders={'input0': dy, 'input1': hidden_states, 'input2': var}, device=device, arch=welder_arch)
        dw = dw_op([dy, hidden_states, var])

        dx_op = CustomOp(ir=f'''
dm5[S, H] = input0[S, H].cast(`float32`) * input2[H].cast(`float32`);
m0[S] +=! dm5[S, H] * input1[S, H].cast(`float32`);
m1[S] = input3[S].cast(`float32`) + const({eps}).cast(`float32`);
dvar[S] = m0[S] * const(-0.5).cast(`float32`) * m1[S].call(`pow`, [const(-1.5).cast(`float32`)]);
dx_1[S, H] = dm5[S, H] / m1[S].call(`sqrt`);
dx_2[S, H] = dvar[S].cast(`float32`) * const(2.0 / {hidden_size}).cast(`float32`) * input1[S, H].cast(`float32`);
output0[S, H] = dx_1[S, H] + dx_2[S, H];
''', input_orders={'input0': dy, 'input1': hidden_states, 'input2': weights, 'input3': var}, device=device, arch=welder_arch)
        dx = dx_op([dy, hidden_states, weights, var])

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

