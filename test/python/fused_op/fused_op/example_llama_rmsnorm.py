import torch
import torch.nn as nn
import os
from .custom_op import CustomOp


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
op_name = "llama_rmsnorm"

# class LlamaRMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-5):
#         """
#         LlamaRMSNorm is equivalent to T5LayerNorm
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)

class FusedLlamaRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weights, eps):
        welder_arch = os.environ["WELDER_ARCH"] if "WELDER_ARCH" in os.environ else "A100"
        hidden_size = hidden_states.shape[-1]

        var_op = CustomOp(ir=f'''
m0[B, S, H] = input0[B, S, H].cast(`float32`);
m1[B, S, H] = m0[B, S, H] * m0[B, S, H];
m2[B, S] +=! m1[B, S, H];
output0[B, S] = m2[B, S] / const({hidden_size}).cast(`float32`) + const({eps}).cast(`float32`);
''', input_orders={'input0': hidden_states}, device=device, arch=welder_arch, op_name=op_name)
        var = var_op([hidden_states])
        
        fused_op = CustomOp(ir=f'''
m0[B, S, H] = input0[B, S, H].cast(`float32`);
m5[B, S, H] = m0[B, S, H] / input2[B, S].call(`sqrt`);
output0[B, S, H] = m5[B, S, H].cast(`float16`) * input1[H];
''', input_orders={'input0': hidden_states, 'input1': weights, 'input2': var}, device=device, arch=welder_arch, op_name=op_name)
        y = fused_op([hidden_states, weights, var])

        ctx.save_for_backward(hidden_states, weights, var)
        return y
    
    @staticmethod
    def backward(ctx, dy):
        welder_arch = os.environ["WELDER_ARCH"] if "WELDER_ARCH" in os.environ else "A100"
        hidden_states, weights, var = ctx.saved_tensors
        hidden_size = hidden_states.shape[-1]

        dw_op = CustomOp(ir=f'''
m6[B, S, H] = input1[B, S, H].cast(`float32`) / input2[B, S].call(`sqrt`);
dw[H] +=! input0[B, S, H].cast(`float32`) * m6[B, S, H].cast(`float16`);
output0[H] = dw[H].cast(`float16`);
''', input_orders={'input0': dy, 'input1': hidden_states, 'input2': var}, device=device, arch=welder_arch, op_name=op_name)
        dw = dw_op([dy, hidden_states, var])

        dsqrtvar_op = CustomOp(ir=f'''
m8[B, S] +=! (input0[B, S, H] * input2[H]).cast(`float32`) * input1[B, S, H].cast(`float32`);
''', input_orders={'input0': dy, 'input1': hidden_states, 'input2': weights}, device=device, arch=welder_arch, op_name=op_name)
        dsqrtvar = dsqrtvar_op([dy, hidden_states, weights])

        dx_op = CustomOp(ir=f'''
sqrtvar[B, S] = input3[B, S].call(`sqrt`);
dvar[B, S] = input4[B, S] * const(-0.5).cast(`float32`) / (sqrtvar[B, S] * input3[B, S]);
dx_1[B, S, H] = (input0[B, S, H] * input2[H]).cast(`float32`) / sqrtvar[B, S];
dx_2[B, S, H] = dvar[B, S] * const(2.0 / {hidden_size}).cast(`float32`) * input1[B, S, H].cast(`float32`);
dx[B, S, H] = dx_1[B, S, H] + dx_2[B, S, H];
output0[B, S, H] = dx[B, S, H].cast(`float16`);
''', input_orders={'input0': dy, 'input1': hidden_states, 'input2': weights, 'input3': var, 'input4': dsqrtvar}, device=device, arch=welder_arch, op_name=op_name)
        dx = dx_op([dy, hidden_states, weights, var, dsqrtvar])

        return dx, dw, None

class FusedLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def reset_parameters(self):
        # self.fc2.reset_parameters()
        pass

    def forward(self, hidden_states):
        return FusedLlamaRMSNormFunc.apply(hidden_states, self.weight, self.variance_epsilon)

