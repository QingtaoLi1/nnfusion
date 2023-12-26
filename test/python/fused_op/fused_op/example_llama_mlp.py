import torch
from torch import nn
import os
from .custom_op import CustomOp


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
op_name = "llama_mlp"

# class LlamaMLP(nn.Module):
#     def __init__(self, hidden_size, intermediate_size, gp, up, dp):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.intermediate_size = intermediate_size
#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.gate_proj.weight = nn.Parameter(gp.weight.clone())
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.up_proj.weight = nn.Parameter(up.weight.clone())
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
#         self.down_proj.weight = nn.Parameter(dp.weight.clone())
#         self.act_fn = nn.SiLU()
 
#     def forward(self, x):
#         # if self.config.pretraining_tp > 1:
#         #    pass
#         # else:
#         down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
 
#         return down_proj
 
class FusedLlamaMLPFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, down_weight):
        welder_arch = os.environ["WELDER_ARCH"] if "WELDER_ARCH" in os.environ else "A100"

        fused_op_0 = CustomOp(ir=f'''
m0[S, INTER] +=! input0[S, D] * input1[INTER, D];
''', input_orders={'input0': x, 'input1': gate_weight}, tags="tensorCoreConfig=(0, 1)", op_name=op_name, device=device, arch=welder_arch)
        y0 = fused_op_0([x, gate_weight])

        fused_op_1 = CustomOp(ir=f'''
m1[S, INTER] +=! input0[S, D] * input1[INTER, D];
''', input_orders={'input0': x, 'input1': up_weight}, tags="tensorCoreConfig=(0, 1)", op_name=op_name, device=device, arch=welder_arch)
        y1 = fused_op_1([x, up_weight])

        fused_op = CustomOp(ir=f'''
m11[S, INTER] = input1[S, INTER].cast(`float32`);
m2[S, INTER] = input1[S, INTER] / (const(1.0).cast(`float16`) + (-m11[S, INTER]).call(`exp`).cast(`float16`));
m3[S, INTER] = m2[S, INTER] * input2[S, INTER];
output0[S, D] +=! m3[S, INTER] * input0[D, INTER];
''', input_orders={'input0': down_weight, 'input1': y0, 'input2': y1}, tags="tensorCoreConfig=(0, 1)", op_name=op_name, device=device, arch=welder_arch)
        y = fused_op([down_weight, y0, y1])
        ctx.save_for_backward(x, gate_weight, up_weight, down_weight, y0, y1)
        return y
 
    @staticmethod
    def backward(ctx, dy):
        welder_arch = os.environ["WELDER_ARCH"] if "WELDER_ARCH" in os.environ else "A100"
        x, gate_weight, up_weight, down_weight, _gx, _ux = ctx.saved_tensors

        dsilugu_op = CustomOp(ir=f'''
m1[S, INTER] +=! input0[S, D] * input1[D, INTER];
''', input_orders={'input0': dy, 'input1': down_weight}, tags="tensorCoreConfig=(0, 1)", op_name=op_name, device=device, arch=welder_arch)
        dsilugu = dsilugu_op([dy, down_weight])

        grad_silu_op =  CustomOp(ir=f'''
m1[S, INTER] = input0[S, INTER].cast(`float32`);
m4[S, INTER] = const(1.0).cast(`float16`) / (const(1.0).cast(`float16`) + (-m1[S, INTER]).call(`exp`).cast(`float16`));
m5[S, INTER] = m4[S, INTER] + input0[S, INTER] * m4[S, INTER] * (const(1.0).cast(`float16`) - m4[S, INTER]);
m2[S, INTER] = input1[S, INTER] * input2[S, INTER] * m5[S, INTER];
''', input_orders={'input0': _gx, 'input1': _ux, 'input2': dsilugu}, op_name=op_name, device=device, arch=welder_arch)
        grad_silu_gx = grad_silu_op([_gx, _ux, dsilugu])

        dx_left_op = CustomOp(ir=f'''
m7[S, D] +=! input0[S, INTER] * input1[INTER, D];
''', input_orders={'input0': grad_silu_gx, 'input1': gate_weight}, tags="tensorCoreConfig=(0, 1)", op_name=op_name, device=device, arch=welder_arch)
        dx_left = dx_left_op([grad_silu_gx, gate_weight])
 
        silu_gate_op = CustomOp(ir=f'''
m81[S, INTER] = input0[S, INTER].cast(`float32`);
m9[S, INTER] = input0[S, INTER] / (const(1.0).cast(`float16`) + (-m81[S, INTER]).call(`exp`).cast(`float16`));
''', input_orders={'input0': _gx}, op_name=op_name, device=device, arch=welder_arch)
        silu = silu_gate_op([_gx])
 
        dx_op = CustomOp(ir=f'''
m10[S, INTER] = input0[S, INTER] * input1[S, INTER];
m11[S, D] +=! m10[S, INTER] * input2[INTER, D];
output0[S, D] = input3[S, D] + m11[S, D];
''', input_orders={'input0': dsilugu, 'input1': silu, 'input2': up_weight, 'input3': dx_left}, tags="tensorCoreConfig=(0, 1)", op_name=op_name, device=device, arch=welder_arch)
        dx = dx_op([dsilugu, silu, up_weight, dx_left])
 
        dgw_op = CustomOp(ir=f'''
m7[INTER, D] +=! input0[S, INTER] * input1[S, D];
''', input_orders={'input0': grad_silu_gx, 'input1': x}, tags="tensorCoreConfig=(0, 1)", op_name=op_name, device=device, arch=welder_arch)
        dgw = dgw_op([grad_silu_gx, x])
 
        duw_op = CustomOp(ir=f'''
m2[S, INTER] = input0[S, INTER] * input1[S, INTER];
m0[INTER, D] +=! m2[S, INTER] * input2[S, D];
''', input_orders={'input0': dsilugu, 'input1': silu, 'input2': x}, tags="tensorCoreConfig=(0, 1)", op_name=op_name, device=device, arch=welder_arch)
        duw = duw_op([dsilugu, silu, x])
 
        ddw_op_2 = CustomOp(ir=f'''
m0[S, INTER] = input0[S, INTER] * input1[S, INTER];
m7[D, INTER] +=! input2[S, D] * m0[S, INTER];
''', input_orders={'input0': silu, 'input1': _ux, 'input2': dy}, tags="tensorCoreConfig=(0, 1)", op_name=op_name, device=device, arch=welder_arch)
        ddw = ddw_op_2([silu, _ux, dy])
 
        return dx, dgw, duw, ddw
 
 
class FusedLlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
 
    def reset_parameters(self):
        self.fc2.reset_parameters()
 
    def forward(self, x):
        return FusedLlamaMLPFunc.apply(x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight)
   
    def get_fc2(self):
        return self.fc2

