import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os

from custom_op import CustomOp
from test_utils import test_forward_time, test_backward_time


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, gp, up, dp):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.gate_proj.weight = nn.Parameter(gp.weight.clone())
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj.weight = nn.Parameter(up.weight.clone())
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.down_proj.weight = nn.Parameter(dp.weight.clone())
        self.act_fn = nn.SiLU()
 
    def forward(self, x):
        # if self.config.pretraining_tp > 1:
        #    pass
        # else:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
 
        return down_proj
 
class FusedMLPFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, down_weight):
        welder_arch = os.environ["WELDER_ARCH"] if "WELDER_ARCH" in os.environ else "A100"
#         fused_op = CustomOp(ir=f'''
# m0[N0, N2] +=! input0[N0, N1] * input1[N2, N1];
# m1[N0, N2] +=! input0[N0, N1] * input2[N2, N1];
# m2[N0, N2] = m0[N0, N2] / (1.0 + (-m0[N0, N2]).call(`exp`));
# m3[N0, N2] = m2[N0, N2] * m1[N0, N2];
# output0[N0, N3] +=! m3[N0, N2] * input3[N3, N2];
# ''', input_orders={'input0':x, 'input1':gate_weight, 'input2':up_weight, 'input3':down_weight}, tags="tensorCoreConfig=(0, 1)", device=device)
# # ''', input_orders={'input0':x, 'input1':gate_weight, 'input2':up_weight, 'input3':down_weight}, device=device, arch=welder_arch)
#         y = fused_op([x, gate_weight, up_weight, down_weight])
        fused_op_0 = CustomOp(ir=f'''
m0[S, INTER] +=! input0[S, D] * input1[INTER, D];
''', input_orders={'input0': x, 'input1': gate_weight}, tags="tensorCoreConfig=(0, 1)", device=device, arch=welder_arch)
        y0 = fused_op_0([x, gate_weight])

        fused_op_1 = CustomOp(ir=f'''
m1[S, INTER] +=! input0[S, D] * input1[INTER, D];
''', input_orders={'input0': x, 'input1': up_weight}, tags="tensorCoreConfig=(0, 1)", device=device, arch=welder_arch)
        y1 = fused_op_1([x, up_weight])

        fused_op = CustomOp(ir=f'''
m11[S, INTER] = input1[S, INTER].cast(`float32`);
m2[S, INTER] = input1[S, INTER] / (const(1.0).cast(`float16`) + (-m11[S, INTER]).call(`exp`).cast(`float16`));
m3[S, INTER] = m2[S, INTER] * input2[S, INTER];
output0[S, D] +=! m3[S, INTER] * input0[D, INTER];
''', input_orders={'input0': down_weight, 'input1': y0, 'input2': y1}, tags="tensorCoreConfig=(0, 1)", device=device, arch=welder_arch)
# ''', input_orders={'input0':x, 'input1':gate_weight, 'input2':up_weight, 'input3':down_weight}, device=device, arch=welder_arch)
        y = fused_op([down_weight, y0, y1])
        ctx.save_for_backward(x, gate_weight, up_weight, down_weight)
        # ctx.p = p
        return y
        '''
m11[S, INTER] = input1[S, INTER].cast(`float32`);
m2[S, INTER] = m11[S, INTER] / (const(1.0).cast(`float32`) + (-m11[S, INTER]).call(`exp`));
m3[S, INTER] = m2[S, INTER].cast(`float16`) * input2[S, INTER];
output0[S, D] +=! m3[S, INTER] * input0[D, INTER];
        '''
# class FusedMLPFunc(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, gate_weight, up_weight, down_weight):
#         print("down_weight:", down_weight)
# #         fused_op = CustomOp(ir=f'''
# # m0[N0, N2] +=! input0[N0, N1] * input1[N2, N1];
# # m1[N0, N2] +=! input0[N0, N1] * input2[N2, N1];
# # m2[N0, N2] = m0[N0, N2] / (1.0 + (-m0[N0, N2]).call(`exp`));
# # m3[N0, N2] = m2[N0, N2] * m1[N0, N2];
# # output0[N0, N3] +=! m3[N0, N2] * input3[N3, N2];
# # ''', input_orders={'input0':x, 'input1':gate_weight, 'input2':up_weight, 'input3':down_weight}, tags="tensorCoreConfig=(0, 1)", device=device)
# # # ''', input_orders={'input0':x, 'input1':gate_weight, 'input2':up_weight, 'input3':down_weight}, device=device)
# #         y = fused_op([x, gate_weight, up_weight, down_weight])
#         fused_op_0 = CustomOp(ir=f'''
# m1[N0, N1] = input0[N0, N1].cast(`float32`);
# m2[N2, N1] = input1[N2, N1].cast(`float32`);
# m0[N0, N2] +=! m1[N0, N1] * m2[N2, N1];
# output0[N0, N2] = m0[N0, N2].cast(`float16`)
# ''', input_orders={'input0':x, 'input1':gate_weight}, tags="tensorCoreConfig=(0, 1)", device=device)
#         y0 = fused_op_0([x, gate_weight])
#         print("y0:", y0)
#         fused_op_1 = CustomOp(ir=f'''
# m1[N0, N1] = input0[N0, N1].cast(`float32`);
# m2[N2, N1] = input1[N2, N1].cast(`float32`);
# m0[N0, N2] +=! m1[N0, N1] * m2[N2, N1];
# output0[N0, N2] = m0[N0, N2].cast(`float16`)
# ''', input_orders={'input0':x, 'input1':up_weight}, tags="tensorCoreConfig=(0, 1)", device=device)
#         y1 = fused_op_1([x, up_weight])
#         print("y1:", y1)
#         fused_op = CustomOp(ir=f'''
# m2[N0, N2] = input1[N0, N2] / (const(1.0).cast(`float16`) + (-input1[N0, N2]).call(`exp`));
# m3[N0, N2] = m2[N0, N2] * input2[N0, N2];
# output0[N0, N3] +=! m3[N0, N2] * input0[N3, N2];
# ''', input_orders={'input0':down_weight, 'input1':y0, 'input2':y1},device=device)
# # ''', input_orders={'input0':x, 'input1':gate_weight, 'input2':up_weight, 'input3':down_weight}, device=device)
# # i0[N3, N2] = input0[N3, N2].cast(`float32`);
# # m22[N0, N2] = m2[N0, N2].cast(`float32`);
# # i2[N0, N2] = input2[N0, N2].cast(`float32`);
#         y = fused_op([down_weight,y0,y1])
#         print("y:", y)
#         ctx.save_for_backward(x, gate_weight, up_weight, down_weight)
#         # ctx.p = p
#         return y
 
    @staticmethod
    def backward(ctx, dy):
        welder_arch = os.environ["WELDER_ARCH"] if "WELDER_ARCH" in os.environ else "A100"
        x, gate_weight, up_weight, down_weight = ctx.saved_tensors
        # print("dy_shape:", dy.shape)                # [S, D]
        # print("x_shape:", x.shape)                  # [S, D]
        # print("down_shape:", down_weight.shape)     # [D, INTER]
        # print("up_shape:", up_weight.shape)         # [INTER, D]
        # print("gate_shape:", gate_weight.shape)     # [INTER, D]

        dx_op_ux = CustomOp(ir=f'''
m0[S, INTER] +=! input0[S, D] * input1[INTER, D];
''', input_orders={'input0': x, 'input1': up_weight}, tags="tensorCoreConfig=(0, 1)", device=device, arch=welder_arch)
        _ux = dx_op_ux([x, up_weight])
        
        dx_op_1 = CustomOp(ir=f'''
m1[S, INTER] +=! input0[S, D] * input1[D, INTER];
''', input_orders={'input0': dy, 'input1': down_weight}, tags="tensorCoreConfig=(0, 1)", device=device, arch=welder_arch)
        dy_dwT = dx_op_1([dy,down_weight])

        dx_op_gx = CustomOp(ir=f'''
m3[S, INTER] +=! input0[S, D] * input1[INTER, D];
''', input_orders={'input0': x, 'input1': gate_weight}, tags="tensorCoreConfig=(0, 1)", device=device, arch=welder_arch)
        _gx = dx_op_gx([x, gate_weight])

        grad_silu_op =  CustomOp(ir=f'''
m4[S, INTER] = const(1.0).cast(`float16`) / (const(1.0).cast(`float16`) + (-input0[S, INTER]).call(`exp`));
m5[S, INTER] = m4[S, INTER] + input0[S, INTER] * m4[S, INTER] * (const(1.0).cast(`float16`) - m4[S, INTER]);
''', input_orders={'input0': _gx}, device=device, arch=welder_arch)
        grad_silu_gx = grad_silu_op([_gx])

        dx_left_op = CustomOp(ir=f'''
m2[S, INTER] = input0[S, INTER] * input1[S, INTER] * input2[S, INTER];
m7[S, D] +=! m2[S, INTER] * input3[INTER, D];
''', input_orders={'input0': _ux, 'input1': dy_dwT, 'input2': grad_silu_gx, 'input3': gate_weight}, tags="tensorCoreConfig=(0, 1)", device=device, arch=welder_arch)
        dx_left = dx_left_op([_ux, dy_dwT, grad_silu_gx, gate_weight])
 
        silu_gate_op = CustomOp(ir=f'''
m8[S, INTER] +=! input0[S, D] * input1[INTER, D];
m9[S, INTER] = m8[S, INTER] / (const(1.0).cast(`float16`) + (-m8[S, INTER]).call(`exp`));
''', input_orders={'input0': x, 'input1': gate_weight}, tags="tensorCoreConfig=(0, 1)", device=device, arch=welder_arch)
        silu_gate = silu_gate_op([x, gate_weight])
 
        dx_op = CustomOp(ir=f'''
m10[S, INTER] = input0[S, INTER] * input1[S, INTER];
m11[S, D] +=! m10[S, INTER] * input2[INTER, D];
output0[S, D] = input3[S, D] + m11[S, D];
''', input_orders={'input0': dy_dwT, 'input1': silu_gate, 'input2': up_weight, 'input3': dx_left}, tags="tensorCoreConfig=(0, 1)", device=device, arch=welder_arch)
        dx = dx_op([dy_dwT, silu_gate, up_weight, dx_left])
 
        dgw_op = CustomOp(ir=f'''
m6[S, INTER] = input0[S, INTER] * input1[S, INTER] * input2[S, INTER];
m7[INTER, D] +=! m6[S, INTER] * input3[S, D];
''', input_orders={'input0': _ux, 'input1': dy_dwT, 'input2': grad_silu_gx, 'input3': x}, tags="tensorCoreConfig=(0, 1)", device=device, arch=welder_arch)
        dgw = dgw_op([_ux, dy_dwT, grad_silu_gx, x])
 
        duw_op = CustomOp(ir=f'''
m2[S, INTER] = input0[S, INTER] * input1[S, INTER];
m0[INTER, D] +=! m2[S, INTER] * input2[S, D];
''', input_orders={'input0': dy_dwT, 'input1': silu_gate, 'input2': x}, tags="tensorCoreConfig=(0, 1)", device=device, arch=welder_arch)
        duw = duw_op([dy_dwT, silu_gate, x])
 
        ddw_op_2 = CustomOp(ir=f'''
m0[S, INTER] = input0[S, INTER] * input1[S, INTER];
m7[D, INTER] +=! input2[S, D] * m0[S, INTER];
''', input_orders={'input0': silu_gate, 'input1': _ux, 'input2': dy}, tags="tensorCoreConfig=(0, 1)", device=device, arch=welder_arch)
        ddw = ddw_op_2([silu_gate, _ux, dy])
 
        return dx, dgw, duw, ddw
 
 
#         x, weight, mask = ctx.saved_tensors
#         p = ctx.p
#         dbias = torch.sum(dy, dim=0)
#         dw_op = CustomOp(ir=f'''
# m0[N0, N1] = input0[N0, N1].cast(`float32`);
# m1[N0, N1] = m0[N0, N1] * const(0.5).cast(`float32`) * (const(1.0).cast(`float32`) + (m0[N0, N1] * const({M_SQRT1_2}).cast(`float32`)).call(`erf`));
# m2[N0, N1] = m1[N0, N1].cast(`float16`);
# m3[N0, N1] = m2[N0, N1] * input2[N0, N1] / const({1-p}).cast(`float16`);
# output0[N2, N1] +=! input1[N0, N2] * m3[N0, N1];
# ''', input_orders={'input0': x, 'input1': dy, 'input2': mask}, tags="tensorCoreConfig=(0, 1)", device=device)
#         dw = dw_op([x, dy, mask])
 
#         dx_op = CustomOp(ir=f'''
# m0[N0, N1] +=! input3[N0, N2] * input1[N2, N1];
# m1[N0, N1] = m0[N0, N1] * input2[N0, N1] * const({1-p}).cast(`float16`);
# m2[N0, N1] = m1[N0, N1].cast(`float32`);
# m3[N0, N1] = const(0.5).cast(`float32`) * (const(1.0).cast(`float32`) + (input0[N0, N1] * const({M_SQRT1_2}).cast(`float32`)).call(`erf`));
# m4[N0, N1] = (const(-0.5).cast(`float32`) * input0[N0, N1] * input0[N0, N1]).call(`exp`) * const({M_2_SQRTPI * M_SQRT1_2 * 0.5}).cast(`float32`);
# output0[N0, N1] = m2[N0, N1] * (m3[N0, N1] + input0[N0, N1] * m4[N0, N1]);
# ''', input_orders={'input0': x, 'input1': weight, 'input2': mask, 'input3': dy}, tags="tensorCoreConfig=(0, 1)", device=device)
#         dx = dx_op([x, weight, mask, dy])
        return None, None, None, None
 
class FusedCustomMLP(nn.Module):
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
        return FusedMLPFunc.apply(x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight)
        # return FusedLinearFunc.apply(x,  torch.eye(self.fc2.weight.shape[0], self.fc2.weight.shape[1]), torch.zeros_like(self.fc2.bias), self.activation_dropout)
   
    def get_fc2(self):
        return self.fc2
   
if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_dtype(torch.float16)
    os.environ["WELDER_ARCH"] = "A100"

    seq_lens = [64, 128, 256, 512, 1024]
    hidden_sizes = [4096, 8192]
    intermediate_sizes = [11008, 28672]
    for seq_len in seq_lens:
        for hidden_size, intermediate_size in zip(hidden_sizes, intermediate_sizes):
            x = torch.randn(seq_len, hidden_size, requires_grad = True, device=device)
            x2 = x.detach().clone().requires_grad_()
            fused = FusedCustomMLP(hidden_size, intermediate_size).to(device)
            ref = LlamaMLP(hidden_size, intermediate_size, getattr(fused, 'gate_proj'), getattr(fused, 'up_proj'), getattr(fused, 'down_proj'))
        
            y_ref = ref(x)
            loss_ref = y_ref.sum()
            loss_ref.backward()
        
            y_fused = fused(x2)
            loss_fused = y_fused.sum()
            loss_fused.backward()
        
            # Check validity
            print (f"------ Vadility Check ({seq_len}, {hidden_size}, {intermediate_size}) ------")
            # print (f"y_ref      : {y_ref[0][:10]}")
            # print (f"y_fused    : {y_fused[0][:10]}")
            # print (f"x_ref_grad : {x.grad[0][:10]}")
            # print (f"x_fused_grad: {x2.grad[0][:10]}")
            # print (f"gate_ref_grad : {ref.gate_proj.weight.grad[:10]}")
            # print (f"w_ref_shape : {ref.gate_proj.weight.shape}")
            # print (f"w_ref_grad_shape : {ref.gate_proj.weight.grad.shape}")
            # print (f"gate_fused_grad: {fused.gate_proj.weight.grad[:10]}")
            # print (f"w_fused_shape: {fused.down_proj.weight.shape}")
            # y_grad = torch.ones_like(y_fused, device=device)
            # y_fused.backward(y_grad)

            def check_all(y_ref, y_fused, x_ref_grad, x_fused_grad,
                          gate_ref_grad, gate_fused_grad, up_ref_grad, up_fused_grad, down_ref_grad, down_fused_grad,
                          atol, rtol):
                error_code = 0
                if (not torch.allclose(y_ref, y_fused, atol=atol, rtol=rtol)):
                    error_code += 1
                    print (f"atol={atol}, rtol={rtol}: Forward y failed.")
                if (not torch.allclose(x_ref_grad, x_fused_grad, atol=atol, rtol=rtol)):
                    error_code += 2
                    print (f"atol={atol}, rtol={rtol}: Backward dx check failed.")
                if (not torch.allclose(gate_ref_grad, gate_fused_grad, atol=atol, rtol=rtol)):
                    error_code += 4
                    print (f"atol={atol}, rtol={rtol}: Backward dgate check failed.")
                if (not torch.allclose(up_ref_grad, up_fused_grad, atol=atol, rtol=rtol)):
                    error_code += 8
                    print (f"atol={atol}, rtol={rtol}: Backward dup check failed.")
                if (not torch.allclose(down_ref_grad, down_fused_grad, atol=atol, rtol=rtol)):
                    error_code += 16
                    print (f"atol={atol}, rtol={rtol}: Backward ddown check failed.")
                if (error_code == 0):
                    print (f"atol={atol}, rtol={rtol}: Passed.")
                return error_code

            error_code = check_all(y_ref, y_fused, x.grad, x2.grad,
                                   ref.gate_proj.weight.grad, fused.gate_proj.weight.grad,
                                   ref.up_proj.weight.grad, fused.up_proj.weight.grad,
                                   ref.down_proj.weight.grad, fused.down_proj.weight.grad,
                                   atol=1e-2, rtol=1e-3)
            if error_code == 0:
                error_code = check_all(y_ref, y_fused, x.grad, x2.grad,
                                       ref.gate_proj.weight.grad, fused.gate_proj.weight.grad,
                                       ref.up_proj.weight.grad, fused.up_proj.weight.grad,
                                       ref.down_proj.weight.grad, fused.down_proj.weight.grad,
                                       atol=1e-3, rtol=1e-3)

            # Check efficiency
            print ("------ Efficiency Check ------")
            repeat = 1000
            test_forward_time(repeat, ref, x)
            test_forward_time(repeat, fused, x2)
            test_backward_time(repeat, ref, x)
            test_backward_time(repeat, fused, x2)
            print ()

            del x, x2, ref, fused, y_ref, y_fused, loss_ref, loss_fused
            torch.cuda.empty_cache()
