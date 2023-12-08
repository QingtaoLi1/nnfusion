import torch
import torch.nn as nn
import os
from custom_op import CustomOp
from test_utils import test_forward_time, test_backward_time


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
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
m0[S, H] = input0[S, H].cast(`float32`);
m5[S, H] = m0[S, H] / (input1[S] + const({eps}).cast(`float32`)).call(`sqrt`);
output0[H] +=! input2[S, H].cast(`float32`) * m5[S, H].cast(`float16`);
''', input_orders={'input0': hidden_states, 'input1': var, 'input2': dy}, device=device, arch=welder_arch)
        dw = dw_op([hidden_states, var, dy])

        dx_op = CustomOp(ir=f'''
dm5[S, H] = input0[S, H].cast(`float32`) * input1[H].cast(`float32`);
m0[S] +=! dm5[S, H] * input2[S, H].cast(`float32`);
m1[S] = input3[S].cast(`float32`) + const({eps}).cast(`float32`);
dvar[S] = m0[S] * const(-0.5).cast(`float32`) * m1[S].call(`pow`, [const(-1.5).cast(`float32`)]);
dx_1[S, H] = dm5[S, H] / m1[S].call(`sqrt`);
dx_2[S, H] = dvar[S].cast(`float32`) * const(2.0 / {hidden_size}).cast(`float32`) * input2[S, H].cast(`float32`);
output0[S, H] = dx_1[S, H] + dx_2[S, H];
''', input_orders={'input0': dy, 'input1': weights, 'input2': hidden_states, 'input3': var}, device=device, arch=welder_arch)
        dx = dx_op([dy, weights, hidden_states, var])

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


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.environ["WELDER_ARCH"] = "A100"
    torch.set_default_dtype(torch.float16)

    seq_lens = [1024]
    hidden_sizes = [16384]

    for max_seq_len in seq_lens:
        for hidden_size in hidden_sizes:
            # Experiment setup
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
            print (f"------ Vadility Check : ({max_seq_len}, {hidden_size}) ------")
            print (f"y_ref      : {y_ref[0][:10]}")
            print (f"y_fused    : {y_fused[0][:10]}")
            print (f"x_ref_grad : {x.grad[0][:10]}")
            print (f"x_fused_grad: {x2.grad[0][:10]}")
            print (f"w_ref_grad : {ref.weight.grad[:10]}")
            print (f"w_fused_grad: {fused.weight.grad[:10]}")
            
            xshape = x.grad.shape
            for i in range(xshape[0]):
                print (f"{i}, ", end="")
                if not torch.allclose(x.grad[i], x2.grad[i], atol=1e-2, rtol=1e-3):
                    for j in range(xshape[1]):
                        if abs(x.grad[i][j] - x2.grad[i][j]) > 1e-2:
                            print (f"Error at ({i}, {j}): {x.grad[i][j]} != {x2.grad[i][j]}")
                            break

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
            print ()

            del x, x2, ref, fused, y_ref, y_fused, loss_ref, loss_fused
            torch.cuda.empty_cache()

            # (seq_len, hidden_size)-Start_Fail_Row
            # Succeeded: (4096, 4096), (4096, 8192), (1024, 4096), (1024, 8192), (512, 4096), (256, 4096)
            # Failed: (512, 8192)-256, (256, 8192)-128, (128, 4096)-64, (128, 8192)-64, (64, 4096)-16, (64, 8192)-16
