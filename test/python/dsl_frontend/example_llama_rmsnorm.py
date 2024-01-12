import torch
import torch.nn as nn
import os
from custom_op import CustomOp, KERNEL_CACHE
from test_utils import test_forward_time, test_backward_time


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
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
        hidden_size = hidden_states.shape[-1]

        var_op = CustomOp(ir=f'''
m0[B, S, H] = input0[B, S, H].cast(`float32`);
m1[B, S, H] = m0[B, S, H] * m0[B, S, H];
m2[B, S] +=! m1[B, S, H];
output0[B, S] = m2[B, S] / const({hidden_size}).cast(`float32`) + const({eps}).cast(`float32`);
''', input_orders={'input0': hidden_states}, device=device, arch=welder_arch)
        var = var_op([hidden_states])
        
        fused_op = CustomOp(ir=f'''
m0[B, S, H] = input0[B, S, H].cast(`float32`);
m5[B, S, H] = m0[B, S, H] / input2[B, S].call(`sqrt`);
output0[B, S, H] = m5[B, S, H].cast(`float16`) * input1[H];
''', input_orders={'input0': hidden_states, 'input1': weights, 'input2': var}, device=device, arch=welder_arch)
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
''', input_orders={'input0': dy, 'input1': hidden_states, 'input2': var}, device=device, arch=welder_arch)
        dw = dw_op([dy, hidden_states, var])

        dsqrtvar_op = CustomOp(ir=f'''
m8[B, S] +=! (input0[B, S, H] * input2[H]).cast(`float32`) * input1[B, S, H].cast(`float32`);
''', input_orders={'input0': dy, 'input1': hidden_states, 'input2': weights}, device=device, arch=welder_arch)
        dsqrtvar = dsqrtvar_op([dy, hidden_states, weights])

        dx_op = CustomOp(ir=f'''
sqrtvar[B, S] = input3[B, S].call(`sqrt`);
dvar[B, S] = input4[B, S] * const(-0.5).cast(`float32`) / (sqrtvar[B, S] * input3[B, S]);
dx_1[B, S, H] = (input0[B, S, H] * input2[H]).cast(`float32`) / sqrtvar[B, S];
dx_2[B, S, H] = dvar[B, S] * const(2.0 / {hidden_size}).cast(`float32`) * input1[B, S, H].cast(`float32`);
dx[B, S, H] = dx_1[B, S, H] + dx_2[B, S, H];
output0[B, S, H] = dx[B, S, H].cast(`float16`);
''', input_orders={'input0': dy, 'input1': hidden_states, 'input2': weights, 'input3': var, 'input4': dsqrtvar}, device=device, arch=welder_arch)
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


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_dtype(torch.float16)

    batch_size = 1
    arches = ['A100', 'V100', 'A6000']
    seq_lens = [64, 128, 256, 512, 1024]
    hidden_sizes = [4096, 8192]

    for arch in arches:
        os.environ["WELDER_ARCH"] = arch
        print (f"============= {arch} ================")
        
        for max_seq_len in seq_lens:
            for hidden_size in hidden_sizes:
                # Experiment setup
                x = torch.randn(batch_size, max_seq_len, hidden_size, requires_grad=True, device=device)
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
                # print (f"y_ref      : {y_ref[0][:10]}, {y_ref.dtype}")
                # print (f"y_fused    : {y_fused[0][:10]}, {y_fused.dtype}")
                # print (f"x_ref_grad : {x.grad[0][:10]}, {x.grad.dtype}")
                # print (f"x_fused_grad: {x2.grad[0][:10]}, {x2.grad.dtype}")
                # print (f"w_ref_grad : {ref.weight.grad[:10]}, {ref.weight.grad.dtype}")
                # print (f"w_fused_grad: {fused.weight.grad[:10]}, {fused.weight.grad.dtype}")
                
                # xshape = x.grad.shape
                # for i in range(xshape[0]):
                #     if not torch.allclose(x.grad[i], x2.grad[i], atol=1e-2, rtol=1e-2):
                #         for j in range(xshape[1]):
                #             if abs(x.grad[i][j] - x2.grad[i][j]) > 1e-2:
                #                 print (f"Error at ({i}, {j}): {x.grad[i][j]} != {x2.grad[i][j]}")
                #                 break
                #         break
                # for i in range(ref.weight.shape[0]):
                #     if not torch.allclose(ref.weight.grad[i], fused.weight.grad[i], atol=1e-2, rtol=1e-2):
                #         print (f"Error at ({i}): {ref.weight.grad[i]} != {fused.weight.grad[i]}")
                #         break

                def check_all(y_ref, y_fused, x_ref_grad, x_fused_grad, w_ref_grad, w_fused_grad, atol, rtol):
                    error_code = 0
                    if (not torch.allclose(y_ref, y_fused, atol=atol, rtol=rtol)):
                        error_code += 1
                        print (f"atol={atol}, rtol={rtol}: Forward check failed.")
                    if (not torch.allclose(w_ref_grad, w_fused_grad, atol=atol, rtol=rtol)):
                        error_code += 2
                        print (f"atol={atol}, rtol={rtol}: Backward dw check failed.")
                    if (not torch.allclose(x_ref_grad, x_fused_grad, atol=atol, rtol=rtol)):
                        error_code += 4
                        print (f"atol={atol}, rtol={rtol}: Backward dx check failed.")
                    if (error_code == 0):
                        print (f"atol={atol}, rtol={rtol}: Passed.")
                    return error_code
                
                error_code = check_all(y_ref, y_fused, x.grad, x2.grad, ref.weight.grad, fused.weight.grad, atol=1e-2, rtol=1e-2)
                if error_code == 0:
                    error_code = check_all(y_ref, y_fused, x.grad, x2.grad, ref.weight.grad, fused.weight.grad, atol=1e-2, rtol=1e-3)                

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
        
        from pathlib import Path
        home_path = os.environ["HOME"]
        Path(f"{home_path}/.kernel/{arch}/llama_rmsnorm/").mkdir(parents=True, exist_ok=True)
        open(f"{home_path}/.kernel/{arch}/__init__.py", 'a').close()
        open(f"{home_path}/.kernel/{arch}/llama_rmsnorm/__init__.py", 'a').close()
        exit_code = os.system(f"mv {home_path}/.kernel/*.json {home_path}/.kernel/{arch}/llama_rmsnorm/")
        print (f"(mv JSON) exit_code: {exit_code}")
        KERNEL_CACHE.clear()
        os.system("rm .antares*.out")

