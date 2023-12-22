import os
import time
import torch
import torch.nn as nn
from fused_op import FusedLlamaRMSNorm, FusedLlamaRMSNormFunc


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
    seq_lens = [1024]
    hidden_sizes = [4096, 8192]
    use_module = True

    for max_seq_len in seq_lens:
        for hidden_size in hidden_sizes:
            x = torch.randn(max_seq_len, hidden_size, requires_grad=True, device=device)
            x2 = x.detach().clone().requires_grad_()
            ref = LlamaRMSNorm(hidden_size).to(device)

            # Run forward and backward
            y_ref = ref(x)
            loss_ref = y_ref.sum()
            loss_ref.backward()

            if use_module:
                fused = FusedLlamaRMSNorm(hidden_size).to(device)

                y_fused = fused(x2)
                loss_fused = y_fused.sum()
                loss_fused.backward()

                # Check validity
                print (f"------ Vadility Check ({max_seq_len}, {hidden_size}) ------")
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
                    error_code = check_all(y_ref, y_fused, x.grad, x2.grad, ref.weight.grad, fused.weight.grad, atol=1e-3, rtol=1e-3)                

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
            else:
                weight = nn.Parameter(torch.ones(hidden_size))
                variance_epsilon = 1e-6
                fused = FusedLlamaRMSNormFunc

                y_fused = fused.apply(x2, weight, variance_epsilon)
                loss_fused = y_fused.sum()
                loss_fused.backward()

                # Check validity
                print (f"------ Vadility Check ({max_seq_len}, {hidden_size}) ------")
                print (f"y_ref      : {y_ref[0][:10]}")
                print (f"y_fused    : {y_fused[0][:10]}")
                print (f"x_ref_grad : {x.grad[0][:10]}")
                print (f"x_fused_grad: {x2.grad[0][:10]}")
                print (f"w_ref_grad : {ref.weight.grad[:10]}")
                print (f"w_fused_grad: {weight.grad[:10]}")
                assert (torch.allclose(y_ref, y_fused, atol=1e-2, rtol=1e-3))
                assert (torch.allclose(x.grad, x2.grad, atol=1e-2, rtol=1e-3))
                assert (torch.allclose(ref.weight.grad, weight.grad, atol=1e-2, rtol=1e-3))

                # Check efficiency
                print ("------ Efficiency Check ------")
                repeat = 1000
                test_forward_time(repeat, ref, x)
                test_forward_time(repeat, fused, x2, weight, variance_epsilon)
                test_backward_time(repeat, ref, x)
                test_backward_time(repeat, fused, x2, weight, variance_epsilon)
                print ()

                del x, x2, ref, y_ref, y_fused, loss_ref, loss_fused
                torch.cuda.empty_cache()


