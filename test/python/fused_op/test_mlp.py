import os
import time
import torch
import torch.nn as nn
from fused_op import FusedLlamaMLP


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
    batch_size = 1
    seq_lens = [64, 128, 256, 512, 1024]
    hidden_sizes = [4096, 8192]
    intermediate_sizes = [11008, 28672]
    for seq_len in seq_lens:
        for hidden_size, intermediate_size in zip(hidden_sizes, intermediate_sizes):
            x = torch.randn(batch_size, seq_len, hidden_size, requires_grad = True, device=device)
            x2 = x.detach().clone().requires_grad_()
            fused = FusedLlamaMLP(hidden_size, intermediate_size).to(device)
            ref = LlamaMLP(hidden_size, intermediate_size, getattr(fused, 'gate_proj'), getattr(fused, 'up_proj'), getattr(fused, 'down_proj'))
        
            y_ref = ref(x)
            loss_ref = y_ref.sum()
            loss_ref.backward()
        
            y_fused = fused(x2)
            loss_fused = y_fused.sum()
            loss_fused.backward()
        
            # Check validity
            print (f"------ Vadility Check ({seq_len}, {hidden_size}, {intermediate_size}) ------")

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
                                   atol=1e-2, rtol=1e-2)
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
