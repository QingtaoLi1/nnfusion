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
        print ("hidden_states\t: ", hidden_states.shape)    # [Batch, Seq, Hidden]
        print ("weights\t: ", weights.shape)                # [Hidden]

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_size = hidden_states.shape[-1]
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
        print ("y\t: ", y.shape)

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
output0[S] +=! m1[S, H] / const({hidden_size}).cast(`float32`);
''', input_orders={'input0': hidden_states}, device=device, arch="A100")
        var = var_op([hidden_states])

        m5_op = CustomOp(ir=f'''
m0[S, H] = input0[S, H].cast(`float32`);
m1[S] = input1[S].cast(`float32`) + const({eps}).cast(`float32`);
output0[S, H] = m0[S, H] / m1[S].call(`sqrt`);
''', input_orders={'input0': hidden_states, 'input1': var}, device=device, arch="A100")
        m5 = m5_op([hidden_states, var])
        
        dw_op = CustomOp(ir=f'''
output0[H] +=! input0[S, H].cast(`float32`) * input1[S, H].cast(`float32`);
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
        print (dx[0][:10])

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
    torch.set_default_dtype(torch.float16)
    x = torch.randn(2048, 512, requires_grad=True, device=device)
    x2 = x.detach().clone()
    ref = LlamaRMSNorm(512).to(device)
    fused = FusedLlamaRMSNorm(512).to(device)
    
    y_ref = ref(x)
    y_ref_grad = torch.ones_like(y_ref, device=device)
    y_ref.backward(y_ref_grad)

    y_fused = fused(x2)
    y_fused_grad = torch.ones_like(y_fused, device=device)
    y_fused.backward(y_fused_grad)

    print (y_ref[0][:10])
    print (y_fused[0][:10])
    print (x.grad[0][:10])
    print (x2.grad[0][:10])

    # start = time.time()
    # for i in range(100):
    #     y = layer.forward(x)
    #     y.backward(y_grad)
    #     #print(x, x.grad, layer.fc2.weight.grad, layer.fc2.bias.grad)
    # end = time.time()
    # print(end-start)
    
    



