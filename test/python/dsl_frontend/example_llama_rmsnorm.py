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

        hidden_size = hidden_states.shape[-1]
        fused_op = CustomOp(ir=f'''
m0[B, S, H] = input0[B, S, H].cast(`float32`);
m1[B, S, H] = (m0[B, S, H]).call(`pow`, [const(2.0)]);
m2[B, S] +=! m1[B, S, H];
m3[B, S] = m2[B, S] / const({hidden_size});
m4[B, S] = const(1.0) / (m3[B, S] + const({eps}).call(`sqrt`));
m5[B, S, H] = m1[B, S, H] * m4[B, S];
output0[B, S, H] = input1[H] * m5[B, S, H].cast(`float16`);
''', input_orders={'input0': hidden_states, 'input1': weights}, tags="tensorCoreConfig=(0, 1)", device=device, arch="A100")
        y = fused_op([hidden_states, weights])
        print ("y\t: ", y.shape)

        ctx.save_for_backward(hidden_states, weights)
        ctx.eps = eps
        return y
    
    @staticmethod
    def backward(ctx, dy):
        hidden_states, weight = ctx.saved_tensors
        eps = ctx.eps
        hidden_size = hidden_states.shape[-1]
        dw_op = CustomOp(ir=f'''
m0[B, S, H] = input0[B, S, H].cast(`float32`);
m1[B, S, H] = (m0[B, S, H]).call(`pow`, [const(2.0)]);
m2[B, S] +=! m1[B, S, H];
m3[B, S] = m2[B, S] / const({hidden_size});
m4[B, S] = const(1.0) / ((m3[B, S] + const({eps})).call(`sqrt`));
m5[B, S, H] = m1[B, S, H] * m4[B, S];
output0[H] +=! m5[B, S, H] * input2[B, S, H];
''', input_orders={'input0': hidden_states, 'input2': dy}, tags="tensorCoreConfig=(0, 1)", device=device, arch="A100")
        dw = dw_op([hidden_states, dy])

        dx_op = CustomOp(ir=f'''
m0[B, S, H] = input0[B, S, H].cast(`float32`);
m1[B, S, H] = (m0[B, S, H]).call(`pow`, [const(2.0)]);
m2[B, S] +=! m1[B, S, H];
m3[B, S] = m2[B, S] / const({hidden_size});
dy[B, S, H] = input2[B, S, H].cast(`float32`);
dm5[B, S, H] = dy[B, S, H] * input1[H];
dm4[B, S] = dm5[B, S, H] * m1[B, S, H];
dm3[B, S] = dm4[B, S] * const(-0.5) * (m3[B, S] + const({eps}).call(`pow`, [const(-1.5)]));
dm2[B, S] = dm3[B, S] / const({hidden_size});
dm1[B, S, H] = dm2[B, S];
output0[B, S, H] = dm1[B, S, H] * const(2.0) * m0[B, S, H];
''', input_orders={'input0': hidden_states, 'input1': weight, 'input2': dy}, tags="tensorCoreConfig=(0, 1)", device=device, arch="A100")
        dx = dx_op([hidden_states, weight, dy])

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
    x = torch.randn(16, 2048, 4096, requires_grad=True, device=device)
    ref = LlamaRMSNorm(4096).to(device)
    fused = FusedLlamaRMSNorm(4096).to(device)
    
    y_ref = ref(x)
    y_fused = fused(x)

    y_grad = torch.ones_like(y_fused, device=device)
    y_fused.backward(y_grad)

    print (y_ref[0][:10])
    print (y_fused[0][:10])

    # start = time.time()
    # for i in range(100):
    #     y = layer.forward(x)
    #     y.backward(y_grad)
    #     #print(x, x.grad, layer.fc2.weight.grad, layer.fc2.bias.grad)
    # end = time.time()
    # print(end-start)
    
    



