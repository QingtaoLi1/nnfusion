import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from custom_op import CustomOp

class CustomMatmul(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        fc2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.fc2.weight = nn.Parameter(fc2.weight.clone())
        self.fc2.bias = nn.Parameter(fc2.bias.clone())

    def reset_parameters(self):
        # self.fc2.reset_parameters()
        pass

    def forward(self, x):
        x = self.fc2(x)
        return x
    
class FusedMatmulFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        print ("x\t: ", x.shape)
        print ("weight\t: ", weight.shape)
        print ("bias\t: ", bias.shape)

        fused_op = CustomOp(ir=f'''
m0[N0, N1] +=! input0[N0, N2] * input1[N1, N2];
output0[N0, N1] = m0[N0, N1] + input2[N1];
''', input_orders={'input0': x, 'input1': weight, 'input2': bias}, tags="tensorCoreConfig=(0, 1)", device=device, arch="A100")
        y = fused_op([x, weight, bias])
        ctx.save_for_backward(x, weight)
        return y
    
    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        print ("x\t: ", x.shape)
        print ("weight\t: ", weight.shape)
        print ("dy\t: ", dy.shape)

        dbias = torch.sum(dy, dim=0)
        print ("dbias\t: ", dbias.shape)

        dw_op = CustomOp(ir=f'''
output0[N1, N2] +=! input1[N0, N1] * input0[N0, N2];
''', input_orders={'input0': x, 'input1': dy}, tags="tensorCoreConfig=(0, 1)", device=device, arch="A100")
        dw = dw_op([x, dy])
        print ("dw\t: ", dw.shape)

        dx_op = CustomOp(ir=f'''
output0[N0, N2] +=! input1[N0, N1] * input0[N1, N2];
''', input_orders={'input0': weight, 'input1': dy}, tags="tensorCoreConfig=(0, 1)", device=device, arch="A100")
        dx = dx_op([weight, dy])
        print ("dx\t: ", dx.shape)

        return dx, dw, dbias

class FusedMatmul(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        fc2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim, dtype=torch.float16)
        self.fc2.weight = nn.Parameter(fc2.weight.clone().half())
        self.fc2.bias = nn.Parameter(fc2.bias.clone().half())

    def reset_parameters(self):
        # self.fc2.reset_parameters()
        pass

    def forward(self, x):
        return FusedMatmulFunc.apply(x, self.fc2.weight, self.fc2.bias)
    

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_dtype(torch.float16)
    x = torch.randn(2048, 16384, requires_grad = True, device=device)
    fc2 = nn.Linear(16384, 4096)
    ref = CustomMatmul(4096, 16384, fc2).to(device)
    fused = FusedMatmul(4096, 16384, fc2).to(device)
    
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
    
    



