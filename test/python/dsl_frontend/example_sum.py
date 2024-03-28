import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from custom_op import CustomOp

class CustomSum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(x, dim=-1)
    
class FusedMatmulFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        print ("x\t: ", x.shape)

        fused_op = CustomOp(ir=f'''
output0[N0] +=! input0[N0, N1];
''', input_orders={'input0': x}, device=device, arch="A100")
        y = fused_op([x])
        ctx.length = x.shape[-1]
        return y
    
    @staticmethod
    def backward(ctx, dy):
        dx_op = CustomOp(ir=f'''
output0[N0, N1] = input0[N0] where N1 in {ctx.length};
''', input_orders={'input0': dy}, device=device, arch="A100")
        dx = dx_op([dy])
        print ("dx\t: ", dx.shape)
        return dx

class FusedSum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return FusedMatmulFunc.apply(x)
    

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_dtype(torch.float16)
    x = torch.randn(2048, 4096, requires_grad = True, device=device)
    ref = CustomSum().to(device)
    fused = FusedSum().to(device)
    
    y_ref = ref(x)
    y_fused = fused(x)

    y_grad = torch.ones_like(y_fused, device=device)
    y_fused.backward(y_grad)

    print (y_ref[:10])
    print (y_fused[:10])

    # start = time.time()
    # for i in range(100):
    #     y = layer.forward(x)
    #     y.backward(y_grad)
    #     #print(x, x.grad, layer.fc2.weight.grad, layer.fc2.bias.grad)
    # end = time.time()
    # print(end-start)
    
    



