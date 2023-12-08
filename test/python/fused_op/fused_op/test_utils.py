import torch
import time


def test_forward_time(repeat, module, *args):
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

