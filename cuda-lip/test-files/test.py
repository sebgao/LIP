import torch
import lip_cuda_interface
from LIP import cuda_lip2d, primitive_lip2d, inplace_primitive_lip2d

def check_close_enough(a, check):
    residual = (a-check).data.abs().mean().cpu().item()
    assert torch.isnan(check).sum() == 0, 'meet NaN(s) in `check`'
    assert residual < 1e-6, 'residual is not small: {}'.format(residual)

x = torch.rand((20, 32, 128, 128)).float()
lgt = torch.rand_like(x).float()*10

print('check forward ...')
print('check inplace_primitive_lip2d ...')

pl = primitive_lip2d(x, lgt)
ipl = inplace_primitive_lip2d(x, lgt.mul(1.0))

check_close_enough(pl.data, ipl.data)

print('check cuda_lip2d ...')
x = x.cuda()
lgt = lgt.cuda()

pl = primitive_lip2d(x, lgt)
cl = cuda_lip2d(x, lgt)

check_close_enough(cl.data, pl.data)


print('check backward ...')
print('check cuda_lip2d ...')
a = torch.rand((20, 32, 128, 128)).float().cuda()
lga = torch.rand_like(a).float().cuda()*10

b = torch.zeros_like(a).copy_(a)
lgb = torch.zeros_like(a).copy_(lga)

a.requires_grad = True
lga.requires_grad = True
b.requires_grad = True
lgb.requires_grad = True

primitive_lip2d(a, lga).pow(2).mean().backward()
cuda_lip2d(b, lgb).pow(2).mean().backward()

check_close_enough(a.grad.data, b.grad.data)
check_close_enough(lga.grad.data, lgb.grad.data)


print('check pooling size and stride ...')
a = torch.rand((20, 32, 128, 128)).float().cuda()
lga = torch.rand_like(a).float().cuda()*10

b = torch.zeros_like(a).copy_(a)
lgb = torch.zeros_like(a).copy_(lga)

a.requires_grad = True
lga.requires_grad = True
b.requires_grad = True
lgb.requires_grad = True

primitive_lip2d(a, lga, kernel=2, stride=2, padding=0).pow(2).mean().backward()
cuda_lip2d(b, lgb, kernel=2, stride=2, padding=0).pow(2).mean().backward()

check_close_enough(a.grad.data, b.grad.data)
check_close_enough(lga.grad.data, lgb.grad.data)

a = torch.rand((20, 32, 128, 128)).float().cuda()
lga = torch.rand_like(a).float().cuda()*10

b = torch.zeros_like(a).copy_(a)
lgb = torch.zeros_like(a).copy_(lga)

a.requires_grad = True
lga.requires_grad = True
b.requires_grad = True
lgb.requires_grad = True

primitive_lip2d(a, lga, kernel=5, stride=3, padding=2).pow(2).mean().backward()
cuda_lip2d(b, lgb, kernel=5, stride=3, padding=2).pow(2).mean().backward()

check_close_enough(a.grad.data, b.grad.data)
check_close_enough(lga.grad.data, lgb.grad.data)

print('all passed!')
print('profiling information ...')
x = x.cuda()
lgt = lgt.cuda()


def extract_total(prof):
    return '\n'.join(str(prof).split('\n')[-3:])

for i in range(100):
    _tt = primitive_lip2d(x, lgt)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        primitive_lip2d(x, lgt)

print('[primitive_lip2d foward]:')
print(extract_total(prof))

for i in range(100):
    cuda_lip2d(x, lgt)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        cuda_lip2d(x, lgt)

print('[cuda_lip2d foward]:')
print(extract_total(prof))

for i in range(100):
    torch.nn.functional.avg_pool2d(x, 3, 2, 1)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        torch.nn.functional.avg_pool2d(x, 3, 2, 1)

print('[torch.nn.functional.avg_pool2d foward]:')
print(extract_total(prof))


for i in range(100):
    primitive_lip2d(x, lgt)

x.requires_grad = True
lgt.requires_grad = True

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        primitive_lip2d(x, lgt).backward(_tt)

print('[primitive_lip2d forward&backward]:')
print(extract_total(prof))

for i in range(100):
    cuda_lip2d(x, lgt)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        cuda_lip2d(x, lgt).backward(_tt)

print('[cuda_lip2d forward&backward]:')
print(extract_total(prof))
