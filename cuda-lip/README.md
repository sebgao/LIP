## Getting Started
CUDA LIP is faster and more memory efficient than the implementation made up of PyTorch primitives. Our CUDA LIP includes forward and backward process, meaning that you can train LIP models with CUDA LIP.

To install CUDA LIP, please run
```
make install
```

After this, you can test CUDA LIP by
```
make test
```
You should see `all passed!` and further the comparison about speed of implementations of LIP. A typical output is like
```
cd test-files && python test.py
check forward ...
check inplace_primitive_lip2d ...
check cuda_lip2d ...
check backward ...
check cuda_lip2d ...
check pooling size and stride ...
all passed!
profiling information ...
[primitive_lip2d foward]:
Self CPU time total: 29.479ms
CUDA time total: 98.881ms

[cuda_lip2d foward]:
Self CPU time total: 2.863ms
CUDA time total: 35.725ms

[torch.nn.functional.avg_pool2d foward]:
Self CPU time total: 956.536us
CUDA time total: 20.110ms

[primitive_lip2d forward&backward]:
Self CPU time total: 401.907ms
CUDA time total: 729.445ms

[cuda_lip2d forward&backward]:
Self CPU time total: 68.416ms
CUDA time total: 233.631ms
```

## Usage
You can import CUDA LIP in Python by
```
from LIP import cuda_lip2d
```
after the installation and use it like
```
cuda_lip2d(x, logit, kernel=3, stride=2, padding=1)
```

## Numeric Stability
We also include the forward code with the numeric stability in `lip_cuda_kernel.cu`. You can comment on and off codes to switch to the numeric-stable version. *Note: this may hurt speed due to more loops in the CUDA kernel.*