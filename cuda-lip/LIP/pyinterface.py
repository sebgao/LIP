from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import torch

import lip_cuda_interface

class CUDA_LIP2d(Function):
    @staticmethod
    def forward(ctx, features, logits, kernel=3, stride=2):
        B, C, H, W = features.size()
        oH = H//stride
        oW = W//stride
        output = features.new_zeros((B, C, oH, oW))
        
        lip_cuda_interface.forward(features, logits, kernel, stride, output)

        ctx.save_for_backward(output, features, logits)

        ctx.kernel = kernel
        ctx.stride = stride
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = torch.zeros_like(ctx.saved_tensors[1])
        grad_w = torch.zeros_like(ctx.saved_tensors[2])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride] + [grad_x, grad_w]

        lip_cuda_interface.backward(*saved)
        
        return saved[-2], saved[-1], None, None

def inplace_primitive_lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp_()
    return F.avg_pool2d(x*weight, kernel, stride, padding).div_(
        F.avg_pool2d(weight, kernel, stride, padding)
    )

def primitive_lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()
    return F.avg_pool2d(x*weight, kernel, stride, padding).div(
        F.avg_pool2d(weight, kernel, stride, padding)
    )

def cuda_lip2d(x, logit, kernel=3, stride=2, padding=1):
    '''
        This function runs the CUDA version of LIP when the input is a
        CUDA tensor, otherwise incurs an error.
        
        Args:
            x : (torch.Tensor) - the input feature map,
            logit : (torch.Tensor) - the input logit *without exp(...)*,
            kernel : (int) - the pooling window size,
            stride : (int) - the pooling window stride,
            padding : (int) - the paddng size.

        Note:
        1. the logit should be of the same shape with the input `x`.
        Broadcasting is currently not supported by CUDA LIP.
        2. the `kernel`, `stride` and `padding` are restricted to int type,
        *not tuples*. 
    '''
    
    if x.is_cuda:
        assert x.size() == logit.size(), 'feature and logits must be same shaped when incurring CUDA program'
        return CUDA_LIP2d.apply(x, logit, kernel, stride)
    else:
        raise Exception('cuda_lip2d meets non-cuda inputs')