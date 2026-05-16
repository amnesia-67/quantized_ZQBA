import torch
import torch.nn as nn

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits):
        ctx.num_bits = num_bits
        if num_bits == 1.58: # Ternary quantization
            return torch.sign(torch.round(input))
        
        # Standard uniform affine quantization
        qmin = 0.0
        qmax = 2.**num_bits - 1.
        scale = (input.max() - input.min()) / (qmax - qmin)
        scale = torch.max(scale, torch.tensor([1e-8], device=input.device))
        
        zero_point = qmin - torch.round(input.min() / scale)
        q_x = torch.round(input / scale + zero_point)
        q_x.clamp_(qmin, qmax)
        
        return (q_x - zero_point) * scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: pass gradients directly through the non-differentiable step
        return grad_output, None

class CustomQuantizer(nn.Module):
    def __init__(self, num_bits=8):
        super(CustomQuantizer, self).__init__()
        self.num_bits = num_bits

    def forward(self, x):
        return STEQuantize.apply(x, self.num_bits)