import torch
import torch.nn as nn
import pdb
import time


class ValueGuide(nn.Module):

    def __init__(
        self, model, clip_grad=True, clip_grad_rule="norm", max_grad_norm=1.0, max_grad_value=1.0
    ):
        super().__init__()
        self.model = model
        self.clip_grad = clip_grad
        self.clip_grad_rule = clip_grad_rule
        self.max_grad_norm = max_grad_norm
        self.max_grad_norm = max_grad_value

    def forward(self, x, cond, t):
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        # start = time.time()
        # print("x.device: ", x.device)
        grad = torch.autograd.grad([y.sum()], [x], retain_graph=True)[0]
        # end = time.time()
        # print("time to compute gradient: ", end-start)
        # print("x[0,0,8]", x[0,0,:])
        # print("grad[0,0,8]", grad[0,0,:])
        # clip gradients
        grad_cost_clipped = self.clip_gradient(grad)
        # Clipp grad
        grad_cost_clipped[..., 0, :] = 0
        grad_cost_clipped[..., -1, :] = 0
        x.detach()
        # import ipdb;ipdb.set_trace()

        # Gradient ascent
        grad = -1.0 * grad_cost_clipped
        return y, grad

    def clip_gradient(self, grad):
        if self.clip_grad:
            if self.clip_grad_rule == "norm":
                return self.clip_grad_by_norm(grad)
            elif self.clip_grad_rule == "value":
                return self.clip_grad_by_value(grad)
            else:
                raise NotImplementedError
        else:
            return grad

    def clip_grad_by_norm(self, grad):
        # clip gradient by norm
        if self.clip_grad:
            grad_norm = torch.linalg.norm(grad + 1e-6, dim=-1, keepdims=True)
            scale_ratio = torch.clip(grad_norm, 0.0, self.max_grad_norm) / grad_norm
            grad = scale_ratio * grad
        return grad

    def clip_grad_by_value(self, grad):
        # clip gradient by value
        if self.clip_grad:
            grad = torch.clip(grad, -self.max_grad_value, self.max_grad_value)
        return grad
