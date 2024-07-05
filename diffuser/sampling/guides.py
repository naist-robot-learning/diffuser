import torch
import torch.nn as nn
import pdb
import time


class ValueGuide(nn.Module):

    def __init__(self, model, weights, clip_grad=True, clip_grad_rule="norm", max_grad_norm=1.0, max_grad_value=1.0):
        super().__init__()
        self.model = model
        self._weights = {"smoothness": weights[0]}
        self._weights["refmass"] = weights[1]
        self._weights["goal_pose"] = weights[2]
        self._weights["via_point1"] = weights[2]
        self._weights["via_point2"] = weights[2]
        self.clip_grad = clip_grad
        self.clip_grad_rule = clip_grad_rule
        self.max_grad_norm = max_grad_norm
        self.max_grad_norm = max_grad_value

    def forward(self, x, cond, t):
        output, output_measured = self.model(x, cond, t)
        for i, (key, val) in enumerate(output.items()):
            output[key] = val.squeeze(dim=-1)
        return output, output_measured.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        cost_dict, cost_measured = self(x, *args)
        # start = time.time()
        # print("x.device: ", x.device)
        accu_grad = 0
        for i, (key, cost) in enumerate(cost_dict.items()):
            grad = torch.autograd.grad([cost.sum()], [x], retain_graph=True)[0]
            # clip gradients
            grad_cost_clipped = self.clip_gradient(grad)
            grad_cost_clipped[..., 0, :] = 0
            # Clip grad
            if key is not "goal_pose":
                grad_cost_clipped[..., -1, :] = 0
            accu_grad += grad_cost_clipped * self._weights[key]
            x.detach()
            # import ipdb;ipdb.set_trace()

        # Gradient ascent
        # accu_grad = -1.0 * accu_grad
        return cost_dict, accu_grad, cost_measured

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
