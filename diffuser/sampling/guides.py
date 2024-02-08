import torch
import torch.nn as nn
import pdb


class ValueGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t):
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        #print("x[0,0,8]", x[0,0,:])
        #print("grad[0,0,8]", grad[0,0,:])
        x.detach()
        #import ipdb;ipdb.set_trace()
        return y, grad
