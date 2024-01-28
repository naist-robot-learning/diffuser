import torch
import torch.nn as nn
import pdb
import numpy as np
import einops
# State-Space (name/joint/parameter):
#         - rootx     slider      position (m)
#         - rootz     slider      position (m)
#         - rooty     hinge       angle (rad)
#         - bthigh    hinge       angle (rad)
#         - bshin     hinge       angle (rad)
#         - bfoot     hinge       angle (rad)
#         - fthigh    hinge       angle (rad)
#         - fshin     hinge       angle (rad)
#         - ffoot     hinge       angle (rad)
#         - rootx     slider      velocity (m/s)
#         - rootz     slider      velocity (m/s)
#         - rooty     hinge       angular velocity (rad/s)
#         - bthigh    hinge       angular velocity (rad/s)
#         - bshin     hinge       angular velocity (rad/s)
#         - bfoot     hinge       angular velocity (rad/s)
#         - fthigh    hinge       angular velocity (rad/s)
#         - fshin     hinge       angular velocity (rad/s)
#         - ffoot     hinge       angular velocity (rad/s)

class CostFn(nn.Module):

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        cond_dim: int,
        dim_mults: tuple=(1, 2, 4, 8),
        out_dim: int=1,
    ):
        super().__init__()
        # transition_dim = 23
        
        self.q_cost = torch.zeros((transition_dim, horizon), requires_grad=False)
        self.q_cost[1,:] = 0.3
        self.q_cost[2,:] = 0.7854
                    

    def forward(self, x: torch.tensor((64,4,23)), 
                      cond: torch.tensor, 
                      time: torch.tensor)->torch.tensor(64):
        '''
            x : [ batch x horizon x transition ]
        '''
        x = einops.rearrange(x, 'b h t -> b t h')
        #print("shape x: ", np.shape(x))  #shape x:  torch.Size([64, 23, 4])
        #print("x: ", x)
        q = x.clone().detach()
        q[:,1,:] = self.q_cost[1,1]
        q[:,2,:] = self.q_cost[2,1]
        self.q_cost = q
        self.q_cost.requires_grad_(True).to("cuda")
        power = (self.q_cost - x).pow(2)
        squared_norm = power.sum(axis=1)
        cost = squared_norm.sum(axis=1)     
        #print("squared of norm: ", cost)
        return cost
