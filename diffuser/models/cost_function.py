import torch
import torch.nn as nn
import pdb
import numpy as np
import einops

# State-Space (name/joint/parameter):
# 0         - rootz     slider      position (m)
# 1         - rooty     hinge       angle (rad)
# 2         - bthigh    hinge       angle (rad)
# 3         - bshin     hinge       angle (rad)
# 4         - bfoot     hinge       angle (rad)
# 5         - fthigh    hinge       angle (rad)
# 6         - fshin     hinge       angle (rad)
# 7         - ffoot     hinge       angle (rad)
# 8         - rootx     slider      velocity (m/s)
# 9         - rootz     slider      velocity (m/s)
#10         - rooty     hinge       angular velocity (rad/s)
#11         - bthigh    hinge       angular velocity (rad/s)
#12         - bshin     hinge       angular velocity (rad/s)
#13         - bfoot     hinge       angular velocity (rad/s)
#14         - fthigh    hinge       angular velocity (rad/s)
#15         - fshin     hinge       angular velocity (rad/s)
#16         - ffoot     hinge       angular velocity (rad/s)

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
        
        self.q_des = torch.zeros((transition_dim, horizon), requires_grad=False)
        #self.q_des[8,:] = 0
        #self.q_des[2,:] = -1.5708 #0.7854/4
        self.q_des[8,:] = 0.5 #m/s
        #self.q_des[10,:] = 0 #rad/s
                    

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
        q[:,8,:] = self.q_des[8,0]
        #print("x[:,8,:]: ", x[:,8,:])
        #q[:,10,:] = self.q_des[10,1]
        self.q_des = q
        self.q_des.requires_grad_(True).to("cuda")
        power = (self.q_des - x).pow(2)
        squared_norm = power.sum(axis=1)
        cost = squared_norm.sum(axis=1)     
        #print("squared of norm: ", cost)
        return cost
