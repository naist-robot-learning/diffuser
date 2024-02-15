import torch
import torch.nn as nn
import pdb
import numpy as np
import einops

# State-Space (name/joint/parameter):
# 0   6       - rootz     slider      position (m)
# 1   7       - rooty     hinge       angle (rad)
# 2   8       - bthigh    hinge       angle (rad)
# 3   9       - bshin     hinge       angle (rad)
# 4  10       - bfoot     hinge       angle (rad)
# 5  11       - fthigh    hinge       angle (rad)
# 6  12       - fshin     hinge       angle (rad)
# 7  13       - ffoot     hinge       angle (rad)
# 8  14       - rootx     slider      velocity (m/s)
# 9  15       - rootz     slider      velocity (m/s)
#10  16       - rooty     hinge       angular velocity (rad/s)
#11  17       - bthigh    hinge       angular velocity (rad/s)
#12  18       - bshin     hinge       angular velocity (rad/s)
#13  19       - bfoot     hinge       angular velocity (rad/s)
#14  20       - fthigh    hinge       angular velocity (rad/s)
#15  21       - fshin     hinge       angular velocity (rad/s)
#16  22       - ffoot     hinge       angular velocity (rad/s)

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
        
        #self.q_des = torch.zeros((64, transition_dim, horizon), requires_grad=False)
        
        #self.q_des[:,14,:] = 2 #m/s         
        #self.q_des.to("cuda")           

    def forward(self,
                x: torch.tensor((1,384,6)), 
                cond: torch.tensor((1,384,6)), 
                time: torch.tensor) -> torch.tensor(1):
        '''
            x : [ batch x horizon x transition ]
        '''
        x = einops.rearrange(x, 'b h t -> b t h')
        q = torch.tensor(x).to("cuda")
        #print("x[0,:,0]", x[0,:,0])
        #import ipdb; ipdb.set_trace()
        q[:, 2, :] = cond["obst"][0][0]
        q[:, 3, :] = cond["obst"][0][1]
        
        
 
        # print("q[0, 8, 0]: ", q[0, 8, 0])
        # print("self.q_des[0, 8,0]: ", self.q_des[0, 8,0])
        # print("x[0, 8, 0]: ", x[0, 8, 0])
        cond_vec = q
        d = cond_vec - x
        euclid = (d).pow(2).sum(axis=1).sqrt()
        eps = 1e-5
        if (euclid > 0.4).all():
            power= x.pow(2)*0.0
        else:
            power = -1/((d).pow(2)+eps)            # max of concave
        
        squared_norm = power.sum(axis=1)
        cost = squared_norm.sum(axis=1)     

        return cost
