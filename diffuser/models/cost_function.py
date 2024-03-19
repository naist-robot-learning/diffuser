import torch
import torch.nn as nn
import pdb
import numpy as np
import einops
from diffuser.robot.UR5kinematicsAndDynamics import compute_reflected_mass

# Trajectory vector (name/parameter):
# 0  - dq1       action angle (rad)
# 1  - dq2       action angle (rad)
# 2  - dq3       action angle (rad)
# 3  - dq4       action angle (rad)
# 4  - dq5       action angle (rad)
# 5  - dq6       action angle (rad)
# 6  - q1        angle (rad)
# 7  - q2        angle (rad)
# 8  - q3        angle (rad)
# 9  - q4        angle (rad)
#10  - q5        angle (rad)
#11  - q6        angle (rad)
#12  - x_goal    x position of goal pose (m)
#13  - y_goal    y position of goal pose (m)
#14  - z_goal    z position of goal pose (m)
#15  - q         1st element of quaternion
#16  - u         2nd element of quaternion
#17  - a         3rd element of quaternion
#18  - t         4th element of quaternion
#19  - x_hand    x position of goal pose (m)
#20  - y_hand    y position of goal pose (m)
#21  - z_hand    z position of goal pose (m)
#22  - q         1st element of quaternion
#23  - u         2nd element of quaternion
#24  - a         3rd element of quaternion
#25  - t         4th element of quaternion

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
                x: torch.tensor((64,32,26)), 
                cond: torch.tensor((64,32,26)), 
                time: torch.tensor) -> torch.tensor(64):
        ''' 
            x : [ batch x horizon x transition ]
        '''
        ###### NAIVE IMPLEMENTATION, BASICALLY USELESS ########
        x = einops.rearrange(x, 'b h t -> b t h')
        batch_size = x.shape[0]
        horizon = x.shape[1]
        transition_dim = x.shape[2]
        cost = torch.empty((batch_size,horizon))
        u = torch.empty(3).to("cuda")
        u[0] = 1; u[1] = 0; u[2] = 0
        for i in range(0,batch_size):
            for j in range(0,horizon):
                cost[i,j] = compute_reflected_mass(x[i,6:12,j], u)           
        final_cost = cost.sum(axis=1).sum(axis=0)
        return final_cost
        
        ####### (TODO) VECTORISED IMPLEMENTATION ######## 
        # x = einops.rearrange(x, 'b h t -> b t h')
        # batch_size = x.shape[0]
        # horizon = x.shape[1]
        # transition_dim = x.shape[2]
        # cost = torch.empty((batch_size,horizon))
        # u = torch.empty((3,1)).to("cuda")
        # u[0] = 1; u[1] = 0; u[2] = 0
        # for i in range(0,batch_size):
        #     cost[i,:] = compute_reflected_mass(x[i,6:12,:], u)
        # final_cost = cost.sum(axis=1)
        # return final_cost