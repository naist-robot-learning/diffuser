import torch
import torch.nn as nn
import pdb
import numpy as np
import einops
from diffuser.robot.UR5kinematicsAndDynamics_vectorized import compute_reflected_mass, fkine

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
#19  - x_hand    x position of hand pose (m)
#20  - y_hand    y position of hand pose (m)
#21  - z_hand    z position of hand pose (m)
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
        # batch_size = x.shape[0]
        # horizon = x.shape[1]
        # transition_dim = x.shape[2]
        # x = einops.rearrange(x, 'b h t -> b t h')
        # cost = torch.empty((batch_size,horizon))
        # u = torch.empty(3).to("cuda")
        # u[0] = 1; u[1] = 0; u[2] = 0
        # for i in range(0,batch_size):
        #     for j in range(0,horizon):
        #         cost[i,j] = compute_reflected_mass(x[i,6:12,j], u)           
        # final_cost = cost.sum(axis=1).sum(axis=0)
        # return final_cost
        
        ####### (TODO) VECTORISED IMPLEMENTATION ######## 
        batch_size = x.shape[0]
        horizon = x.shape[1]
        transition_dim = x.shape[2]
        x = einops.rearrange(x, 'b h t -> b t h')
        cost = torch.empty((batch_size, horizon)).to('cuda')
        u = torch.empty((batch_size*horizon, 3, 1), dtype=torch.float32).to('cuda')
        #u[:,0] = 1/(2)**(1/2); u[:,1] = 0; u[:,2] = 1/(2)**(1/2)
        #u[:,0] = 1; u[:,1] = 0; u[:,2] = 0
        #import ipdb; ipdb.set_trace()
        x_ = einops.rearrange(x, 'b t h -> t (b h)').to('cuda')
        x_tcp = fkine(x_[6:12,:])[:,0:3].unsqueeze(2)
        x_hand = x_[19:22,:].permute(1,0).unsqueeze(2)
        ## compute normalized direction vector from x_tcp to x_hand
        u = (x_hand - x_tcp)/torch.linalg.norm((x_hand - x_tcp), dim=1, ord=2).unsqueeze(2)
        cost = compute_reflected_mass(x[:,6:12,:], u)
        #cost = compute_reflected_mass(x[:,:6,:], u)
        final_cost = -1*cost.sum(axis=1)
        #import ipdb; ipdb.set_trace()
        return final_cost