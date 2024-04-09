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
# 10  - q5        angle (rad)
# 11  - q6        angle (rad)
# 12  - x_goal    x position of goal pose (m)
# 13  - y_goal    y position of goal pose (m)
# 14  - z_goal    z position of goal pose (m)
# 15  - q         1st element of quaternion
# 16  - u         2nd element of quaternion
# 17  - a         3rd element of quaternion
# 18  - t         4th element of quaternion
# 19  - x_hand    x position of hand pose (m)
# 20  - y_hand    y position of hand pose (m)
# 21  - z_hand    z position of hand pose (m)
# 22  - q         1st element of quaternion
# 23  - u         2nd element of quaternion
# 24  - a         3rd element of quaternion
# 25  - t         4th element of quaternion


class CostFn(nn.Module):

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        cond_dim: int,
        dim_mults: tuple = (1, 2, 4, 8),
        out_dim: int = 1,
    ):
        super().__init__()
        self.transition_dim = transition_dim

        # self.q_des = torch.zeros((64, transition_dim, horizon), requires_grad=False)

        # self.q_des[:,14,:] = 2 #m/s
        # self.q_des.to("cuda")

    def forward(
        self,
        x: torch.tensor,
        cond: dict,
        time: torch.tensor,
    ) -> torch.tensor(64):
        """
        x : [ batch x horizon x transition ]
        """
        if self.transition_dim == cond[0].shape[1]:
            action_dim = 0
        else:
            action_dim = self.transition_dim - cond[0].shape[1]

        batch_size = x.shape[0]
        horizon = x.shape[1]
        transition_dim = x.shape[2]
        x = einops.rearrange(x, "b h t -> b t h")
        cost = torch.empty((batch_size, horizon)).to("cuda")
        u = torch.empty((batch_size * horizon, 3, 1), dtype=torch.float32).to("cuda")
        # u[:,0] = 1/(2)**(1/2); u[:,1] = 0; u[:,2] = 1/(2)**(1/2)
        # u[:,0] = 1; u[:,1] = 0; u[:,2] = 0
        # import ipdb; ipdb.set_trace()
        x_ = einops.rearrange(x, "b t h -> t (b h)").to("cuda")
        x_tcp = fkine(x_[action_dim : action_dim + 6, :])[:, :3].unsqueeze(2)
        hand_idx = action_dim + 6 + 3
        x_hand = x_[hand_idx : hand_idx + 3, :].permute(1, 0).unsqueeze(2)
        ## compute normalized direction vector from x_tcp to x_hand
        u = (x_hand - x_tcp) / torch.linalg.norm((x_hand - x_tcp), dim=1, ord=2).unsqueeze(2)
        cost = compute_reflected_mass(x[:, action_dim : action_dim + 6, :], u)
        # cost = compute_reflected_mass(x[:,:6,:], u)
        final_cost = -1 * cost.sum(axis=1)
        # import ipdb; ipdb.set_trace()
        return final_cost
