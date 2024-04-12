import torch
import torch.nn as nn
from diffuser.models.costs import CostGPTrajectoryPositionOnlyWrapper, ReflectedMassCost

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
        normalizer: object,
        dim_mults: tuple = (1, 2, 4, 8),
        out_dim: int = 1,
    ):
        super().__init__()
        self.transition_dim = transition_dim
        self._normalizer = normalizer

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
        horizon = x.shape[1]
        tensor_args = {"device": "cuda", "dtype": torch.float32}

        x = self._normalizer.unnormalize(x, "observations")
        if self.transition_dim == cond[0].shape[1]:
            action_dim = 0
        else:
            action_dim = self.transition_dim - cond[0].shape[1]

        trajectory_duration = 5.0
        dt = trajectory_duration / horizon

        cost_smoothness = CostGPTrajectoryPositionOnlyWrapper(
            n_dof=6, n_support_points=horizon, dt=dt, sigma_gp=1.0, tensor_args=tensor_args
        )

        cost_reflectedmass = ReflectedMassCost(action_dim)
        Q1 = 1.0
        Q2 = 1.0
        cs_val = cost_smoothness(x[:, :, :6])
        crm_val = cost_reflectedmass(x, cond)
        final_cost = Q1 * cs_val + Q2 * crm_val

        return final_cost
