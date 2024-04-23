import torch
import torch.nn as nn
from diffuser.models.costs import (
    CostGPTrajectoryPositionOnlyWrapper,
    ReflectedMassCost,
    GoalPoseCost,
)

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
        batch_size: int,
        action_dim: int,
        transition_dim: int,
        cond_dim: int,
        normalizer: object,
        dim_mults: tuple = (1, 2, 4, 8),
        out_dim: int = 1,
    ):
        super().__init__()
        self.transition_dim = transition_dim
        self._normalizer = normalizer
        tensor_args = {"device": "cuda", "dtype": torch.float32}
        trajectory_duration = 5.0
        dt = trajectory_duration / horizon
        self._smoothness_cost = CostGPTrajectoryPositionOnlyWrapper(
            n_dof=6, n_support_points=horizon, dt=dt, sigma_gp=1.0, tensor_args=tensor_args
        )
        action_dim = 0
        self._reflected_mass_cost = ReflectedMassCost(action_dim, batch_size, horizon)
        self._goal_pose_cost = GoalPoseCost()
        self._Q1 = 0.0 # 0.0001
        self._Q2 = 0.0    # 1.5
        self._Q3 = 1.0    # 5.0

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

        x = self._normalizer.unnormalize(x, "observations")
        
        if self._Q1:
            cs_val = self._smoothness_cost(x[:, :, :6])
        else:
            cs_val = 0
            
        if self._Q2:
            crm_val = self._reflected_mass_cost(x, cond)
        else:
            crm_val = 0
            
        if self._Q3:
            cgp_val = self._goal_pose_cost(x, cond)
        else:
            cgp_val = 0
            
        final_cost = self._Q1 * cs_val + self._Q2 * crm_val + self._Q3 * cgp_val

        return final_cost
