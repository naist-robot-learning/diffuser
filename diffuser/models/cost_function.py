import torch
import torch.nn as nn
from diffuser.models.costs import (
    CostGPTrajectoryPositionOnlyWrapper,
    ReflectedMassCost,
    GoalPoseCost,
)

# Trajectory vector (name/parameter):
# 1  - q1        angle (rad)
# 2  - q2        angle (rad)
# 3  - q3        angle (rad)
# 4  - q4        angle (rad)
# 5  - q5        angle (rad)
# 6  - q6        angle (rad)


class CostFn(nn.Module):

    def __init__(
        self,
        horizon: int,
        batch_size: int,
        action_dim: int,
        transition_dim: int,
        test_cost: str,
        cond_dim: int,
        normalizer: object,
        dim_mults: tuple = (1, 2, 4, 8),
        out_dim: int = 1,
    ):
        super().__init__()
        self._test_cost=test_cost
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
        self._Q1 = 0.0#0.02 # 0.0001
        self._Q2 = 1.0    # 1.5
        self._Q3 = 1000.0    # 5.0 #3000 with Q1

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
        
        #grad_cost = self._Q1 * cs_val + self._Q2 * crm_val + self._Q3 * cgp_val
        grad_cost = self._Q3 * cgp_val
        if 'Q1' in self._test_cost:
            grad_cost += self._Q1 * cs_val
        elif 'Q2' in self._test_cost:
            grad_cost += self._Q2 * crm_val
            
        final_cost = self._Q1 * cs_val + self._Q2 * crm_val

        return grad_cost, final_cost
