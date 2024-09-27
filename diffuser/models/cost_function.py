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
        self._test_cost = test_cost
        self.transition_dim = transition_dim
        self._normalizer = normalizer
        tensor_args = {"device": "cuda", "dtype": torch.float32}
        trajectory_duration = 5.0
        dt = 0.1  # trajectory_duration / horizon
        self._smoothness_cost = CostGPTrajectoryPositionOnlyWrapper(
            n_dof=transition_dim, n_support_points=horizon, dt=dt, sigma_gp=1.0, tensor_args=tensor_args
        )
        action_dim = 0
        self._reflected_mass_cost = ReflectedMassCost(action_dim, batch_size, horizon)
        self._goal_pose_cost = GoalPoseCost()
        self._viapoint_costs = GoalPoseCost()
        self._Q1 = 1.0  # 0.02 # 0.0001
        self._Q2 = 1.0  # 1.0  # 1.5

    def forward(
        self,
        x: torch.tensor,
        cond: dict,
        time: torch.tensor,
    ) -> torch.tensor(64):
        """
        x : [ batch x horizon x transition ]
        """
        n = self.transition_dim
        horizon = x.shape[1]

        x = self._normalizer.unnormalize(x, "observations")

        sm_val = rm_val = gp_val = vp_val = 0.0
        cost = {}

        # Compute smoothness cost
        if self._Q1:
            sm_val = self._smoothness_cost(x[:, :, :n])
            cost["smoothness"] = sm_val

        # Compute reflected mass cost
        if self._Q2:
            rm_val = self._reflected_mass_cost(x, cond["hand_pose"])
            cost["refmass"] = rm_val

        # Compute goal pose cost
        last_horizon_step = -5
        if "goal_pose" in cond:
            gp_val = self._goal_pose_cost(x, last_horizon_step, cond["goal_pose"])
            # For testing
            # gp_val = torch.zeros(64).to("cuda") * x.sum(axis=1)[:, 0]
            cost["goal_pose"] = gp_val

        # Compute via point costs
        vp_val = 0
        cnt = 1
        for i, (t, value) in enumerate(cond.items()):
            if type(t) is str or t == 0:
                continue
            else:
                key_name = "via_point" + str(cnt)
                cost[key_name] = self._viapoint_costs(x, t, value)
                vp_val += cost[key_name]
                cnt += 1

        final_cost = sm_val + rm_val + gp_val + vp_val

        return cost, final_cost
