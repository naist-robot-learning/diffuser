from abc import ABC, abstractmethod
from diffuser.robot.KUKALWR4KinematicsAndDynamics_vectorized import compute_reflected_mass, fkine
import einops
import numpy as np
import torch


def finite_difference_vector(x, dt=1.0, method="forward"):
    # finite differences with zero padding at the borders
    diff_vector = torch.zeros_like(x)
    if method == "forward":
        diff_vector[..., :-1, :] = torch.diff(x, dim=-2) / dt
    elif method == "backward":
        diff_vector[..., 1:, :] = (x[..., 1:, :] - x[..., :-1, :]) / dt
    elif method == "central":
        diff_vector[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) / (2 * dt)
    else:
        raise NotImplementedError
    return diff_vector


class Cost(ABC):
    def __init__(self, n_dof, horizon, tensor_args=None, **kwargs):
        self.n_dof = n_dof
        self.dim = 2 * n_dof  # position + velocity
        self.n_support_points = horizon

        self.tensor_args = tensor_args

    def set_cost_factors(self):
        pass

    def __call__(self, trajs, **kwargs):
        return self.eval(trajs, **kwargs)

    @abstractmethod
    def eval(self, trajs, **kwargs):
        pass

    @abstractmethod
    def get_linear_system(self, trajs, **kwargs):
        pass


class GPFactor:

    def __init__(
        self,
        dim,
        sigma,
        d_t,
        num_factors,
        tensor_args=None,
        Q_c_inv=None,
    ):
        self.dim = dim
        self.d_t = d_t
        self.tensor_args = tensor_args
        self.state_dim = self.dim * 2  # position and velocity
        self.num_factors = num_factors
        self.idx1 = torch.arange(0, self.num_factors, device=tensor_args["device"])
        self.idx2 = torch.arange(1, self.num_factors + 1, device=tensor_args["device"])
        self.phi = self.calc_phi()
        if Q_c_inv is None:
            Q_c_inv = torch.eye(dim, **tensor_args) / sigma**2
        self.Q_c_inv = torch.zeros(num_factors, dim, dim, **tensor_args) + Q_c_inv
        self.Q_inv = self.calc_Q_inv()  # shape: [num_factors, state_dim, state_dim]

        ## Pre-compute constant Jacobians
        self.H1 = self.phi.unsqueeze(0).repeat(self.num_factors, 1, 1)
        self.H2 = -1.0 * torch.eye(self.state_dim, **self.tensor_args).unsqueeze(0).repeat(
            self.num_factors,
            1,
            1,
        )

    def calc_phi(self):
        I = torch.eye(self.dim, **self.tensor_args)
        Z = torch.zeros(self.dim, self.dim, **self.tensor_args)
        phi_u = torch.cat((I, self.d_t * I), dim=1)
        phi_l = torch.cat((Z, I), dim=1)
        phi = torch.cat((phi_u, phi_l), dim=0)
        return phi

    def calc_Q_inv(self):
        m1 = 12.0 * (self.d_t**-3.0) * self.Q_c_inv
        m2 = -6.0 * (self.d_t**-2.0) * self.Q_c_inv
        m3 = 4.0 * (self.d_t**-1.0) * self.Q_c_inv

        Q_inv_u = torch.cat((m1, m2), dim=-1)
        Q_inv_l = torch.cat((m2, m3), dim=-1)
        Q_inv = torch.cat((Q_inv_u, Q_inv_l), dim=-2)
        return Q_inv

    def get_error(self, x_traj, calc_jacobian=True):
        batch, horizon = x_traj.shape[0], x_traj.shape[1]
        state_1 = torch.index_select(x_traj, 1, self.idx1).unsqueeze(-1)
        state_2 = torch.index_select(x_traj, 1, self.idx2).unsqueeze(-1)
        error = state_2 - self.phi @ state_1
        if calc_jacobian:
            H1 = self.H1
            H2 = self.H2
            # H1 = self.H1.unsqueeze(0).repeat(batch, 1, 1, 1)
            # H2 = self.H2.unsqueeze(0).repeat(batch, 1, 1, 1)
            return error, H1, H2
        else:
            return error


class CostGPTrajectory(Cost):

    def __init__(self, n_dof, n_support_points, dt, sigma_gp=None, **kwargs):
        super().__init__(n_dof, n_support_points, **kwargs)
        self.dt = dt

        self.sigma_gp = sigma_gp

        self.set_cost_factors()

    def set_cost_factors(self):
        # ========= Cost factors ===============
        self.gp_prior = GPFactor(
            self.n_dof,
            self.sigma_gp,
            self.dt,
            self.n_support_points - 1,
            self.tensor_args,
        )

    def eval(self, trajs, **observation):
        # trajs = trajs.reshape(-1, self.n_support_points, self.dim)

        # GP cost
        err_gp = self.gp_prior.get_error(trajs, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0]  # repeated Q_inv
        w_mat = w_mat.reshape(1, 1, self.dim, self.dim)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1).squeeze()
        costs = gp_costs
        return costs

    def get_linear_system(self, trajs, **observation):
        pass


class CostGPTrajectoryPositionOnlyWrapper(CostGPTrajectory):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, trajs, **observation):
        vel = finite_difference_vector(trajs, dt=self.dt, method="central")
        trajs_tmp = torch.cat((trajs, vel), dim=-1)
        return super().eval(trajs_tmp, **observation)


class ReflectedMassCost:
    def __init__(self, action_dim, batch_size, horizon) -> None:
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.horizon = horizon
        # self._cost = torch.empty((batch_size, horizon)).to("cuda")
        # self._u = torch.empty((batch_size * horizon, 3, 1), dtype=torch.float32).to("cuda")

    def __call__(self, x: torch.tensor, target_pose: torch.tensor) -> torch.tensor:
        """computes reflected mass cost in the direction of the hand_pose found in
        cond["hand_pose"]

        Args:
            x (torch.tensor): trajectories. Can include: joint positions,
                                                        tcp position,
                                                        hand position
            cond (dict): include first timestep of state and "hand_pose"

        Returns:
            torch.tensor: cost in batch shape
        """
        n_dof = 7
        x = einops.rearrange(x, "b h t -> b t h")
        x_ = einops.rearrange(x, "b t h -> t (b h)")
        x_tcp = fkine(x_[self.action_dim : self.action_dim + n_dof, :])[:, :3].unsqueeze(2)
        x_hand = target_pose[:, :3].unsqueeze(2).repeat(self.horizon, 1, 1)

        ## compute normalized direction vector from x_tcp to x_hand
        u = (x_hand - x_tcp) / torch.linalg.norm((x_hand - x_tcp), dim=1, ord=2).unsqueeze(2)
        cost = compute_reflected_mass(x[:, self.action_dim : self.action_dim + n_dof, :], u)

        # cost = compute_reflected_mass(x[:,:6,:], u)
        final_cost = cost.sum(axis=1)
        return final_cost


class GoalPoseCost:
    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.tensor, t, value) -> torch.tensor:
        """computes cost of reaching the goal pose

        Args:
            x (torch.tensor): trajectories. Could include: joint positions,
                                                        tcp position,
                                                        hand position

        Returns:
            torch.tensor: cost in batch shape
        """
        batch_size, horizon, transition_dim = x.shape
        # cost = torch.empty((batch_size, horizon)).to("cuda")
        """.unsqueeze(dim=1)"""
        x_ = einops.rearrange(x[:, t, :transition_dim].unsqueeze(dim=1), "b h t-> t (b h)").to("cuda")
        x_tcp = fkine(x_)[:, :3]
        x_tcp = einops.rearrange(x_tcp, "(b h) t -> b h t", b=batch_size, h=1)
        x_goal = value[:, :3].unsqueeze(1).repeat(1, 1, 1)  # x_goal shape now (batch_size, 1, position)
        cost = torch.linalg.norm((x_goal - x_tcp), dim=2, ord=2)
        final_cost = cost.sum(axis=1)
        return final_cost
