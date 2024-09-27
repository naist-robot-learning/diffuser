from abc import ABC, abstractmethod
from diffuser.robot.UR5kinematicsAndDynamics_vectorized import compute_reflected_mass, fkine
import einops
import kornia
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

        n_dof = 6
        b, h, t = x.shape
        x = einops.rearrange(x, "b h t -> b t h")
        x_ = einops.rearrange(x, "b t h -> t (b h)")
        x_tcp, _ = fkine(x_[self.action_dim : self.action_dim + n_dof, :])
        x_tcp = x_tcp[:, :3].unsqueeze(2)
        x_hand = target_pose[:, :3].unsqueeze(2).repeat(self.horizon, 1, 1)

        ## compute normalized direction vector from x_tcp to x_hand

        # u = (x_hand - x_tcp) / torch.linalg.norm((x_hand - x_tcp), dim=1, ord=2).unsqueeze(2)
        u = torch.zeros((b * h, 3, 1)).to("cuda")
        u[:, 0] = 1
        cost = compute_reflected_mass(x[:, self.action_dim : self.action_dim + n_dof, :], u)

        # cost = compute_reflected_mass(x[:,:6,:], u)
        final_cost = cost.sum(axis=1)
        # import ipdb; ipdb.set_trace()
        return final_cost


class GoalPoseCost:
    def __init__(self) -> None:
        self.gamma = 0.35

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
        sth, other = fkine(einops.rearrange(x, "b h t-> t (b h)").to("cuda"))

        other = kornia.geometry.Quaternion.from_matrix(other[:, :3, :3]).data
        other_xyzw = torch.cat((other[:, 1:], other[:, :1]), dim=1)  # Now in (x, y, z, w)
        other = einops.rearrange(other_xyzw, "(b h) t -> b h t", b=batch_size, h=horizon)

        sth = einops.rearrange(sth, "(b h) t -> b h t", b=batch_size)


        if abs(t) > 1:
            x_ = einops.rearrange(x[:, t:, :transition_dim], "b h t-> t (b h)").to("cuda")
        else:
            x_ = einops.rearrange(x[:, t:, :transition_dim].unsqueeze(dim=1), "b h t-> t (b h)").to("cuda")
        x_tcp, H = fkine(x_)
        rpy_tcp = x_tcp[:, 3:]
        x_tcp = x_tcp[:, :3]

        x_tcp = einops.rearrange(x_tcp, "(b h) t -> b h t", b=batch_size, h=abs(t))
        rpy_tcp = einops.rearrange(rpy_tcp, "(b h) t -> b h t", b=batch_size, h=abs(t))

        # Position
        x_goal = value[:, :3].unsqueeze(1).repeat(1, abs(t), 1)  # x_goal shape now (batch_size, 1, position)
        position_cost = torch.linalg.norm((x_goal - x_tcp), dim=2, ord=2)

        # Orientation

        q_tcp = kornia.geometry.Quaternion.from_matrix(H[:, :3, :3]).data
        # Rearrange from (w, x, y, z) to (x, y, z, w)
        q_tcp_xyzw = torch.cat((q_tcp[:, 1:], q_tcp[:, :1]), dim=1)  # Now in (x, y, z, w)
        q_tcp = einops.rearrange(q_tcp_xyzw, "(b h) t -> b h t", b=batch_size)

        q_goal = value[:, 3:].unsqueeze(1).repeat(1, abs(t), 1)

        q_goal_inv = torch.cat((-q_goal[:, :, :3], q_goal[:, :, 3:]), dim=2)  # In (x, y, z, w)
        q_diff = self.multiply_quaternions(q_tcp, q_goal_inv)

        logmap_Q = self.quaternion_LogMap(q_diff)

        orientation_cost = torch.linalg.norm(logmap_Q, dim=2, ord=2)

        final_cost = self.gamma * (position_cost.sum(axis=1)) + (1 - self.gamma) * orientation_cost.sum(axis=1)

        return final_cost

    def quaternion_LogMap(self, Q):
        """
        Computes the logarithmic map of a unit quaternion, projecting it to the tangent space.

        Args:
            q (torch.Tensor: BxHxT): A tensor of shape (4,) representing a quaternion [x, y, z, w]. B=64, H=5, T=6

        Returns:
            torch.Tensor: A tensor of shape (3,) representing the tangent vector.
        """

        # Extract the real and imaginary parts
        x, y, z, w = Q[:, :, 0], Q[:, :, 1], Q[:, :, 2], Q[:, :, 3]

        # Calculate the angle theta
        theta = 2 * torch.atan(1 / w).clamp(min=1e-8)  # Clamp to avoid division by zero

        sin_theta = torch.sin(theta / 2)
        threshold = torch.zeros_like(sin_theta) * 1e-8
        # Compute the tangent vector
        scale = torch.where(sin_theta.abs() > threshold, theta / sin_theta, torch.zeros_like(theta))
        delta_w = torch.stack([scale * x, scale * y, scale * z], axis=2)

        return delta_w

    def euler_to_quaternion(self, rpy):

        yaw = rpy[:, :, 0]
        pitch = rpy[:, :, 1]
        roll = rpy[:, :, 2]
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return torch.stack([qx, qy, qz, qw], axis=2)

    def multiply_quaternions(self, q0, q1):
        """Quaternion multiplication

        Args:
            q0 (torch.tensor): Quaternion in x,y,z,w
            q1 (torch.tensor): Quaternion in x,y,z,w

        Returns:
            t: Quaternion in x,y,z,w
        """

        q_0 = q0[:, :, 3]
        q_1 = q0[:, :, 0]
        q_2 = q0[:, :, 1]
        q_3 = q0[:, :, 2]

        r0 = q1[:, :, 3]
        r1 = q1[:, :, 0]
        r2 = q1[:, :, 1]
        r3 = q1[:, :, 2]

        t0 = r0 * q_0 - r1 * q_1 - r2 * q_2 - r3 * q_3
        t1 = r0 * q_1 + r1 * q_0 - r2 * q_3 + r3 * q_2
        t2 = r0 * q_2 + r1 * q_3 + r2 * q_0 - r3 * q_1
        t3 = r0 * q_3 - r1 * q_2 + r2 * q_1 + r3 * q_0

        return torch.stack([t1, t2, t3, t0], axis=2)
