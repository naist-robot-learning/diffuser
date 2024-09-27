import os
import numpy as np
import einops
from diffuser.robot.UR5kinematicsAndDynamics_vectorized import compute_reflected_mass, fkine
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


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


def get_velocity(x, dt=0.1):
    # If there is no velocity in the state, then compute it via finite difference
    vel = finite_difference_vector(x, dt=dt, method="central")
    return vel


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_with_axes(ax, x, y, z, roll, pitch, yaw):
    for i in range(len(x)):
        # Define rotation matrix from roll, pitch, and yaw at each point
        rotation_matrix = R.from_euler("zyx", [roll[i], pitch[i], yaw[i]], degrees=False).as_matrix()

        # Define unit vectors along X, Y, and Z axes
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])

        # Rotate unit vectors using rotation matrix
        rotated_x_axis = np.dot(rotation_matrix, x_axis)
        rotated_y_axis = np.dot(rotation_matrix, y_axis)
        rotated_z_axis = np.dot(rotation_matrix, z_axis)

        # Plot rotated coordinate system vectors at each point
        if i % 5 == 0 or i == len(x) - 1:
            ax.quiver(
                x[i],
                y[i],
                z[i],
                rotated_x_axis[0],
                rotated_x_axis[1],
                rotated_x_axis[2],
                color="r",
                length=0.015,
                normalize=True,
            )
            ax.quiver(
                x[i],
                y[i],
                z[i],
                rotated_y_axis[0],
                rotated_y_axis[1],
                rotated_y_axis[2],
                color="g",
                length=0.015,
                normalize=True,
            )
            ax.quiver(
                x[i],
                y[i],
                z[i],
                rotated_z_axis[0],
                rotated_z_axis[1],
                rotated_z_axis[2],
                color="b",
                length=0.015,
                normalize=True,
            )
        else:
            ax.plot(x[i], y[i], z[i])


def visualize_trajectories(directory):
    # Get list of all .npz files in the directory
    npz_files = [file for file in os.listdir(directory) if file.endswith(".npz")]
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Loop through each .npz file
    file = {}
    cnt = 0
    # Initialize an empty list to store all the loaded arrays
    all_data = []
    cnt = 0

    # Assuming npz_files and directory are defined
    for npz_file in npz_files:
        data = {}
        print("File name: ", npz_file)

        # Load arrays from the .npz file
        npz_data = np.load(os.path.join(directory, npz_file))

        # Access the 'joint_position' array from the loaded .npz file and squeeze to remove unnecessary dimensions
        q = npz_data["joint_position"].squeeze()  # Shape: (Trials x horizon x transition)

        if "LinearInterpolation" not in npz_file:
            # q = q[300:1300, :]  # Slice to keep the range 300:1300
            # Select 40 evenly spaced indices between 0 and the length of the sliced range
            num_points = 40

            indices = np.linspace(0, q.shape[0] - 1, num_points).astype(int)
            q = q[indices, :]

            q[:, 0] = q[:, 0] - torch.pi / 2

        # q = q[:, :-1, 0]
        import ipdb

        ipdb.set_trace()

        # Add the loaded data to the list
        all_data.append(q)

        # Debugging: check the first few files if necessary
        if cnt < 2 and len(npz_files) > 1:
            cnt += 1
            continue

        # Concatenate all data along the first dimension (axis 0)
        q = np.stack(all_data, axis=0)  # Concatenating along dimension 0

        if len(q.shape) < 3:
            q_torch = torch.tensor(q, dtype=torch.float32).unsqueeze(dim=0)
        else:
            q_torch = torch.tensor(q, dtype=torch.float32)
        # Compute velocities
        q_dot = get_velocity(q_torch, 0.01)
        q_torch = einops.rearrange(q_torch, "b h t -> b t h")
        b, t, h = q_torch.shape

        ux = torch.zeros_like(q_torch[:, :3, :])  # .view(b * h, 3).unsqueeze(dim=2).to("cuda")
        uy = torch.zeros_like(q_torch[:, :3, :])  # .view(b * h, 3).unsqueeze(dim=2).to("cuda")
        uz = torch.zeros_like(q_torch[:, :3, :])  # .view(b * h, 3).unsqueeze(dim=2).to("cuda")

        ux = einops.rearrange(ux, "b t h -> (b h) t").unsqueeze(dim=2).to("cuda")
        uy = einops.rearrange(uy, "b t h -> (b h) t").unsqueeze(dim=2).to("cuda")
        uz = einops.rearrange(uz, "b t h -> (b h) t").unsqueeze(dim=2).to("cuda")

        ux[:, 0, :] = 1
        uy[:, 1, :] = 1
        uz[:, 2, :] = 1

        rm_x = compute_reflected_mass(q_torch, ux)
        rm_y = compute_reflected_mass(q_torch, uy)
        rm_z = compute_reflected_mass(q_torch, uz)
        import ipdb

        ipdb.set_trace()
        data["RM_x_y_z"] = np.array([rm_x.cpu().numpy(), rm_y.cpu().numpy(), rm_z.cpu().numpy()])
        data["q_b_h_t"] = q
        data["q_dot_b_h_t"] = q_dot

        print("File name: ", npz_file)
        print("Avg rm_x per Trajectory: ", rm_x.mean(dim=1))
        print("Avg_rm_y per Trajectory: ", rm_y.mean(dim=1))
        print("Avg_rm_z per Trajectory: ", rm_z.mean(dim=1))

        print("Avg rm_x overall exp: ", rm_x.mean())
        print("Avg_rm_y overall exp: ", rm_y.mean())
        print("Avg_rm_z overall exp: ", rm_z.mean())

        print("Max RM_x per Trajectory: ", rm_x.max(dim=1)[0])
        print("Min RM_x per Trajectory: ", rm_x.min(dim=1)[0])
        print("-----------------------------")
        print("Max RM_x overall: ", rm_x.max())
        print("Min RM_x overall: ", rm_x.min())

        print("Max RM_y per Trajectory: ", rm_y.max(dim=1)[0])
        print("Min RM_y per Trajectory: ", rm_y.min(dim=1)[0])
        print("-----------------------------")
        print("Max RM_y overall: ", rm_y.max())
        print("Min RM_y overall: ", rm_y.min())
        print("Max RM_z per Trajectory: ", rm_z.max(dim=1)[0])
        print("Min RM_z per Trajectory: ", rm_z.min(dim=1)[0])
        print("-----------------------------")
        print("Max RM_z overall: ", rm_z.max())
        print("Min RM_z overall: ", rm_z.min())

        # assert q_torch == 3  # batch, horizon, state_dim
        q_torch = einops.rearrange(q_torch, "b t h -> t (b h)")

        x_poses, _ = fkine(q_torch.to("cuda"))
        x_poses = einops.rearrange(x_poses, "(b h) t -> b h t", h=h)[:, :, :3]
        x_dot = get_velocity(x_poses, 0.002)

        data["x_b_h_t"] = x_poses.cpu().numpy()
        data["x_dot_b_h_t"] = x_dot.cpu().numpy()

        velocity_magnitude = torch.linalg.norm(x_dot, dim=-1)
        # Select a batch element to plot
        batch_idx = 0
        time_steps = torch.arange(velocity_magnitude.shape[1]) * 0.01
        time_steps = time_steps.cpu().numpy()
        velocity_magnitude = velocity_magnitude.cpu().numpy()

        sum_var_waypoints = 0.0
        for via_points in x_poses.permute(1, 0, 2):  # horizon, batch, position
            parwise_distance_between_points_via_point = torch.cdist(via_points, via_points, p=2)
            distances = torch.triu(parwise_distance_between_points_via_point, diagonal=1).view(-1)
            sum_var_waypoints += torch.var(distances)

        print("way point variance: ", sum_var_waypoints)
        data["x_batch_variance"] = sum_var_waypoints.cpu().numpy()

        q_torch = einops.rearrange(q_torch, "t (b h) ->  b h t", h=h, b=b)

        sum_var_waypoints = 0.0
        for via_points in q_torch.permute(1, 0, 2):  # horizon, batch, position
            parwise_distance_between_points_via_point = torch.cdist(via_points, via_points, p=2)
            distances = torch.triu(parwise_distance_between_points_via_point, diagonal=1).view(-1)
            sum_var_waypoints += torch.var(distances)

        data["q_batch_variance"] = sum_var_waypoints.cpu().numpy()

        print("way point variance config space: ", sum_var_waypoints)
        q = torch.tensor(q).unsqueeze(dim=0)
        for i in range(0, len(q)):
            if i % 1 == 0:
                # print("i: ", i)

                # if len(q[i].shape) > 2:
                #     qt = torch.tensor(q[i].squeeze())
                # else:

                if len(q.shape) > 2:
                    qt = torch.tensor(q).squeeze()
                else:
                    qt = torch.tensor(q).unsqueeze(0)

                if len(qt.shape) < 3:
                    qt = qt.unsqueeze(0)

                qt = einops.rearrange(qt, "b h t -> t (b h)").to("cuda")
                pose, _ = fkine(qt)
                pose = pose.cpu().numpy()
                pose = einops.rearrange(pose, "(b h) t -> b t h", b=b, h=h)
                x, y, z = pose[:, 0, :], pose[:, 1, :], pose[:, 2, :]

                # Combine reflected masses (or choose one direction)

                ax.scatter(
                    x[:, 0],
                    y[:, 0],
                    z[:, 0],
                    marker="o",
                    color="b",
                    s=100,
                )

                # ax.scatter(x[1:], y[1:], z[1:], s=50)
                ax.scatter(x[:, 1:], y[:, 1:], z[:, 1:], s=50)

                # plot_with_axes(ax, x, y, z, roll, pitch, yaw)
        import ipdb

        ipdb.set_trace()
        # ax.text(x[:, 0] + 0.01, y[:, 0] - 0.15, z[:, 0] + 0.17, "Start", fontsize=11, ha="left")
        ax.scatter(x[:, -1], y[:, -1], z[:, -1], marker="o", s=100, color="g")
        # ax.text(x[:, -1] + 0.01, y[:, -1] - 0.15, z[:, -1] - 0.17, "Goal", fontsize=11, ha="left")
        # # Set common axis limits
        # xlim = (0.1, 0.7)
        # ylim = (-0.3, 0.3)
        # zlim = (0.0, 0.6)

        # # Set consistent axis limits and ticks
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        # ax.set_zlim(zlim)
        # ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.1, 0.1))
        # ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.1, 0.1))
        # ax.set_zticks(np.arange(zlim[0], zlim[1] + 0.1, 0.1))
        ax.tick_params(axis="both", which="major", labelsize=9)

        # Apply the set_axes_equal function to ensure equal aspect ratios
        set_axes_equal(ax)

        # Increase the size of axis labels and ticks
        # ax.tick_params(axis="both", which="major", labelsize="14")

        ax.set_xlabel(r"X (m)", fontsize="12", labelpad=15)
        ax.set_ylabel(r"Y (m)", fontsize="12", labelpad=15)
        ax.set_zlabel(r"Z (m)", fontsize="12", labelpad=15)
        # ax.view_init(elev=19, azim=-19)
        ax.view_init(elev=19, azim=45)
        # Add color bar for reference
        filename = f"0_output_middle_x_min_results"
        np.savez(f"/home/ws/src/diffuser/Experiments/{exp_name}/{filename}.npz", **data)

    # plt.tight_layout()
    # ax.title.set_position([0.5, 0.01])
    # cax.set_xlabel("Reflected Mass", fontsize=12)

    # plt.rcParams["text.usetex"] = False
    # plt.rcParams["axes.titley"] = 0.5
    # plt.rcParams["axes.titlepad"] = 0.1
    # plt.tight_layout()
    plt.show()


# Directory containing .npz files
# directory = (
#     "logs/ur5_coppeliasim_full_path/plans/release_H48_T25_LimitsNormalizer_b64_condFalse/0/"  # Current directory
# )
directory = (
    "logs/ur5_coppeliasim_full_path/plans/release_H48_T25_LimitsNormalizer_b64_condFalse/0/"  # Current directory
)
exp_name = "C10_C14_diffuser"
exp_name = "RealRobotTomm/Exp_middle/output"
directory = f"/home/ws/src/diffuser/Experiments/{exp_name}"  # /new_ones"  # Current directory
# exp_name = "exp10_tomm_Diffusion-Model_x"
# directory = f"/home/ws/src/diffuser/logs/tomm_coppeliasim_full_path/plans/release_H48_T25_LimitsNormalizer_b64_condFalse/0/{exp_name}"
# directory = "/home/ws/src/diffuser/Experiments/C11_C8"  # Current directory
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
# Compute metric on arrays in .npz files in the directory
visualize_trajectories(directory)
