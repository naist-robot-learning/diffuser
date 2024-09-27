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
    # ax = fig.add_subplot(111, projection="3d")
    # axes = [
    #     fig.add_subplot(221, projection="3d"),
    #     fig.add_subplot(222, projection="3d"),
    #     fig.add_subplot(223, projection="3d"),
    #     fig.add_subplot(224, projection="3d"),
    # ] [left, bottom, width, height]
    ax1 = fig.add_axes([0.05, 0.53, 0.45, 0.45], projection="3d")
    ax2 = fig.add_axes([0.35, 0.53, 0.45, 0.45], projection="3d")
    ax3 = fig.add_axes([0.05, 0.05, 0.45, 0.45], projection="3d")
    ax4 = fig.add_axes([0.35, 0.05, 0.45, 0.45], projection="3d")
    axes = [ax1, ax2, ax3, ax4]

    # Loop through each .npz file
    file = {}
    for npz_file in npz_files:
        data = {}
        # Load arrays from the .npz file
        print("File name: ", npz_file)
        # Original string
        input_string = f"{npz_file}"

        # Split the string by '_'
        parts = input_string.split("_")

        # Extract the relevant parts
        algorithm = parts[1]  # RRT

        start_end = parts[2:4]  # C11_C8
        axis = parts[4]  # x
        metric = parts[5]  # max
        metric = metric[:-4]  # remove the .npz
        if metric == "max" or axis == "z" or axis == "y":
            continue

        npz_data = np.load(os.path.join(directory, npz_file))
        q = npz_data["joint_position"]  # Trials x horizon x transition 100 x 48 x 6

        if len(q.shape) > 3:
            q_torch = torch.tensor(q, dtype=torch.float32).squeeze(dim=1)
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

        x_poses = fkine(q_torch.to("cuda"))
        x_poses = einops.rearrange(x_poses, "(b h) t -> b h t", h=48)[:, :, :3]
        x_dot = get_velocity(x_poses, 0.002)

        data["x_b_h_t"] = x_poses.cpu().numpy()
        data["x_dot_b_h_t"] = x_dot.cpu().numpy()

        velocity_magnitude = torch.linalg.norm(x_dot, dim=-1)
        # Select a batch element to plot
        batch_idx = 0
        time_steps = torch.arange(velocity_magnitude.shape[1]) * 0.01
        time_steps = time_steps.cpu().numpy()
        velocity_magnitude = velocity_magnitude.cpu().numpy()

        # Plot the velocity magnitude over time for the selected batch element
        # plt.figure(figsize=(12, 6))
        # plt.plot(time_steps, velocity_magnitude[batch_idx], label="Overall Velocity")
        # plt.xlabel("Time Step")
        # plt.ylabel("Velocity Magnitude")
        # plt.title(f"Overall Velocity vs. Time for Batch Element {batch_idx}")
        # plt.grid(True)
        # plt.show()

        # plt.figure(figsize=(12, 6))
        # plt.plot(time_steps, x_dot[0, :, 0].cpu().numpy(), label="Velocity X")
        # plt.plot(time_steps, x_dot[0, :, 1].cpu().numpy(), label="Velocity Y")
        # plt.plot(time_steps, x_dot[0, :, 2].cpu().numpy(), label="Velocity Z")
        # plt.xlabel("Time Step")
        # plt.ylabel("Velocity")
        # plt.title(f"Velocity vs. Time for Batch Element {batch_idx}")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        sum_var_waypoints = 0.0
        for via_points in x_poses.permute(1, 0, 2):  # horizon, batch, position
            parwise_distance_between_points_via_point = torch.cdist(via_points, via_points, p=2)
            distances = torch.triu(parwise_distance_between_points_via_point, diagonal=1).view(-1)
            sum_var_waypoints += torch.var(distances)

        print("way point variance: ", sum_var_waypoints)
        data["x_batch_variance"] = sum_var_waypoints.cpu().numpy()

        q_torch = einops.rearrange(q_torch, "t (b h) ->  b h t", h=48)

        sum_var_waypoints = 0.0
        for via_points in q_torch.permute(1, 0, 2):  # horizon, batch, position
            parwise_distance_between_points_via_point = torch.cdist(via_points, via_points, p=2)
            distances = torch.triu(parwise_distance_between_points_via_point, diagonal=1).view(-1)
            sum_var_waypoints += torch.var(distances)

        data["q_batch_variance"] = sum_var_waypoints.cpu().numpy()

        print("way point variance config space: ", sum_var_waypoints)

        for i in range(0, len(q)):
            if i % 1 == 0:
                # print("i: ", i)

                if len(q[i].shape) > 2:
                    qt = torch.tensor(q[i].squeeze())
                else:
                    qt = torch.tensor(q[i])
                qt = einops.rearrange(qt, "h t -> t h").to("cuda")
                pose = fkine(qt).cpu().numpy()
                x, y, z = pose[:, 0], pose[:, 1], pose[:, 2]
                roll, pitch, yaw = pose[:, 3], pose[:, 4], pose[:, 5]

                # Combine reflected masses (or choose one direction)
                if axis == "x":
                    # print(f"Axis: {axis}")
                    rm_total = rm_x[i].cpu().numpy()  # + rm_y + rm_z  # Example of combining
                    ax = axes[1]
                    ax1 = axes[3]
                elif axis == "y":
                    rm_total = rm_y[i].cpu().numpy()  # + rm_y + rm_z  # Example of combining
                    ax = axes[2]
                elif axis == "z":
                    rm_total = rm_z[i].cpu().numpy()  # + rm_y + rm_z  # Example of combining
                    ax = axes[3]
                else:
                    rm_total = rm_x[i].cpu().numpy()  # + rm_y + rm_z  # Example of combining
                    ax = axes[0]
                    ax1 = axes[2]
                ax.scatter(
                    x[0],
                    y[0],
                    z[0],
                    marker="o",
                    color="b",
                    s=100,
                )
                ax1.scatter(
                    x[0],
                    y[0],
                    z[0],
                    marker="o",
                    color="b",
                    s=100,
                )

                # Normalize reflected mass values for color mapping
                norm = plt.Normalize(0.0, 0.4)
                colors = plt.cm.viridis(norm(rm_total))

                # Plot the trajectory with varying color based on reflected mass
                for j in range(len(x) - 1):
                    ax.scatter(x[j : j + 2], y[j : j + 2], z[j : j + 2], color=colors[j], s=4)
                    ax1.scatter(x[j : j + 2], y[j : j + 2], z[j : j + 2], color=colors[j], s=4)

                ax1.view_init(elev=87, azim=0)
                ax1.set_zticks([])  # Remove z-axis ticks
                ax1.set_zticklabels([])
                ax1.set_title("Top Down View", y=0.95)
                ax1.set_xlabel(r"X (m)", fontsize="12", labelpad=15)
                ax1.set_ylabel(r"Y (m)", fontsize="12", labelpad=15)

                # ax.scatter(x[1:], y[1:], z[1:], s=5)

                # plot_with_axes(ax, x, y, z, roll, pitch, yaw)

        ax.text(x[0] + 0.01, y[0] - 0.15, z[0] + 0.17, "Start", fontsize=11, ha="left")
        ax.scatter(x[-1], y[-1], z[-1], marker="o", s=100, color="g")
        ax.text(x[-1] + 0.01, y[-1] - 0.15, z[-1] - 0.17, "Goal", fontsize=11, ha="left")
        # Set common axis limits
        xlim = (0.1, 0.7)
        ylim = (-0.3, 0.3)
        zlim = (0.0, 0.6)

        # Set consistent axis limits and ticks
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.1, 0.1))
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.1, 0.1))
        ax.set_zticks(np.arange(zlim[0], zlim[1] + 0.1, 0.1))
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
        mappable = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        mappable.set_array(rm_total)

        # Split the start_end part to get start and end nodes
        start, end = start_end[0].split("C")[1], start_end[1].split("C")[1]

        # Create the title using string formatting
        title = f"$Path\ C{start} \rightarrow C{end}: {metric}\ m_{axis}(q)$"
        title = f"Path C{start} \u2192 C{end}: RI {metric}imization along axis-{axis}"
        title = f"Reflected Inertia optimization ({metric}) \n C{start} \u2192 C{end}: axis-{axis}"
        ax.set_title(title, fontsize=12, y=0.95)
        filename = f"{npz_file}_results"
        np.savez(f"{filename}.npz", **data)

    # Create a color bar manually on the left side
    cbar_ax = fig.add_axes([0.75, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    fig.colorbar(mappable, cax=cbar_ax, ax=axes, label="Reflected Inertia (Kg)")
    # plt.tight_layout()
    # ax.title.set_position([0.5, 0.01])
    # cax.set_xlabel("Reflected Mass", fontsize=12)

    # plt.rcParams["text.usetex"] = False
    # plt.rcParams["axes.titley"] = 0.5
    # plt.rcParams["axes.titlepad"] = 0.1
    # plt.tight_layout()

    image_name = f"Reflected-Inertia-optimization-C{start}-C{end}"
    plt.savefig(f"{image_name}.pdf", format="pdf")
    plt.show()


# Directory containing .npz files
# directory = (
#     "logs/ur5_coppeliasim_full_path/plans/release_H48_T25_LimitsNormalizer_b64_condFalse/0/"  # Current directory
# )
directory = (
    "logs/ur5_coppeliasim_full_path/plans/release_H48_T25_LimitsNormalizer_b64_condFalse/0/"  # Current directory
)
exp_name = "C10_C14_diffuser"
directory = f"/home/ws/src/diffuser/Experiments/{exp_name}"  # /new_ones"  # Current directory
exp_name = "exp10_tomm_Diffusion-Model_x"
directory = f"/home/ws/src/diffuser/logs/tomm_coppeliasim_full_path/plans/release_H48_T25_LimitsNormalizer_b64_condFalse/0/{exp_name}"
# directory = "/home/ws/src/diffuser/Experiments/C11_C8"  # Current directory
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
# Compute metric on arrays in .npz files in the directory
visualize_trajectories(directory)
