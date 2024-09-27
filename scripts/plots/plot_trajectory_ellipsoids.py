import numpy as np
from rm_analysis import (
    extract_trajectory,
    forward_kinematics,
    compute_trajectory_in_cartesian,
)
from diffuser.robot.UR5kinematicsAndDynamics_vectorized import compute_kinetic_energy_matrix
import einops
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import torch
from diffuser.robot.UR5kinematicsAndDynamics_vectorized import compute_reflected_mass, fkine
import torch


def compute_traj_eigendecomposition(theta):

    q_ = torch.tensor(theta, dtype=torch.float32).unsqueeze(dim=0)
    q_ = einops.rearrange(q_, "B H T -> B T H")

    M_x_inv = compute_kinetic_energy_matrix(q_)
    L, V_ = torch.linalg.eig(M_x_inv)
    L = 1 / L.real.cpu()
    L = torch.sqrt(L) * 0.1
    V = V_.real.cpu().numpy()
    return L, V


def ellipsoid_surface_points(center, radii, rotation, num_points=20):
    # Generate spherical coordinates
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)

    # Convert spherical coordinates to Cartesian coordinates
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # Apply rotation matrix
    points = np.stack((x, y, z), axis=-1)
    rotated_points = np.dot(points, rotation) + center
    return rotated_points


def plot_ellipsoid(ax, x_init, Lambda, Q):

    # Define ellipsoid parameters
    center = (x_init[0], x_init[1], x_init[2])  # Center of the ellipsoid
    radii = (Lambda[0], Lambda[1], Lambda[2])  # Length of the semi-axes along each axis

    rotation = Q.T  # Rotation matrix (identity matrix for no rotation)

    # Generate surface points of the ellipsoid
    surface_points = ellipsoid_surface_points(center, radii, rotation)
    X, Y, Z = surface_points[:, :, 0], surface_points[:, :, 1], surface_points[:, :, 2]
    # Plot the ellipsoid mesh
    ellipsoid = ax.plot_wireframe(
        surface_points[:, :, 0], surface_points[:, :, 1], surface_points[:, :, 2], color="b", alpha=0.3
    )
    return ellipsoid


# Plotting function
def update_plot(frame, quiver, ax):

    # Update joint angles
    new_positions = positions[frame]
    new_T = T[frame].cpu().numpy()

    # Update frame at the tip of the robot
    tip_position = new_positions[:3]  # Get the position of the end effector (tip)
    frame_axes = new_T[:3, :3]  # Extract rotation matrix from the transformation matrix
    # Define scaling factor
    scale_factor = 0.05  # Adjust this value to shorten or lengthen the axes

    plot_ellipsoid(ax, tip_position, L[frame], V[frame])
    # Scale the frame axes
    frame_axes = frame_axes * scale_factor
    quiver.set_segments([np.array([tip_position, tip_position + frame_axes[:, i]]) for i in range(3)])
    quiver._offsets3d = (tip_position[0], tip_position[1], tip_position[2])
    quiver.set_color(["r", "g", "b"])
    return quiver


# Directory containing .npz files
directory = "/home/ws/src/diffuser/Experiments/C10_C14_diffuser/plots"  # Current directory

# Directory containing .npz files
directory = "/home/ws/src/diffuser/Experiments/RealRobotTomm/Exp_middle/output_reordered/plots"  # Current directory
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# Compute metric on arrays in .npz files in the directory
exp_number = 0
q, npz_file_name = extract_trajectory(directory, exp_number)

q = q[300:1300, :]  # Slice to keep the range 300:1300
# Select 40 evenly spaced indices between 0 and the length of the sliced range
num_points = 40
indices = np.linspace(0, q.shape[0] - 1, num_points).astype(int)
q = q[indices, :]

q[:, 0] = q[:, 0] - np.pi / 2
# Define joint angles (in radians)
traj = 3
theta = q
if len(q.shape) > 3:
    q_torch = torch.tensor(q, dtype=torch.float32).squeeze(dim=1)
else:
    q_torch = torch.tensor(q, dtype=torch.float32)

# Compute velocities

q_torch = einops.rearrange(q_torch.unsqueeze(0), "b h t -> b t h")
q_torch = einops.rearrange(q_torch, "b t h -> t (b h)")


# Compute trajectory in Cartesian space
# x = compute_trajectory_in_cartesian(theta)
# Initialize plot
positions, T = fkine(q_torch.to("cuda"))
positions = positions.cpu().numpy()
##############################################################
# initial TCP position
x_init = positions[0]
# Plotting
fig = plt.figure(facecolor="none", figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")
# Plot trajectory
for pos in positions:
    ax.scatter(pos[0], pos[1], pos[2], color="r", s=5)

# goal = goal_pose[0][0, :3]
goal = positions[-1]
# goal = np.array([0.55, 0, 0.1])
ax.scatter(goal[0], goal[1], goal[2], marker="^", s=50, color="green")
# hand = hand[0].squeeze()
# ax.scatter(hand[0], hand[1], hand[2], marker="*", s=50, color="red")
# Define the table top dimensions
table_width = 0.8
table_length = 1.0
table_height = 1.0  # Height above the origin
x_offset = 1.0
y_offset = 1.0
# # Plot table top as a rectangle
# table_top = ax.plot(
#     [-table_width / 2 + x_offset, table_width / 2 + 0.5 + x_offset],
#     [-table_length / 2 + y_offset, -table_length / 2 + 0.5 + y_offset],
#     [table_height, table_height],
#     color="green",
#     linewidth=2,
# )
# ax.plot(
#     [table_width / 2  + x_offset, table_width / 2 + 0.5 + x_offset],
#     [-table_length / 2  + y_offset, table_length / 2 + 1.5 + y_offset],
#     [table_height, table_height],
#     color="green",
#     linewidth=2,
# )
# ax.plot(
#     [table_width / 2 + 0.5 + x_offset, -table_width / 2 + 0.5 + x_offset],
#     [table_length / 2 + 1.5 + y_offset, table_length / 2 + 1.5 + y_offset],
#     [table_height, table_height],
#     color="green",
#     linewidth=2,
# )
# ax.plot(
#     [-table_width / 2 + 0.5 + x_offset, -table_width / 2 + 0.5 + x_offset],
#     [table_length / 2 + 1.5 + y_offset, -table_length / 2 + 1.5 + y_offset],
#     [table_height, table_height],
#     color="green",
#     linewidth=2,
# )

L, V = compute_traj_eigendecomposition(theta)
# Plot Ellipsoid

ellipsoid = plot_ellipsoid(ax, x_init, L[0], V[0])


# positions, T = forward_kinematics(theta[0])


# Draw initial frame at the tip of the robot
tip_position = positions[-1]  # Get the position of the end effector (tip)
frame_axes = T[:3, :3].cpu().numpy()  # Extract rotation matrix from the transformation matrix
quiver = ax.quiver(
    tip_position[0],
    tip_position[1],
    tip_position[2],
    [frame_axes[0, 0], frame_axes[1, 0], frame_axes[2, 0]],
    [frame_axes[0, 1], frame_axes[1, 1], frame_axes[2, 1]],
    [frame_axes[0, 2], frame_axes[1, 2], frame_axes[2, 2]],
    color=["r", "g", "b"],
    length=0.1,
)


# Setting labels and aspect ratio
ax.set_xlabel("X (m)", fontsize=14)
ax.set_ylabel("Y (m)", fontsize=14)
ax.set_zlabel("Z (m)", fontsize=14)
ax.set_box_aspect([1, 1, 1])
ticks = np.linspace(-0.5, 0.5, num=5)
ticksz = np.linspace(-0.5, 1.0, num=7)
ax.tick_params(axis="both", which="major", labelsize=12)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_zticks(ticksz)
# ax.set_xlim(-0.5, 0.5)
# ax.set_ylim(-0.5, 0.5)
# ax.set_zlim(-0.5, 1.0)

# Set camera perspective
# ax.view_init(elev=10, azim=35)  # Adjust elevation and azimuth as desired
# ax.dist = 5
# ax.grid(False)
# Create animation
# plt.axis("off")
numbers_to_plot = [0, 11, 15, 20, 39]
for i in range(len(theta)):
    if i in numbers_to_plot:
        update_plot(i, quiver, ax)

# ani = FuncAnimation(fig, update_plot, frames=48, fargs=(line, line2, scatter, quiver), interval=48, blit=False)

# ani.save(f"{npz_file_name}_animation_{traj}_presentation_diffuser.gif", writer="pillow", fps=30)

# Show plot
plt.show()
