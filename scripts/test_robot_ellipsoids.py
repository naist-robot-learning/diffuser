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

def compute_traj_eigendecomposition(theta):
    
    q_ = torch.tensor(theta).unsqueeze(dim=0)
    q_ = einops.rearrange(q_, 'B H T -> B T H')
    
    M_x_inv = compute_kinetic_energy_matrix(q_)
    L, V_ = torch.linalg.eig(M_x_inv)
    L = 1/L.real.cpu()
    L = torch.sqrt(L)*0.1
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
def plot_ellipsoid(x_init, Lambda, Q):
    
    # Define ellipsoid parameters
    center = (x_init[0], x_init[1], x_init[2])  # Center of the ellipsoid
    radii = (Lambda[0], Lambda[1], Lambda[2])    # Length of the semi-axes along each axis
    
    rotation = Q.T # Rotation matrix (identity matrix for no rotation)
    
    # Generate surface points of the ellipsoid
    surface_points = ellipsoid_surface_points(center, radii, rotation)
    
    # Plot the ellipsoid mesh
    ellipsoid = ax.plot_wireframe(surface_points[:, :, 0], surface_points[:, :, 1], surface_points[:, :, 2], color='b', alpha=0.3)
    return ellipsoid 
# Plotting function
def update_plot(frame, line, line2, scatter, quiver):
    
    # Update joint angles
    new_positions, T = forward_kinematics(theta[frame])
    # Update plot
    line.set_data_3d(
        [p[0] for p in new_positions[:-1]], 
        [p[1] for p in new_positions[:-1]], 
        [p[2] for p in new_positions[:-1]]
        )
    line2.set_data_3d(
        [p[0] for p in new_positions[-2:]], 
        [p[1] for p in new_positions[-2:]], 
        [p[2] for p in new_positions[-2:]]
    )
    scatter._offsets3d = (
        np.array([p[0] for p in new_positions[:-2]]), 
        np.array([p[1] for p in new_positions[:-2]]), 
        np.array([p[2] for p in new_positions[:-2]])
        )
    
    # Update frame at the tip of the robot
    tip_position = new_positions[-1]  # Get the position of the end effector (tip)
    frame_axes = T[:3, :3]  # Extract rotation matrix from the transformation matrix
    # Define scaling factor
    scale_factor = 0.05  # Adjust this value to shorten or lengthen the axes
    # Plot ellipsoid
    # Remove wireframe plot
    # import ipdb 
    # ipdb.set_trace()
    
    global ellipsoid
    if ellipsoid:
        ellipsoid.remove()
    #ellipsoid.set_visible(False)
    # Clear previous wireframe plot
    #ellipsoid.set_data_3d([], [], [])
    
    ellipsoid = plot_ellipsoid(tip_position, L[frame], V[frame])
    # Scale the frame axes
    frame_axes = frame_axes * scale_factor
    quiver.set_segments([np.array([tip_position, tip_position + frame_axes[:, i]]) for i in range(3)])
    quiver._offsets3d = (tip_position[0], tip_position[1], tip_position[2])
    quiver.set_color(['r', 'g', 'b'])
    return line, line2, scatter, quiver


# Directory containing .npz files
directory = "logs/ur5_coppeliasim_full_path/plans/release_H48_T20_LimitsNormalizer_b64_condFalse/0/"  # Current directory


# Compute metric on arrays in .npz files in the directory
exp_number = 0
q, goal_pose, hand= extract_trajectory(directory, exp_number)

# Define joint angles (in radians)
#theta = [0, np.pi/4, np.pi/2, np.pi/4, np.pi/3, np.pi/6]
traj = 3
theta = np.reshape(q[traj], (48,6))

#theta[0][-3] = theta[0][-3] + np.pi/4 
#theta = np.zeros((48,6))


# Compute trajectory in Cartesian space
x = compute_trajectory_in_cartesian(theta)

# initial TCP position
x_init = x[0]
# Plotting
fig = plt.figure(facecolor='none', figsize=(14,10))
ax = fig.add_subplot(111, projection='3d')
# Plot trajectory
for pos in x:
    ax.scatter(pos[0], pos[1], pos[2], color='r', s=5)

goal = goal_pose[0][0,:3]

ax.scatter(goal[0], goal[1], goal[2], marker='^', s=50, color='green')
hand = hand[0].squeeze()
ax.scatter(hand[0], hand[1], hand[2], marker='*', s=50, color='red')
# Define the table top dimensions
table_width = 1.0
table_length = 1.5
table_height = 0.0  # Height above the origin

# Plot table top as a rectangle
table_top = ax.plot(
    [-table_width / 2, table_width / 2], 
    [-table_length / 2, -table_length / 2], 
    [table_height, table_height], 
    color='goldenrod', 
    linewidth=2
)
ax.plot(
    [table_width / 2, table_width / 2], 
    [-table_length / 2, table_length / 2], 
    [table_height, table_height], 
    color='goldenrod', 
    linewidth=2
)
ax.plot(
    [table_width / 2, -table_width / 2], 
    [table_length / 2, table_length / 2], 
    [table_height, table_height], 
    color='goldenrod', 
    linewidth=2
)
ax.plot(
    [-table_width / 2, -table_width / 2], 
    [table_length / 2, -table_length / 2], 
    [table_height, table_height], 
    color='goldenrod', 
    linewidth=2
)

L, V = compute_traj_eigendecomposition(theta)
# Plot Ellipsoid

ellipsoid = plot_ellipsoid(x_init, L[0], V[0])

# Initialize plot
positions, T = forward_kinematics(theta[0])

line, = ax.plot([p[0] for p in positions[:-1]], [p[1] for p in positions[:-1]], [p[2] for p in positions[:-1]], color='lightgray', linewidth=20, alpha=1)
line2, = ax.plot([p[0] for p in positions[-2:]], [p[1] for p in positions[-2:]], [p[2] for p in positions[-2:]], color='dimgray', linewidth=12, alpha=1)
scatter = ax.scatter([p[0] for p in positions[:-2]], [p[1] for p in positions[:-2]], [p[2] for p in positions[:-2]], color='skyblue', s=500, alpha=1)

# Draw initial frame at the tip of the robot
tip_position = positions[-1]  # Get the position of the end effector (tip)
frame_axes = T[:3, :3]  # Extract rotation matrix from the transformation matrix
quiver = ax.quiver(tip_position[0], tip_position[1], tip_position[2],
                   [frame_axes[0, 0], frame_axes[1, 0], frame_axes[2, 0]],
                   [frame_axes[0, 1], frame_axes[1, 1], frame_axes[2, 1]],
                   [frame_axes[0, 2], frame_axes[1, 2], frame_axes[2, 2]], color=['r', 'g', 'b'], length=0.1)



# Setting labels and aspect ratio
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1,1,1])
ticks = np.linspace(-0.5, 0.5, num=5)
ticksz = np.linspace(-0.5, 1.0, num=7)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticksz)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 1.0)

# Set camera perspective
ax.view_init(elev=10, azim=135)  # Adjust elevation and azimuth as desired
#ax.dist = 5
# ax.grid(False)
# Create animation
ani = FuncAnimation(fig, update_plot, frames=48, fargs=(line, line2, scatter, quiver), interval=48, blit=False)

ani.save('robot_animation.gif', writer='pillow', fps=30)

# Show plot
plt.axis('off')
plt.show()