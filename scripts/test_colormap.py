import numpy as np
import matplotlib.pyplot as plt

# Define the table top dimensions
table_width = 1.0
table_length = 1.5
table_height = 0.0  # Height above the origin

# Define the robot arm dimensions
arm_length = 0.5

# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot table top as a rectangle
table_top = plt.Rectangle((-table_width / 2, -table_length / 2), table_width, table_length, color='gray')
ax.add_patch(table_top)
art3d.pathpatch_2d_to_3d(table_top, z=table_height, zdir="z")

# Plot robot arm as a line
robot_arm = np.array([[0, 0, table_height], [arm_length, 0, table_height]])
ax.plot(robot_arm[:, 0], robot_arm[:, 1], robot_arm[:, 2], color='blue')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Table Top and Robot Arm')

# Set limits and aspect ratio
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-0.5, 1.5])
ax.set_box_aspect([1, 1, 1])

plt.show()