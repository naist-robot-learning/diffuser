import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_trajectory(num_points=1000):
    # Generate random trajectory points
    t = np.linspace(0, 10, num_points)  # Time points
    x = np.sin(t) + np.random.normal(0, 0.1, size=num_points)  # X-coordinate with noise
    y = np.cos(t) + np.random.normal(0, 0.1, size=num_points)  # Y-coordinate with noise
    z = t + np.random.normal(0, 0.1, size=num_points)  # Z-coordinate with noise
    return x, y, z

# Plot the trajectories in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(100):
    x, y, z = generate_trajectory()
    ax.plot(x, y, z, label=f'Trajectory {i+1}')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('100 Trajectories in 3D')
plt.show()