import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_batch(batch_size=100, num_points=1000):
    # Generate a batch of trajectories
    batch = []
    for _ in range(batch_size):
        t = np.linspace(0, 10, num_points)  # Time points
        x = np.sin(t) + np.random.normal(0, 0.1, size=num_points)  # X-coordinate with noise
        y = np.cos(t) + np.random.normal(0, 0.1, size=num_points)  # Y-coordinate with noise
        z = t + np.random.normal(0, 0.1, size=num_points)  # Z-coordinate with noise
        batch.append((x, y, z))
    return batch

batches = [generate_batch() for _ in range(6)]  # Generate 6 batches of trajectories

for i, batch in enumerate(batches):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j, trajectory in enumerate(batch):
        x, y, z = trajectory
        ax.plot(x, y, z, label=f'Trajectory {j+1}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title(f'Batch {i+1}')
plt.show()