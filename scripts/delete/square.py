import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product


def sample_points_near_cube_corners(n_points, cube_size, radius):
    points = []
    # Define the 8 corners of the cube
    corners = [
        (0, 0, 0),
        (cube_size, 0, 0),
        (cube_size, cube_size, 0),
        (0, cube_size, 0),
        (0, 0, cube_size),
        (cube_size, 0, cube_size),
        (cube_size, cube_size, cube_size),
        (0, cube_size, cube_size),
    ]

    for _ in range(n_points):
        # Randomly choose a corner
        corner = corners[np.random.choice(len(corners))]

        # Randomly sample a point within a sphere around the chosen corner
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1, 1)
        u = np.random.uniform(0, 1)

        theta = np.arccos(costheta)
        r = radius * (u ** (1 / 3))

        x = corner[0] + r * np.sin(theta) * np.cos(phi)
        y = corner[1] + r * np.sin(theta) * np.sin(phi)
        z = corner[2] + r * np.cos(theta)

        points.append((x, y, z))

    return points


# Parameters
cube_size = 10
radius = 1
n_points = 100

# Sample points
points = sample_points_near_cube_corners(n_points, cube_size, radius)

# Plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# Draw cube edges
r = [0, cube_size]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s - e)) == r[1] - r[0]:
        ax.plot3D(*zip(s, e), color="k")

# Scatter points
x_points, y_points, z_points = zip(*points)
ax.scatter(x_points, y_points, z_points, color="red", s=10)

ax.set_xlim([-radius, cube_size + radius])
ax.set_ylim([-radius, cube_size + radius])
ax.set_zlim([-radius, cube_size + radius])
ax.set_aspect("auto")
plt.show()
