import os
import numpy as np
import einops

from diffuser.robot.KUKALWR4KinematicsAndDynamics_vectorized import compute_reflected_mass, fkine

# from diffuser.robot.UR5kinematicsAndDynamics_vectorized import fkine
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class Renderer:
    """Class to render trajectories in cartesian space acepting config-space inputs q.
    TODO: pass the forward and reflected mass functions maybe as functors
    """

    def __init__(self, sample_directory=None):
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._directory = sample_directory
        self.q = None

    def load_trajectory(self, q):
        if isinstance(q, torch.Tensor):
            q = q.cpu().numpy()
        elif isinstance(q, np.ndarray):
            pass
        else:
            raise TypeError("Input should be a NumPy array or a PyTorch tensor.")
        self.q = q

    def _plot_with_axes(ax, x, y, z, roll, pitch, yaw):
        for i in range(len(x)):
            # Define rotation matrix from roll, pitch, and yaw at each point
            rotation_matrix = R.from_euler("xyz", [roll[i], pitch[i], yaw[i]], degrees=True).as_matrix()

            # Define unit vectors along X, Y, and Z axes
            x_axis = np.array([1, 0, 0])
            y_axis = np.array([0, 1, 0])
            z_axis = np.array([0, 0, 1])

            # Rotate unit vectors using rotation matrix
            rotated_x_axis = np.dot(rotation_matrix, x_axis)
            rotated_y_axis = np.dot(rotation_matrix, y_axis)
            rotated_z_axis = np.dot(rotation_matrix, z_axis)

            # Plot rotated coordinate system vectors at each point
            if i == 0 or i == len(x) - 1:
                ax.quiver(
                    x[i],
                    y[i],
                    z[i],
                    rotated_x_axis[0],
                    rotated_x_axis[1],
                    rotated_x_axis[2],
                    color="r",
                    length=0.01,
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
                    length=0.01,
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
                    length=0.01,
                    normalize=True,
                )
            else:
                ax.plot(x[i], y[i], z[i])

    def _plot_trajectories(self, q, color=None):

        for i in range(0, len(q)):
            if i % 1 == 0:  # Adjust i to add or reduce the number of plotted trajs
                print("i: ", i)
                qt = torch.tensor(q[i])
                qt = einops.rearrange(qt[0], "h t -> t h").to("cuda")
                pose = fkine(qt).cpu().numpy()
                x, y, z = pose[:, 0], pose[:, 1], pose[:, 2]
                roll, pitch, yaw = pose[:, 3], pose[:, 4], pose[:, 5]
                self._ax.scatter(x[0], y[0], z[0], marker="^", color="b", s=1000)
                self._ax.scatter(x[1:], y[1:], z[1:])

                # plot_with_axes(ax, x, y, z, roll, pitch, yaw)
        if color:
            self._ax.scatter(x[-1], y[-1], z[-1], marker="*", color=color, s=1500)
        else:
            self._ax.scatter(x[-1], y[-1], z[-1], marker="*", color="r", s=1500)
        self._ax.set_xlim(0.1, 0.5)
        self._ax.set_ylim(0.0, 0.6)
        self._ax.set_zlim(0.0, 0.5)

        # Set axis ticks
        self._ax.set_xticks(np.arange(0.1, 0.5, 0.05))
        self._ax.set_yticks(np.arange(0.0, 0.6, 0.05))
        self._ax.set_zticks(np.arange(0.0, 0.5, 0.05))

        # Increase the size of axis labels and ticks
        self._ax.tick_params(axis="both", which="major", labelsize="14")

        self._ax.set_xlabel("X", fontsize="20", labelpad=15)
        self._ax.set_ylabel("Y", fontsize="20", labelpad=15)
        self._ax.set_zlabel("Z", fontsize="20", labelpad=15)

        plt.show()

    def unpack_sample_trajectory(self):
        # Get list of all .npz files in the directory
        npz_files = [file for file in os.listdir(directory) if file.endswith(".npz")]

        # Loop through each .npz file
        for npz_file in npz_files:
            # Load arrays from the .npz file
            print("File name: ", npz_file)
            npz_data = np.load(os.path.join(directory, npz_file))
            q = npz_data["joint_position"]  # Trials x horizon x transition 100 x 48 x 6
        plt.title(f"{npz_file}")
        return q

    def visualize_trajectories(self, q=None):
        if self.q:
            self._plot_trajectories(self.q)
            return
        if q:
            self._plot_trajectories(q)
        else:
            if directory:
                q = self.unpack_sample_trajectory()
                self._plot_trajectories(q)
            else:
                raise ValueError("Neither a q nor a sample directory was given")


if __name__ == "__main__":
    # Directory containing .npz files
    directory = (
        "logs/kuka_coppeliasim_full_path/plans/release_H48_T20_LimitsNormalizer_b64_condFalse/0/"  # Current directory
    )
    # Compute metric on arrays in .npz files in the directory
    renderer = Renderer(sample_directory=directory)
    renderer.visualize_trajectories()
