import numpy as np

import einops
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import torch


class AnimatorBase:
    def __init__(self):
        self._x = None
        pass

    def get_x_traj(self):
        return self._x


class DiffusionAnimator(AnimatorBase):
    def __init__(self, sample_directory=None, robot_type="UR5"):
        super().__init__()
        if robot_type == "kuka":
            from rm_analysisKUKA import (
                extract_trajectory,
                forward_kinematics,
                compute_trajectory_in_cartesian,
            )
            from diffuser.robot.KUKALWR4KinematicsAndDynamics_vectorized import compute_kinetic_energy_matrix

            self.compute_kinetic_energy_matrix = compute_kinetic_energy_matrix
            self.forward_kinematics = forward_kinematics
        elif robot_type == "UR5":
            from rm_analysisUR5 import (
                extract_trajectory,
                forward_kinematics,
                compute_trajectory_in_cartesian,
            )
            from diffuser.robot.UR5kinematicsAndDynamics_vectorized import compute_kinetic_energy_matrix, fkine

            self.compute_kinetic_energy_matrix = compute_kinetic_energy_matrix
            self.forward_kinematics = forward_kinematics
        else:
            from rm_analysisKUKA import (
                extract_trajectory,
                forward_kinematics,
                compute_trajectory_in_cartesian,
            )
            from diffuser.robot.KUKALWR4KinematicsAndDynamics_vectorized import compute_kinetic_energy_matrix

            self.compute_kinetic_energy_matrix = compute_kinetic_energy_matrix
            self.forward_kinematics = forward_kinematics
        self.compute_trajectory_in_cartesian = fkine
        print(f"importing {robot_type} rm_analysis")
        self._directory = sample_directory
        experiment_file_number = 0
        self._ellipsoid = None
        self._L = None
        self._V = None
        self._fig = None
        if sample_directory:

            q, goal, hand = extract_trajectory(directory=self._directory, exp_number=experiment_file_number)
            self._q = q
            self._goal = goal
            self._hand = hand
        else:
            self._q_init = np.array([1.38361077, -0.96197214, -2.49578144, -1.25463394, 1.57076255, -0.18717196])
            self._goal = np.array([0.55, 0.2, 0.1, 0.7071, 0.7071, 0.0, 0.0])
            self._hand = np.array([0.55, 0.2, 0.1, 0.7071, 0.7071, 0.0, 0.0])

    def load_trajectory(self, theta=None):
        """Function to load the trajectory to the class

        Args:
            theta (numpy.array): Trajectory in C-Space. Shape: (batch, horizon, n_dof)
                                 typically (64, 48, 7)

        Raises:
            ValueError: Returns an error if no trajectory was given and there is
                        no sample directory to use.
        """

        x = self.compute_trajectory_in_cartesian(einops.rearrange(theta, "B H T -> T (B H)", H=48))
        self._x = einops.rearrange(x, "(B H) T -> B H T", H=48).cpu().numpy()

    def _initialize_plot(self):

        if not self._fig:
            self._fig = plt.figure(facecolor="none", figsize=(14, 10))
            self._ax = self._fig.add_subplot(111, projection="3d")

        goal = self._goal[:3]

        self._ax.scatter(goal[0], goal[1], goal[2], marker="^", s=50, color="green")
        hand = self._hand
        self._ax.scatter(hand[0], hand[1], hand[2], marker="*", s=50, color="red")
        # Define the table top dimensions
        table_width = 1.0  # 1.0
        table_length = 0.75  # 1.5
        table_height = 0.0  # Height above the origin

        # Plot table top as a rectangle
        table_top = self._ax.plot(
            [-table_width / 2, table_width / 2],
            [-table_length / 2, -table_length / 2],
            [table_height, table_height],
            color="goldenrod",
            linewidth=2,
        )
        self._ax.plot(
            [table_width / 2, table_width / 2],
            [-table_length / 2, table_length / 2],
            [table_height, table_height],
            color="goldenrod",
            linewidth=2,
        )
        self._ax.plot(
            [table_width / 2, -table_width / 2],
            [table_length / 2, table_length / 2],
            [table_height, table_height],
            color="goldenrod",
            linewidth=2,
        )
        self._ax.plot(
            [-table_width / 2, -table_width / 2],
            [table_length / 2, -table_length / 2],
            [table_height, table_height],
            color="goldenrod",
            linewidth=2,
        )

        # Initialize plot
        positions, T = self.forward_kinematics(self._q_init)
        (line,) = self._ax.plot(
            [p[0] for p in positions[:-1]],
            [p[1] for p in positions[:-1]],
            [p[2] for p in positions[:-1]],
            color="lightblue",
            linewidth=22,
            alpha=1,
        )
        (line2,) = self._ax.plot(
            [p[0] for p in positions[-2:]],
            [p[1] for p in positions[-2:]],
            [p[2] for p in positions[-2:]],
            color="dimgray",
            linewidth=12,
            alpha=1,
        )
        scatter = self._ax.scatter(
            [p[0] for p in positions[:-1]],
            [p[1] for p in positions[:-1]],
            [p[2] for p in positions[:-1]],
            color="lightgray",
            s=500,
            alpha=1,
        )

        # Draw initial frame at the tip of the robot
        tip_position = positions[-1]  # Get the position of the end effector (tip)
        frame_axes = T[:3, :3]  # Extract rotation matrix from the transformation matrix
        quiver = self._ax.quiver(
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
        self._ax.grid(False)
        self._ax.set_axis_off()
        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        self._ax.set_zlabel("Z")
        self._ax.set_box_aspect([1, 1, 1])
        ticks = np.linspace(-0.5, 0.5, num=5)
        ticksz = np.linspace(-0.5, 1.0, num=7)
        self._ax.set_xticks(ticks)
        self._ax.set_yticks(ticks)
        self._ax.set_zticks(ticksz)
        self._ax.set_xlim(-0.5, 0.5)
        self._ax.set_ylim(-0.5, 0.5)
        self._ax.set_zlim(-0.5, 1.0)

        # Set camera perspective
        self._ax.view_init(elev=10, azim=25)  # Adjust elevation and azimuth as desired
        # ax.dist = 5
        # ax.grid(False)
        return line, line2, scatter, quiver

    # Plotting function
    def _update_plot(
        self,
        frame,
        color=None,
        traj=None,
    ):

        # Update joint angles
        if traj:
            pos = self._x[traj, frame, :3]
        else:
            pos = self._x[:, frame, :3]
        # Plot entire trajectory points
        for pos in self._x:
            if color is None:
                self._ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color="r", s=5)
            else:
                self._ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color=color, s=5)
        # Update plot

        return

    def render_robot_animation(
        self, num, color: bool = False, save: bool = False, name: str = None, animate=False, plot=True
    ):

        # Create animation
        if animate:

            ani = FuncAnimation(
                self._fig, self._update_plot, frames=48, fargs=(line, line2, scatter, quiver), interval=48, blit=False
            )

            if save:
                if name:
                    ani.save(name + ".gif", writer="pillow", fps=30)
                else:
                    ani.save("robot_animation4.gif", writer="pillow", fps=30)

            # Show plot
            plt.axis("off")
            plt.show()
        if plot:

            path_length = len(self._x[0])
            colors = ["red", "white"]
            cmap = LinearSegmentedColormap.from_list("red_white_cmap", colors, N=path_length)
            colors_map = ["green", "white"]
            cmap_map = LinearSegmentedColormap.from_list("green_white_cmap", colors_map, N=path_length)
            # Generate the colors based on the custom colormap
            colors = cmap(np.linspace(0, 1, path_length))
            colors_map = cmap_map(np.linspace(0, 1, path_length))
            line, line2, scatter, quiver = self._initialize_plot()
            if color:
                for i in range(path_length):
                    traj = 0
                    self._update_plot(i, colors_map, traj)

            for i in range(path_length):
                if color:
                    self._update_plot(i, colors)
                else:
                    self._update_plot(i)

            plt.savefig(f"diffusion_step_single{num}.pdf", format="pdf", dpi=500)
            plt.savefig(f"diffusion_step_single{num}.png", format="png", dpi=500)
            self._ax.clear()


if __name__ == "__main__":

    "****** Example usage **********"
    # Directory containing .npz files
    directory = (
        "logs/kuka_coppeliasim_full_path/plans/release_H48_T20_LimitsNormalizer_b64_condFalse/0/"  # Current directory
    )

    animator = DiffusionAnimator(sample_directory=directory)
    animator.load_trajectory(theta=None)
    animator.render_robot_animation(save=False)
