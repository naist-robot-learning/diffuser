import os
import numpy as np
import einops
from diffuser.robot.UR5kinematicsAndDynamics_vectorized import compute_reflected_mass, fkine
import torch


def compute_metric_on_arrays(directory):
    # Get list of all .npz files in the directory
    npz_files = [file for file in os.listdir(directory) if file.endswith(".npz")]

    # Loop through each .npz file
    for npz_file in npz_files:
        # Load arrays from the .npz file
        npz_data = np.load(os.path.join(directory, npz_file))
        q = npz_data["joint_position"]
        q = torch.tensor(q).unsqueeze(-1)
        dim = q.shape[1]
        horizon = q.shape[0]
        hand_pose = torch.tensor(npz_data["robot_hand_pose"])
        x_ = einops.rearrange(q, "b t h -> t (b h)").to("cuda")
        x_tcp = fkine(x_[:6, :])[:, :3].unsqueeze(2)
        x_hand = hand_pose[:, :3].unsqueeze(2).to("cuda")
        u = (x_hand - x_tcp) / torch.linalg.norm((x_hand - x_tcp), dim=1, ord=2).unsqueeze(2)
        cost = compute_reflected_mass(q[:, :6, :], u)
        # u = torch.empty((horizon, 3, 1), dtype=torch.float32).to("cuda")
        # u[:, 0] = 1
        # u[:, 1] = 0
        # u[:, 2] = 0
        # Loop through each array in the .npz file
        print(f"\nMetrics for arrays in {npz_file}:")
        mean = torch.mean(cost)
        std_dev = torch.std(cost)
        print(f"Reflected mass cost: Mean = {mean}, Standard Deviation = {std_dev}")

        # Close the .npz file
        npz_data.close()


# Directory containing .npz files
directory = "logs/ur5_coppeliasim_full_path_goal/plans/release_H48_T22_LimitsNormalizer_b128_condFalse/0/"  # Current directory


# Compute metric on arrays in .npz files in the directory
compute_metric_on_arrays(directory)
