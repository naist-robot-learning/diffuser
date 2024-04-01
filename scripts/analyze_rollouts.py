import os
import numpy as np
from diffuser.robot.UR5kinematicsAndDynamics_vectorized import compute_reflected_mass
import torch

def compute_metric_on_arrays(directory):
    # Get list of all .npz files in the directory
    npz_files = [file for file in os.listdir(directory) if file.endswith('.npz')]

    # Loop through each .npz file
    for npz_file in npz_files:
        # Load arrays from the .npz file
        npz_data = np.load(os.path.join(directory, npz_file))
        q = npz_data['joint_position']
        q = torch.tensor(q).unsqueeze(-1)
        dim = q.shape[1]
        horizon = q.shape[0]
        hand_pose = npz_data['robot_hand_pose']
        u = torch.empty((horizon, 3, 1), dtype=torch.float32).to('cuda')
        u[:,0] = 1; u[:,1] = 0; u[:,2] = 0
        cost = 0
        cost = compute_reflected_mass(q, u).sum(axis=1)
        
        # Loop through each array in the .npz file
        print(f"\nMetrics for arrays in {npz_file}:")
        mean = torch.mean(cost)
        std_dev = torch.std(cost)
        print(f"Joint_position: Mean = {mean}, Standard Deviation = {std_dev}")
        

        # Close the .npz file
        npz_data.close()

# Directory containing .npz files
directory = "logs/ur5_coppeliasim_full_path_plus_hand_v1/plans/release_H32_T128_LimitsNormalizer_b64_condFalse/0/"  # Current directory


# Compute metric on arrays in .npz files in the directory
compute_metric_on_arrays(directory)