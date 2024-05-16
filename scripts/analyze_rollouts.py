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
        print("File name: ", npz_file)
        npz_data = np.load(os.path.join(directory, npz_file))
        q = npz_data["joint_position"]  #Trials x horizon x transition 100 x 48 x 6
        values = torch.tensor(npz_data["values"])     #Trials x horizon 100 x 48
        # Loop through each array in the .npz file
        print(f"\nMetrics for arrays in {npz_file}:")
        mean = torch.mean(values)
        std_dev = torch.std(values)
        print(f"Reflected mass cost: Mean = {mean}, Standard Deviation = {std_dev}")

        # Close the .npz file
        npz_data.close()


# Directory containing .npz files
directory = "logs/ur5_coppeliasim_full_path/plans/release_H48_T20_LimitsNormalizer_b64_condFalse/0/"  # Current directory


# Compute metric on arrays in .npz files in the directory
compute_metric_on_arrays(directory)
