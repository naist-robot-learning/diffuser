import os
import numpy as np
import einops
from diffuser.robot.UR5kinematicsAndDynamics_vectorized import compute_reflected_mass, fkine
from diffuser.robot.KUKALWR4KinematicsAndDynamics_vectorized import fkine as fkineKuka
import torch


def compute_metric_on_arrays(directory):
    # Get list of all .npz files in the directory
    npz_files = [file for file in os.listdir(directory) if file.endswith(".npz")]

    full_data = []
    full_data_time = []
    full_data_path_length = []
    full_joint_traj = []
    # Loop through each .npz file
    for i, npz_file in enumerate(npz_files):
        # Load arrays from the .npz file
        print("File name: ", npz_file)
        npz_data = np.load(os.path.join(directory, npz_file))
        # q = npz_data["joint_position"]  # Trials x horizon x transition 100 x 48 x 6
        # values = torch.tensor(npz_data["values"])  # Trials x horizon 100 x 48

        print("Batch size of trajectories ", len(npz_data["rm_masses"]))

        qs = torch.tensor(npz_data["q"]).to("cuda")
        qs = einops.rearrange(qs, "B H T ->  T (B H) ")

        traj_points = fkine(qs)
        traj_points = einops.rearrange(traj_points, "(B H) T -> B H T", H=48).cpu().numpy()
        path_length = []

        for traj in traj_points:
            d = 0
            for i in range(len(traj) - 1):
                d += np.linalg.norm(traj[i + 1, :3] - traj[i, :3])
            path_length.append(d)
        rm_mass = npz_data["rm_masses"].reshape(-1, 48)
        full_data.append(rm_mass)

        joint_traj = npz_data["q"]
        full_joint_traj.append(joint_traj)
        computation_time = npz_data["time"].reshape(-1, 1)
        full_data_time.append(computation_time)

        full_data_path_length.append(np.array(path_length).reshape(-1, 1))
        values = torch.tensor(rm_mass)
        # Loop through each array in the .npz file
        print(f"\nMetrics for arrays in {npz_file}:")
        mean = torch.mean(values)
        std_dev = torch.std(values)
        print(f"Reflected mass cost: Mean = {mean}, Standard Deviation = {std_dev}")
        # Close the .npz file
        npz_data.close()

    # Stack all trajectories into a single numpy array
    full_data = np.vstack(full_data)
    full_data_time = np.vstack(full_data_time)
    full_data_path_length = np.vstack(full_data_path_length)
    full_joint_traj = np.vstack(full_joint_traj)

    np.savez(
        foldername,
        joint_position=full_joint_traj,
        time=full_data_time,
    )
    # Transpose the array to make each column a trajectory
    full_data = full_data.T
    full_data_time = full_data_time.T
    full_data_path_length = full_data_path_length.T
    full_joint_traj = full_joint_traj.T

    cum_rm_mass = full_data.sum(axis=0)  # shape(1, full batch)
    full_data_torch = torch.tensor(full_data)
    print("\nBatch size: ", full_data.shape[1])
    print("full batch time max: ", full_data_time.max())
    print("full batch time min: ", full_data_time.min())
    print("full batch time mean: ", full_data_time.mean())
    print("full batch time std: ", full_data_time.std())

    print("full batch rm max value; ", full_data_torch.max())
    print("full batch rm min value: ", full_data_torch.min())
    print("full batch rm mean: ", full_data_torch.mean())
    print("full batch rm std: ", full_data_torch.std())

    print("Batch cumulative rm mass max: ", cum_rm_mass.max())
    print("Batch cumulative rm mass min: ", cum_rm_mass.min())
    print("Batch cumulative rm mass mean: ", cum_rm_mass.mean())
    print("Batch cumulative rm mass std: ", cum_rm_mass.std())

    print("Batch path length max: ", full_data_path_length.max())
    print("Batch path length min: ", full_data_path_length.min())
    print("Batch path length mean: ", full_data_path_length.mean())
    print("Batch path length std: ", full_data_path_length.std())

    from robotRenderer import RobotAnimator

    theta = full_joint_traj[:, :, 2]
    theta = einops.rearrange(theta, "T H -> H T")

    import ipdb

    ipdb.set_trace()
    anim = RobotAnimator(robot_type="UR5")


# Directory containing .npz files
# directory = (
#     "logs/ur5_coppeliasim_full_path/plans/release_H48_T25_LimitsNormalizer_b64_condFalse/0/"  # Current directory
# )
# directory = "/home/ws/src/diffuser/Experiments/ur5_exp_09072024_1e-1_descent"
foldername = "exp_RRT_RM_z_C11_C8_min"
directory = f"/home/ws/src/diffuser/Experiments/C11_C8/{foldername}"


# Compute metric on arrays in .npz files in the directory
compute_metric_on_arrays(directory)
