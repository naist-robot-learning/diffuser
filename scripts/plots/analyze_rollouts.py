import os
import numpy as np
import einops
from diffuser.robot.UR5kinematicsAndDynamics_vectorized import compute_reflected_mass, fkine
from diffuser.robot.KUKALWR4KinematicsAndDynamics_vectorized import fkine as fkineKuka
import torch


def write_table_to_file(file, data):
    file.write(data + "\n")


def compute_metric_on_arrays(directory):
    # Get list of all .npz files in the directory
    npz_files = [file for file in os.listdir(directory) if file.endswith(".npz")]

    # Loop through each .npz file
    for i, npz_file in enumerate(npz_files):

        full_data = []
        full_data_time = []
        full_data_path_length = []
        full_joint_traj = []
        # Load arrays from the .npz file
        print("File name: ", npz_file)
        npz_data = np.load(os.path.join(directory, npz_file))
        # q = npz_data["joint_position"]  # Trials x horizon x transition 100 x 48 x 6
        # values = torch.tensor(npz_data["values"])  # Trials x horizon 100 x 48

        print("Batch size of trajectories ", len(npz_data["joint_position"]))

        qs = torch.tensor(npz_data["joint_position"], dtype=torch.float32).to("cuda")
        if len(qs.size()) > 3:
            qs = qs.squeeze(dim=1)
        qs_rearrange = einops.rearrange(qs, "B H T ->  T (B H) ")

        traj_points = fkine(qs_rearrange)
        traj_points = einops.rearrange(traj_points, "(B H) T -> B H T", H=48).cpu().numpy()
        path_length = []

        for traj in traj_points:
            d = 0
            for i in range(len(traj) - 1):
                d += np.linalg.norm(traj[i + 1, :3] - traj[i, :3])
            path_length.append(d)

        # rm_mass = npz_data["rm_masses"].reshape(-1, 48)
        b, h, t = qs.shape

        qs = einops.rearrange(qs, "B H T -> B T H")
        ux = torch.zeros_like(qs[:, :3, :]).to("cuda")
        uy = torch.zeros_like(qs[:, :3, :]).to("cuda")
        uz = torch.zeros_like(qs[:, :3, :]).to("cuda")
        ux = einops.rearrange(ux, "B T H -> (B H) T").unsqueeze(dim=2)
        uy = einops.rearrange(uy, "B T H -> (B H) T").unsqueeze(dim=2)
        uz = einops.rearrange(uz, "B T H -> (B H) T").unsqueeze(dim=2)

        ux[:, 0, :] = 1
        uy[:, 1, :] = 1
        uz[:, 2, :] = 1

        rm_x = compute_reflected_mass(qs, ux)
        rm_y = compute_reflected_mass(qs, uy)
        rm_z = compute_reflected_mass(qs, uz)
        rm_x_numpy = rm_x.cpu().numpy()
        rm_y_numpy = rm_y.cpu().numpy()
        rm_z_numpy = rm_z.cpu().numpy()
        full_data.append([rm_x_numpy, rm_y_numpy, rm_z_numpy])

        joint_traj = npz_data["joint_position"]
        if len(joint_traj.shape) > 3:
            joint_traj = joint_traj.squeeze()

        full_joint_traj.append(joint_traj)
        computation_time = npz_data["time"].reshape(-1, 1)
        full_data_time.append(computation_time)

        full_data_path_length.append(np.array(path_length).reshape(-1, 1))
        # values = torch.tensor(rm_mass)
        # # Loop through each array in the .npz file
        # print(f"\nMetrics for arrays in {npz_file}:")
        # mean = torch.mean(values)
        # std_dev = torch.std(values)
        # print(f"Reflected mass cost: Mean = {mean}, Standard Deviation = {std_dev}")
        # # Close the .npz file
        npz_data.close()

        # # Stack all trajectories into a single numpy array
        full_data = np.vstack(full_data)
        full_data_time = np.vstack(full_data_time)
        full_data_path_length = np.vstack(full_data_path_length)
        full_joint_traj = np.vstack(full_joint_traj)
        # Transpose the array to make each column a trajectory
        full_data = full_data.T
        full_data_time = full_data_time.T
        full_data_path_length = full_data_path_length.T
        full_joint_traj = full_joint_traj.T

        cum_rm_mass = full_data.sum(axis=0)  # shape(1, full batch)
        full_data_torch = torch.tensor(full_data)
        np.set_printoptions(precision=3, suppress=True)
        torch.set_printoptions(precision=3)

        with open(f"{npz_file}_table_values", "w") as file:

            # file.write(f"File name: {npz_file}\n")
            write_table_to_file(file, f"File name: {npz_file}")
            # print("\nBatch size: ", full_data.shape[1])
            write_table_to_file(file, f"full batch time max: {full_data_time.max()}")
            write_table_to_file(file, f"full batch time min: ({full_data_time.min()}")
            write_table_to_file(file, f"full batch time mean: {full_data_time.mean()}")
            write_table_to_file(file, f"full batch time std: {full_data_time.std()}")

            fd_mean = full_data_time.mean()
            fd_std = full_data_time.std()
            fd_min_val = full_data_time.min()
            fd_max_val = full_data_time.max()

            write_table_to_file(file, f"time mean and std: ({fd_mean:.3f} +/- {fd_std:.3f})")
            write_table_to_file(file, f"min,max: ({fd_min_val:.3f}, {fd_max_val:.3f})")

            # print("Avg rm_x per Trajectory: ", rm_x.mean(dim=1))
            # print("Avg_rm_y per Trajectory: ", rm_y.mean(dim=1))
            # print("Avg_rm_z per Trajectory: ", rm_z.mean(dim=1))

            write_table_to_file(file, f"Avg rm_x overall exp: {rm_x.mean()}")
            write_table_to_file(file, f"Avg_rm_y overall exp: {rm_y.mean()}")
            write_table_to_file(file, f"Avg_rm_z overall exp: {rm_z.mean()}")

            write_table_to_file(file, f"Std rm_x overall exp: {rm_x.std()}")
            write_table_to_file(file, f"Std_rm_y overall exp: {rm_y.std()}")
            write_table_to_file(file, f"Std_rm_z overall exp: {rm_z.std()}")
            rm_x_mean = rm_x.mean().cpu().numpy()
            rm_y_mean = rm_y.mean().cpu().numpy()
            rm_z_mean = rm_z.mean().cpu().numpy()
            rm_x_std = rm_x.std().cpu().numpy()
            rm_y_std = rm_y.std().cpu().numpy()
            rm_z_std = rm_z.std().cpu().numpy()
            write_table_to_file(file, f"Avg RI_x, RI_y, RI_z")
            write_table_to_file(file, f"rm_x mean and std: ({rm_x_mean:.3f} +/- {rm_x_std:.3f})")
            write_table_to_file(file, f"rm_y mean and std: ({rm_y_mean:.3f} +/- {rm_y_std:.3f})")
            write_table_to_file(file, f"rm_z_mean and std: ({rm_z_mean:.3f} +/- {rm_z_std:.3f})")

            # print("Max RM_x per Trajectory: ", rm_x.max(dim=1)[0])
            # print("Min RM_x per Trajectory: ", rm_x.min(dim=1)[0])
            print("-----------------------------")
            write_table_to_file(file, f"Max RM_x overall: {rm_x.max()}")
            write_table_to_file(file, f"Min RM_x overall: {rm_x.min()}")
            # print("Max RM_y per Trajectory: ", rm_y.max(dim=1)[0])
            # print("Min RM_y per Trajectory: ", rm_y.min(dim=1)[0])
            print("-----------------------------")

            write_table_to_file(file, f"Max RM_y overall: {rm_y.max()}")
            write_table_to_file(file, f"Min RM_y overall: {rm_y.min()}")
            # print("Max RM_z per Trajectory: ", rm_z.max(dim=1)[0])
            # print("Min RM_z per Trajectory: ", rm_z.min(dim=1)[0])
            print("-----------------------------")
            write_table_to_file(file, f"Max RM_z overall: {rm_z.max()}")
            write_table_to_file(file, f"Min RM_z overall: {rm_z.min()}")

            write_table_to_file(file, f"min,max rm_x, rm_y, rm_z")
            rm_x_min = rm_x.min().cpu().numpy()
            rm_y_min = rm_y.min().cpu().numpy()
            rm_z_min = rm_z.min().cpu().numpy()
            rm_x_max = rm_x.max().cpu().numpy()
            rm_y_max = rm_y.max().cpu().numpy()
            rm_z_max = rm_z.max().cpu().numpy()
            write_table_to_file(file, f"({rm_x_min:.3f},{rm_x_max:.3f})")
            write_table_to_file(file, f"({rm_y_min:.3f},{rm_y_max:.3f})")
            write_table_to_file(file, f"({rm_z_min:.3f},{rm_z_max:.3f})")

            write_table_to_file(file, f"full batch rm max value; {full_data_torch.max()}")
            write_table_to_file(file, f"full batch rm min value: {full_data_torch.min()}")
            write_table_to_file(file, f"full batch rm mean: {full_data_torch.mean()}")
            write_table_to_file(file, f"full batch rm std: {full_data_torch.std()}")

            write_table_to_file(file, f"Batch cumulative rm max (x,y,z): {cum_rm_mass.max(axis=0)}")
            write_table_to_file(file, f"Batch cumulative rm min (x,y,z): {cum_rm_mass.min(axis=0)}")
            write_table_to_file(file, f"Batch cumulative rm mean (x,y,z): {cum_rm_mass.mean(axis=0)}")
            write_table_to_file(file, f"Batch cumulative rm std (x,y,z): {cum_rm_mass.std(axis=0)}")

            cum_rm_mean_x = cum_rm_mass.mean(axis=0)[0]
            cum_rm_mean_y = cum_rm_mass.mean(axis=0)[1]
            cum_rm_mean_z = cum_rm_mass.mean(axis=0)[2]

            cum_rm_std_x = cum_rm_mass.std(axis=0)[0]
            cum_rm_std_y = cum_rm_mass.std(axis=0)[1]
            cum_rm_std_z = cum_rm_mass.std(axis=0)[2]
            write_table_to_file(file, f"cum rm mean +/- std")
            write_table_to_file(file, f"({cum_rm_mean_x:.3f} +/- {cum_rm_std_x:.3f})")
            write_table_to_file(file, f"({cum_rm_mean_y:.3f} +/- {cum_rm_std_y:.3f})")
            write_table_to_file(file, f"({cum_rm_mean_z:.3f} +/- {cum_rm_std_z:.3f})")

            write_table_to_file(file, f"min,max cum rm_x, rm_y, rm_z")

            cum_rm_min_x = cum_rm_mass.min(axis=0)[0]
            cum_rm_min_y = cum_rm_mass.min(axis=0)[1]
            cum_rm_min_z = cum_rm_mass.min(axis=0)[2]

            cum_rm_max_x = cum_rm_mass.max(axis=0)[0]
            cum_rm_max_y = cum_rm_mass.max(axis=0)[1]
            cum_rm_max_z = cum_rm_mass.max(axis=0)[2]

            write_table_to_file(file, f"({cum_rm_min_x:.3F}, {cum_rm_max_x:.3F})")
            write_table_to_file(file, f"({cum_rm_min_y:.3F}, {cum_rm_max_y:.3F})")
            write_table_to_file(file, f"({cum_rm_min_z:.3F}, {cum_rm_max_z:.3F})")

            write_table_to_file(file, f"Batch cumulative rm mass max overall: {cum_rm_mass.max()}")
            write_table_to_file(file, f"Batch cumulative rm mass min overall: {cum_rm_mass.min()}")
            write_table_to_file(file, f"Batch cumulative rm mass mean overall: {cum_rm_mass.mean()}")
            write_table_to_file(file, f"Batch cumulative rm mass std overall: {cum_rm_mass.std()}")

            write_table_to_file(file, f"Batch path length max: {full_data_path_length.max()}")
            write_table_to_file(file, f"Batch path length min: {full_data_path_length.min()}")
            write_table_to_file(file, f"Batch path length mean: {full_data_path_length.mean()}")
            write_table_to_file(file, f"Batch path length std: {full_data_path_length.std()}")

            write_table_to_file(
                file, f"path_length: ({full_data_path_length.mean()} +/- {full_data_path_length.std()})"
            )
        from robotRenderer import RobotAnimator

        theta = full_joint_traj[:, :, 2]
        theta = einops.rearrange(theta, "T H -> H T")

        anim = RobotAnimator(robot_type="UR5")


# Directory containing .npz files
directory = (
    "logs/ur5_coppeliasim_full_path/plans/release_H48_T25_LimitsNormalizer_b64_condFalse/0/"  # Current directory
)
# directory = "/home/ws/src/diffuser/Experiments/ur5_exp_09072024_1e-1_descent"
directory = "/home/ws/src/diffuser/Experiments/C10_C14_diffuser/"

# Compute metric on arrays in .npz files in the directory
compute_metric_on_arrays(directory)
