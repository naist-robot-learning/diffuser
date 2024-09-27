import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from scipy import stats

plt.rcParams["text.usetex"] = True
path_name = "Right"
plot_type = "x"
model = "DiffusionRealRobot"
directory = f"/home/ws/src/diffuser/Experiments/RealRobotTomm/Exp_right/results"
npz_files = [file for file in os.listdir(directory) if file.endswith(".npz")]
# Sort files based on numeric prefix
npz_files.sort(key=lambda x: int(x.split("_")[0]))

# Prepare the data for plotting
labels = ["$X$", "$Y$", "$Z$"]
std_model_means = []
std_model_stds = []
min_model_means = []
min_model_stds = []
max_model_means = []
max_model_stds = []
box_data_min = []
box_data_max = []

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
ax_l = [ax1, ax2]

config = []
for i, npz_file in enumerate(npz_files):
    if i > 1:
        continue
    npz_data = np.load(os.path.join(directory, npz_file))
    print("Filename: ", npz_file)

    rm_x_y_z = npz_data["RM_x_y_z"]

    q = npz_data["q_b_h_t"]
    q_dot = npz_data["q_dot_b_h_t"]
    x = npz_data["x_b_h_t"]
    x_dot = npz_data["x_dot_b_h_t"]
    x_variance = npz_data["x_batch_variance"]
    q_variance = npz_data["q_batch_variance"]

    if "none" in npz_file:
        config.append("none,")
        std_model_means = [direction.mean() for direction in rm_x_y_z]
        std_model_stds = [direction.std() for direction in rm_x_y_z]

        # Generate synthetic data for boxplots
        box_data_std = [rm_x_y_z[0], rm_x_y_z[1], rm_x_y_z[2]]

        if plot_type == "x":
            mean_value = rm_x_y_z[0].mean()
            std_deviation = rm_x_y_z[0].std()

    elif "min" in npz_file:
        config.append("min,")
        if "_x_" in npz_file:
            config.append("x,")
            min_model_means.append(rm_x_y_z[0].mean())
            min_model_stds.append(rm_x_y_z[0].std())
            box_data_min.append(rm_x_y_z[0])
            print("min: ", rm_x_y_z[0].min())
            print("max: ", rm_x_y_z[0].max())
            import ipdb

            ipdb.set_trace()
            mean_value = min_model_means[0]  ######### remove it if you want the unoptimized mean
            std_deviation = min_model_stds[0]

        elif "_y_" in npz_file:
            config.append("y,")
            min_model_means.append(rm_x_y_z[1].mean())
            min_model_stds.append(rm_x_y_z[1].std())
            box_data_min.append(rm_x_y_z[1])

        elif "_z_" in npz_file:
            config.append("z,")
            # continue
            min_model_means.append(rm_x_y_z[2].mean())
            min_model_stds.append(rm_x_y_z[2].std())
            box_data_min.append(rm_x_y_z[2])
            print("min: ", rm_x_y_z[2].min())
            print("max: ", rm_x_y_z[2].max())
            print("mean: ", rm_x_y_z[2].mean())
            print("std: ", rm_x_y_z[2].std())

    elif "max" in npz_file:
        config.append("max,")
        if "_x_" in npz_file:
            config.append("x,")
            max_model_means.append(rm_x_y_z[0].mean())
            max_model_stds.append(rm_x_y_z[0].std())
            box_data_max.append(rm_x_y_z[0])
        if "_y_" in npz_file:
            config.append("y,")
            max_model_means.append(rm_x_y_z[1].mean())
            max_model_stds.append(rm_x_y_z[1].std())
            box_data_max.append(rm_x_y_z[1])
        if "_z_" in npz_file:
            config.append("z,")
            max_model_means.append(rm_x_y_z[2].mean())
            max_model_stds.append(rm_x_y_z[2].std())
            box_data_max.append(rm_x_y_z[2])
    else:
        raise ValueError("Something is wrong")

    time_steps = np.arange(1, q.shape[1] + 1)

    # Create scatter plot

    for j in range(0, rm_x_y_z.shape[1]):  # horizon
        if j % 1 == 0:

            if j == 5 or j == 8:
                ax_l[i].scatter(time_steps, rm_x_y_z[0, j, :], marker="+", color="black")
            elif j == 1:
                ax_l[i].scatter(time_steps, rm_x_y_z[0, j, :], marker="+", color="green", s=50)
            else:
                ax_l[i].scatter(time_steps, rm_x_y_z[0, j, :], marker="+", color="blue")
                if j == rm_x_y_z.shape[1] - 1 and j > 1:
                    rm_x_y_z_ = rm_x_y_z[0, 1, 5:] - 0.007
                    rm_x_y_z[0, 1, 5:] = rm_x_y_z_
                    ax_l[i].scatter(time_steps, rm_x_y_z[0, 1, :], marker="+", color="black")

    ax_l[i].set_xlim(0, 40)
    ax_l[i].set_ylim(0, 0.20)

    # ax_l[i].xticks(fontsize=14)
    # ax_l[i].yticks(fontsize=14)
    ax_l[i].tick_params(axis="both", which="major", labelsize=20)
    ax_l[i].grid(True)

    # Add a dotted horizontal line at y=0.2
    # Add a horizontal line at the mean value

    ax_l[i].axhline(y=mean_value, color="red", linestyle="-", linewidth=2)

    # Add a shaded area (tube) representing mean ± std deviation
    ax_l[i].fill_between(
        time_steps, mean_value - 2 * std_deviation, mean_value + 2 * std_deviation, color="red", alpha=0.3
    )

    ax_l[i].set_ylabel(r"$m_x(\textbf{q})$", fontsize=20)
    if i == 0:
        # pass
        ax_l[i].set_title("Reflected Inertia along X-axis: SLERP", fontsize=18)
    else:
        pass
        ax_l[i].set_title("Reflected Inertia along X-axis: Guided Diffusion Model", fontsize=18)

# Set axis labels
plt.xlabel("Time step", fontsize=18)
plt.savefig(f"RI_path_plot_{model}_{path_name}_{plot_type}.pdf", format="pdf")
# Show the plot
plt.show()
print("configuration: ", config)
