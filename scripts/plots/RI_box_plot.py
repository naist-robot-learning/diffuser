import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from scipy import stats

# Data: Mean and Standard Deviation
# data = {
#     "Diffusion Model Std": {
#         "RI_x": [0.239, 0.083],
#         "RI_y": [0.073, 0.008],
#         "RI_z": [1.390, 0.609],
#     },
#     "Diffusion Model with RI_x min": {
#         "RI_x": [0.172, 0.042],
#         "RI_y": [0.069, 0.003],
#         "RI_z": [0.544, 0.405],
#     },
# }
model = "RRT"
path_name = "C10-C14"
plot_type = "x"
directory = f"/home/ws/src/diffuser/results/RRT/{path_name}"
npz_files = [file for file in os.listdir(directory) if file.endswith(".npz")]
# Sort files based on numeric prefix
npz_files.sort(key=lambda x: int(x.split("_")[0]))

# Prepare the data for plotting
labels = ["$X$", "$Y$"]  # , "$Z$"]
std_model_means = []
std_model_stds = []
min_model_means = []
min_model_stds = []
max_model_means = []
max_model_stds = []
box_data_min = []
box_data_max = []

for npz_file in npz_files:
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
        std_model_means = [direction.mean() for direction in rm_x_y_z]
        std_model_stds = [direction.std() for direction in rm_x_y_z]
        # Generate synthetic data for boxplots
        import ipdb

        ipdb.set_trace()
        box_data_std = [rm_x_y_z[0], rm_x_y_z[1], rm_x_y_z[2]]
        print("min: ", rm_x_y_z[0].min())
        print("max: ", rm_x_y_z[0].max())
        print("min: ", rm_x_y_z[1].min())
        print("max: ", rm_x_y_z[1].max())
        print("min: ", rm_x_y_z[2].min())
        print("max: ", rm_x_y_z[2].max())

    elif "min" in npz_file:
        if "_x_" in npz_file:
            min_model_means.append(rm_x_y_z[0].mean())
            min_model_stds.append(rm_x_y_z[0].std())
            box_data_min.append(rm_x_y_z[0])
            print("min: ", rm_x_y_z[0].min())
            print("max: ", rm_x_y_z[0].max())

        elif "_y_" in npz_file:
            if plot_type == "x":
                continue
            min_model_means.append(rm_x_y_z[1].mean())
            min_model_stds.append(rm_x_y_z[1].std())
            box_data_min.append(rm_x_y_z[1])
            print("min: ", rm_x_y_z[1].min())
            print("max: ", rm_x_y_z[1].max())

        elif "_z_" in npz_file:
            if plot_type == "xy" or plot_type == "x":
                continue
            min_model_means.append(rm_x_y_z[2].mean())
            min_model_stds.append(rm_x_y_z[2].std())
            box_data_min.append(rm_x_y_z[2])
            print("min: ", rm_x_y_z[2].min())
            print("max: ", rm_x_y_z[2].max())
            print("mean: ", rm_x_y_z[2].mean())
            print("std: ", rm_x_y_z[2].std())

    elif "max" in npz_file:
        if "_x_" in npz_file:
            max_model_means.append(rm_x_y_z[0].mean())
            max_model_stds.append(rm_x_y_z[0].std())
            box_data_max.append(rm_x_y_z[0])
        if "_y_" in npz_file:
            max_model_means.append(rm_x_y_z[1].mean())
            max_model_stds.append(rm_x_y_z[1].std())
            box_data_max.append(rm_x_y_z[1])
        if "_z_" in npz_file:
            max_model_means.append(rm_x_y_z[2].mean())
            max_model_stds.append(rm_x_y_z[2].std())
            box_data_max.append(rm_x_y_z[2])
    else:
        raise ValueError("Something is wrong")

# Prepare data for Seaborn
plot_data = []
group_labels = []

b, h = box_data_std[0].shape
for i, label in enumerate(labels):
    if (plot_type == "xy" and i == 2) or (plot_type == "x" and i >= 1) or (plot_type == "y" and i == 0):
        continue

    plot_data.extend(box_data_std[i].reshape(-1))
    group_labels.extend([f"{label} (Std)"] * b * h)

    plot_data.extend(box_data_min[i].reshape(-1))
    group_labels.extend([f"{label} (min)"] * b * h)

# Define the color palette manually
palette = {
    "$X$ (Std)": "gray",
    "$X$ (min)": "blue",
    "$Y$ (Std)": "gray",
    "$Y$ (min)": "blue",
    "$Z$ (Std)": "gray",
    "$Z$ (min)": "blue",
    # Add entries for other labels if needed
}

# Convert numpy arrays to pandas Series
group_labels = pd.Series(group_labels)
plot_data = pd.Series(plot_data)
# Plotting the box plot
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.set_context("talk", font_scale=0.8)

sns.boxplot(x=group_labels, y=plot_data, notch=True, palette=palette)

# Perform t-tests and annotate results
for i, label in enumerate(labels):
    if (plot_type == "xy" and i == 2) or (plot_type == "x" and i >= 1):
        continue
    t_stat, p_val = stats.ttest_ind(box_data_std[i].reshape(-1), box_data_min[i].reshape(-1), equal_var=False)
    x_pos = (2 * i) + 0.5
    y_pos = (
        max(
            std_model_means[i].reshape(-1) + std_model_stds[i].reshape(-1),
            min_model_means[i].reshape(-1) + min_model_stds[i].reshape(-1),
        )
        + 0.1
    )
    plt.text(x_pos, y_pos, f"p = {p_val:.3f}", ha="center", va="bottom", color="blue")

# plt.xlabel("Unit direction")
plt.xlabel("")
plt.ylabel("Reflected Inertia (Kg)")
plt.xticks()
# plt.tight_layout()
plt.savefig(f"boxplot_RI_min_{model}_{path_name}_{plot_type}.pdf", format="pdf", bbox_inches="tight")
plt.show()
# Print t-test results
for i, label in enumerate(labels):
    t_stat, p_val = stats.ttest_ind(box_data_std[i].reshape(-1), box_data_min[i].reshape(-1), equal_var=False)
    print(f"T-test for {label}: t-statistic = {t_stat:.3f}, p-value = {p_val:.3f}")
