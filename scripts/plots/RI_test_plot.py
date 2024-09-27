import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual data)
case_1_inertia = [5.0, 7.0, 6.0]  # Inertia values for Case 1 (x, y, z)
case_2_inertia = 4.0  # Inertia value for Case 2 (e.g., x-direction)
case_3_inertia = 5.5  # Inertia value for Case 3 (e.g., y-direction)
case_4_inertia = 6.5  # Inertia value for Case 4 (e.g., z-direction)

# Prepare data for plotting
labels = ["X", "Y", "Z"]
case_1_values = case_1_inertia
case_2_values = [case_2_inertia, np.nan, np.nan]  # Assume inertia only in the x-direction
case_3_values = [np.nan, case_3_inertia, np.nan]  # Assume inertia only in the y-direction
case_4_values = [np.nan, np.nan, case_4_inertia]  # Assume inertia only in the z-direction

# Bar width
width = 0.2

# Position of bars on x-axis
x = np.arange(len(labels))

# Plotting the data
plt.figure(figsize=(10, 6))
plt.bar(x - width, case_1_values, width, label="Case 1")
plt.bar(x, case_2_values, width, label="Case 2")
plt.bar(x + width, case_3_values, width, label="Case 3")
plt.bar(x + 2 * width, case_4_values, width, label="Case 4")

# Adding labels and title
plt.xlabel("Direction")
plt.ylabel("Reflected Inertia")
plt.title("Reflected Inertia in Different Directions for Multiple Cases")
plt.xticks(x + width / 2, labels)
plt.legend()

# Show plot
plt.grid(True)
plt.show()
