import pandas as pd
import matplotlib.pyplot as plt

# Sample data (replace with your actual data)

data = {
    "Setup": ["Setup1", "Setup2", "Setup3"],
    "Computation Time": [120, 150, 130],
    "MIN Computation Time": [100, 120, 110],
    "MAX Computation Time": [140, 170, 150],
    "RM": [0.8, 0.85, 0.78],
    "MIN RM": [0.75, 0.8, 0.72],
    "MAX RM": [0.85, 0.9, 0.82],
    "CumRM": [10, 15, 12],
    "MIN CumRM": [9, 14, 11],
    "MAX CumRM": [11, 16, 13],
    "Path Length (mean + std)": ["5±0.5", "6±0.6", "4.5±0.4"],
}
# Create DataFrame
df = pd.DataFrame(data)

# Plot the table
fig, ax = plt.subplots(figsize=(12, 4))

# Set the figure size as needed
ax.axis("tight")
ax.axis("off")
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")

# Adjust layout
plt.tight_layout()

# Save the table as an image

plt.savefig("table_image.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.show()
