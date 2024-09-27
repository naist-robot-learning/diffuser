import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Hypothetical 64 samples from a posterior distribution
samples = np.random.normal(loc=2.0, scale=1.0, size=64)

# Estimate the density using Kernel Density Estimation (KDE)
kde = gaussian_kde(samples)
x_vals = np.linspace(min(samples), max(samples), 1000)
kde_vals = kde(x_vals)

# Plot the KDE and the samples
plt.plot(x_vals, kde_vals, label="Estimated Posterior Density")
plt.hist(samples, bins=15, density=True, alpha=0.5, label="Posterior Samples")
plt.legend()
plt.show()

# Find the MAP estimate
map_index = np.argmax(kde_vals)
map_estimate = x_vals[map_index]

print("MAP Estimate:", map_estimate)
