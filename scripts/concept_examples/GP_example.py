import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Step 1: Generate synthetic data
np.random.seed(1)
X = np.random.uniform(-5, 5, 20).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

# Step 2: Define the kernel and the Gaussian Process
# Kernel: k(x_i, x_j) = C * RBF
kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1**2)

# Step 3: Fit the model
gp.fit(X, y)

# Step 4: Make predictions
x_pred = np.linspace(-6, 6, 1000).reshape(-1, 1)
y_pred, sigma = gp.predict(x_pred, return_std=True)

# Step 5: Plot the results
plt.figure(figsize=(10, 5))
plt.plot(X, y, "r.", markersize=10, label="Observations")
plt.plot(x_pred, np.sin(x_pred), "b-", label="True function")
plt.plot(x_pred, y_pred, "k-", label="GPR prediction")
plt.fill_between(
    x_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, label="95% confidence interval"
)

plt.title("Gaussian Process Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
