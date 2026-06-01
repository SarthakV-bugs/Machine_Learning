from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Create data
X, y = make_moons(n_samples=200, noise=0.25, random_state=42)

# Visualize
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1")
plt.legend(); plt.title("Non-linear 2-class dataset"); plt.show()
