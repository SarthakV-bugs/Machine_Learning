# Generating a 3D SSE "bowl" for linear regression and plotting it.
# This code creates synthetic data, computes SSE over a grid of slopes and intercepts,
# and shows a 3D surface with the analytic least-squares solution marked.
# The plot uses matplotlib (single plot, no explicit colors set).
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection
from matplotlib import ticker

# Reproducible synthetic data
np.random.seed(42)
n = 50
x = np.linspace(0, 10, n)
true_slope = 2.5
true_intercept = 1.0
noise = np.random.normal(scale=2.0, size=n)
y = true_slope * x + true_intercept + noise

# Grid of slopes (m) and intercepts (b) to evaluate SSE on
m_vals = np.linspace(true_slope - 3.0, true_slope + 3.0, 120)
b_vals = np.linspace(true_intercept - 6.0, true_intercept + 6.0, 120)
M, B = np.meshgrid(m_vals, b_vals)

# Compute SSE for every (m, b) pair efficiently
# SSE(m,b) = sum (y_i - (m*x_i + b))^2
# We'll compute residuals for shape (len(b_vals), len(m_vals), n) in a vectorized way and sum over axis=2
# But to keep memory manageable we use a formula: SSE = sum(y^2) - 2*m*sum(x*y) - 2*b*sum(y) + m^2*sum(x^2) + 2*m*b*sum(x) + n*b^2
sum_y = np.sum(y)
sum_x = np.sum(x)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x * x)
sum_y2 = np.sum(y * y)
n_points = len(x)

# Using the polynomial expansion to compute SSE quickly for the whole grid:
# SSE = sum((y - (m x + b))^2) = sum(y^2) - 2 m sum(xy) - 2 b sum(y) + m^2 sum(x^2) + 2 m b sum(x) + n b^2
SSE = (sum_y2
       - 2 * M * sum_xy
       - 2 * B * sum_y
       + (M**2) * sum_x2
       + 2 * M * B * sum_x
       + n_points * (B**2))

# Analytical least squares solution (normal equations) for reference
# Design matrix with column of ones and x
A = np.vstack([x, np.ones_like(x)]).T
m_hat, b_hat = np.linalg.lstsq(A, y, rcond=None)[0]

# Prepare 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Surface plot of SSE(m,b)
surf = ax.plot_surface(M, B, SSE, linewidth=0, antialiased=True, alpha=0.9)

# Mark the analytic solution as a point on the surface
sse_at_hat = np.sum((y - (m_hat * x + b_hat))**2)
ax.scatter([m_hat], [b_hat], [sse_at_hat], s=60, marker='o')

# Add a contour projection onto the bottom plane for clarity
cset = ax.contour(M, B, SSE, zdir='z', offset=np.min(SSE) - 50, levels=12)

# Labels and title
ax.set_xlabel('slope (m)')
ax.set_ylabel('intercept (b)')
ax.set_zlabel('SSE')
ax.set_title('SSE(m, b) surface (the 3D least-squares "bowl")\nAnalytic solution marked as a point')

# Put a small text with analytic solution values
ax.text(m_hat, b_hat, sse_at_hat + 10, f"m̂={m_hat:.3f}\nb̂={b_hat:.3f}\nSSE={sse_at_hat:.1f}")

# Adjust view angle for a clearer "bowl" look
ax.view_init(elev=30, azim=-60)

# Make z ticks nicer
ax.zaxis.set_major_locator(ticker.MaxNLocator(6))

plt.show()

# Print computed analytic solution to console as well
print(f"Analytic least-squares solution: slope = {m_hat:.6f}, intercept = {b_hat:.6f}, SSE = {sse_at_hat:.6f}")
