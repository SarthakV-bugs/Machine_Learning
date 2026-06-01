import numpy as np

# Define the Transform function φ(x)
def transform(x):
    x1, x2, x3 = x
    return np.array([
        x1 * x1,
        x1 * x2,
        x1 * x3,
        x2 * x1,
        x2 * x2,
        x2 * x3,
        x3 * x1,
        x3 * x2,
        x3 * x3
    ])

# Define the kernel function K(x, y) = (x · y)^2
def kernel(x, y):
    return (np.dot(x, y)) ** 2

# Main demonstration
def main():
    # Define vectors
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])

    # Transform vectors
    phi_x = transform(x)
    phi_y = transform(y)

    # Compute dot product in higher-dimensional space
    dot_high_dim = np.dot(phi_x, phi_y)
    print("Dot product in higher-dimensional space (φ(x) · φ(y)):", dot_high_dim)

    # Compute kernel output
    kernel_output = kernel(x, y)
    print("Kernel output K(x, y) = (<x, y>)^2:", kernel_output)

    # Comparison
    if np.isclose(dot_high_dim, kernel_output):
        print("Both values are equal — kernel trick successfully demonstrated!")
    else:
        print("Values differ — check the mapping or kernel definition.")

if __name__ == "__main__":
    main()
