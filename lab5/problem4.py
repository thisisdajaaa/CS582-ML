import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

# Generate 10 points on an inner and outer circle
theta = np.linspace(0, 2 * np.pi, 10)
inner_circle = np.c_[np.cos(theta), np.sin(theta)]
outer_circle = np.c_[2 * np.cos(theta), 2 * np.sin(theta)]

# Combine data
X = np.vstack((inner_circle, outer_circle))
y = np.array([-1] * 10 + [1] * 10)  # Labels: -1 for inner circle, +1 for outer circle


# Polynomial kernel transformation (mapping to 3D space)
def polynomial_kernel(x):
    return np.c_[x[:, 0] ** 2, x[:, 1] ** 2, x[:, 0] * x[:, 1]]


X_poly = polynomial_kernel(X)

# Train SVM with a polynomial kernel
svm_model = SVC(kernel="poly", degree=2, C=1)
svm_model.fit(X_poly, y)

# Plot original 2D circles
plt.figure(figsize=(6, 6))
plt.scatter(
    inner_circle[:, 0], inner_circle[:, 1], color="red", label="Inner Circle (-1)"
)
plt.scatter(
    outer_circle[:, 0], outer_circle[:, 1], color="blue", label="Outer Circle (+1)"
)
plt.legend()
plt.title("Original Data (Inner and Outer Circles)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid()
plt.show()

# Plot transformed data in 3D space
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    X_poly[:10, 0],
    X_poly[:10, 1],
    X_poly[:10, 2],
    color="red",
    label="Inner Circle (-1)",
)
ax.scatter(
    X_poly[10:, 0],
    X_poly[10:, 1],
    X_poly[10:, 2],
    color="blue",
    label="Outer Circle (+1)",
)
ax.set_xlabel("X^2")
ax.set_ylabel("Y^2")
ax.set_zlabel("XY")
ax.set_title("Transformed Data (Polynomial Kernel)")
ax.legend()
plt.show()
