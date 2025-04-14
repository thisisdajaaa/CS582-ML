import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Define the given data points
positive_class = np.array([[2, 2], [2, -2], [-2, -2], [-2, 2]])
negative_class = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])

X = np.vstack((positive_class, negative_class))
y = np.array([1, 1, 1, 1, -1, -1, -1, -1])  # Labels


# Transformation function
def transform(x1, x2):
    if np.sqrt(x1**2 + x2**2) > 2:
        return np.array([4 - x2 + abs(x1 - x2), 4 - x1 + abs(x1 - x2)])
    else:
        return np.array([x1, x2])


# Apply transformation to all points
X_transformed = np.array([transform(x1, x2) for x1, x2 in X])

# Train SVM on transformed data
svm_model = SVC(kernel="linear")
svm_model.fit(X_transformed, y)

# Get decision boundary
w = svm_model.coef_[0]
b = svm_model.intercept_[0]
x_plot = np.linspace(-1, 6, 100)
y_plot = -(w[0] / w[1]) * x_plot - (b / w[1])

# Plot transformed data
plt.figure(figsize=(6, 6))
plt.scatter(
    X_transformed[:4, 0], X_transformed[:4, 1], color="blue", label="Class 1 (+1)"
)
plt.scatter(
    X_transformed[4:, 0], X_transformed[4:, 1], color="red", label="Class 2 (-1)"
)
plt.plot(x_plot, y_plot, "k-", label="SVM Decision Boundary")

plt.legend()
plt.title("Transformed Data and Decision Boundary")
plt.xlabel("Transformed X")
plt.ylabel("Transformed Y")
plt.grid()
plt.show()
