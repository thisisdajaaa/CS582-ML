import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron

# Define the dataset (same as Problem 1)
class_1 = np.array([[1, 1], [1, 2], [2, 1]])  # Positive class (+1)
class_2 = np.array([[0, 0], [1, 0], [0, 1]])  # Negative class (-1)

# Combine data points
X = np.vstack((class_1, class_2))
y = np.array([1, 1, 1, -1, -1, -1])  # Labels

# Train SVM model
svm_model = SVC(kernel="linear")
svm_model.fit(X, y)

# Get SVM decision boundary
w_svm = svm_model.coef_[0]
b_svm = svm_model.intercept_[0]
x_plot = np.linspace(-1, 3, 100)
y_svm = -(w_svm[0] / w_svm[1]) * x_plot - (b_svm / w_svm[1])

# Train Perceptron model
perceptron = Perceptron()
perceptron.fit(X, y)

# Get Perceptron decision boundary
w_p = perceptron.coef_[0]
b_p = perceptron.intercept_[0]
y_perceptron = -(w_p[0] / w_p[1]) * x_plot - (b_p / w_p[1])

# Plot data points
plt.figure(figsize=(6, 6))
plt.scatter(class_1[:, 0], class_1[:, 1], color="blue", label="Class 1 (+1)")
plt.scatter(class_2[:, 0], class_2[:, 1], color="red", label="Class 2 (-1)")
plt.plot(x_plot, y_svm, "k-", label="SVM Boundary")
plt.plot(x_plot, y_perceptron, "g--", label="Perceptron Boundary")

# Highlight support vectors (for SVM)
support_vectors = svm_model.support_vectors_
plt.scatter(
    support_vectors[:, 0],
    support_vectors[:, 1],
    s=100,
    edgecolors="k",
    facecolors="none",
    label="SVM Support Vectors",
)

plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.legend()
plt.title("SVM vs Perceptron Decision Boundaries")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid()
plt.show()
