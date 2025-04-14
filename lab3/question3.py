import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load Pima Indian dataset (download from UCI repository or use sklearn)
data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

df = pd.read_csv(data_url, names=column_names)

# Split into input features (X) and target variable (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Standardize data for better MLP performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define Neural Network parameters
input_size = X.shape[1]  # 8 features
hidden_size1 = 16  # First hidden layer (increased neurons)
hidden_size2 = 8  # Second hidden layer
output_size = 1  # Binary classification (0 or 1)
learning_rate = 0.01
epochs = 1000

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size1) * 0.01
b1 = np.zeros((1, hidden_size1))
W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
b2 = np.zeros((1, hidden_size2))
W3 = np.random.randn(hidden_size2, output_size) * 0.01
b3 = np.zeros((1, output_size))


# Activation function and its derivative (ReLU for hidden layers, Sigmoid for output)
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Training loop
losses = []
for epoch in range(epochs):
    # Forward propagation
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)  # Final output

    # Compute loss (Binary Cross-Entropy)
    loss = -np.mean(y_train * np.log(A3 + 1e-8) + (1 - y_train) * np.log(1 - A3 + 1e-8))
    losses.append(loss)

    # Backpropagation
    dA3 = A3 - y_train.reshape(-1, 1)  # Output layer gradient
    dZ3 = dA3 * sigmoid_derivative(A3)
    dW3 = np.dot(A2.T, dZ3) / X_train.shape[0]
    db3 = np.sum(dZ3, axis=0, keepdims=True) / X_train.shape[0]

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * relu_derivative(A2)
    dW2 = np.dot(A1.T, dZ2) / X_train.shape[0]
    db2 = np.sum(dZ2, axis=0, keepdims=True) / X_train.shape[0]

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(A1)
    dW1 = np.dot(X_train.T, dZ1) / X_train.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / X_train.shape[0]

    # Gradient descent updates
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Plot loss curve
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# Evaluate on test data
Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)

Z2_test = np.dot(A1_test, W2) + b2
A2_test = relu(Z2_test)

Z3_test = np.dot(A2_test, W3) + b3
A3_test = sigmoid(Z3_test)

y_pred = (A3_test > 0.5).astype(int)  # Convert probabilities to binary labels

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
