import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the Palmerston North ozone dataset
data = np.loadtxt("data/PNOz.dat")

# Normalize data
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Define sequence length for time series (window size)
seq_length = 10  # Number of previous values used as input

# Prepare input-output pairs
X, y = [], []
for i in range(len(data) - seq_length):
    X.append(data[i : i + seq_length])  # Input sequence
    y.append(data[i + seq_length])  # Next value in sequence

X, y = np.array(X), np.array(y).reshape(-1, 1)

# Split into training and test sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define network parameters
input_size = seq_length  # 10 previous time steps as input
hidden_size = 10  # Number of hidden neurons
output_size = 1  # Predicting a single value
learning_rate = 0.01
epochs = 1000

# Initialize weights and biases (including recurrent weights)
np.random.seed(42)
W_in = np.random.randn(input_size, hidden_size) * 0.01  # Input-to-hidden weights
W_rec = np.random.randn(hidden_size, hidden_size) * 0.01  # Recurrent weights
W_out = np.random.randn(hidden_size, output_size) * 0.01  # Hidden-to-output weights
b_h = np.zeros((1, hidden_size))  # Hidden layer bias
b_o = np.zeros((1, output_size))  # Output layer bias


# Activation function (ReLU for hidden, Linear for output)
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


# Training loop
losses = []
for epoch in range(epochs):
    hidden_state = np.zeros((X_train.shape[0], hidden_size))  # Initialize hidden state

    # Forward pass (process input through all time steps)
    hidden_state = relu(np.dot(X_train, W_in) + np.dot(hidden_state, W_rec) + b_h)
    output = np.dot(hidden_state, W_out) + b_o  # Final prediction

    # Compute loss (Mean Squared Error)
    loss = np.mean((output - y_train) ** 2)
    losses.append(loss)

    # Backpropagation
    d_output = output - y_train  # Derivative of MSE loss
    dW_out = np.dot(hidden_state.T, d_output) / X_train.shape[0]
    db_o = np.sum(d_output, axis=0, keepdims=True) / X_train.shape[0]

    d_hidden = np.dot(d_output, W_out.T) * relu_derivative(hidden_state)
    dW_rec = np.dot(hidden_state.T, d_hidden) / X_train.shape[0]
    dW_in = np.dot(X_train.T, d_hidden) / X_train.shape[0]
    db_h = np.sum(d_hidden, axis=0, keepdims=True) / X_train.shape[0]

    # Update weights
    W_out -= learning_rate * dW_out
    b_o -= learning_rate * db_o
    W_rec -= learning_rate * dW_rec
    W_in -= learning_rate * dW_in
    b_h -= learning_rate * db_h

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Plot training loss
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# Evaluate on test set
hidden_state = np.zeros((X_test.shape[0], hidden_size))
hidden_state = relu(np.dot(X_test, W_in) + np.dot(hidden_state, W_rec) + b_h)
y_pred = np.dot(hidden_state, W_out) + b_o

# Rescale back to original range
y_test_orig = scaler.inverse_transform(y_test)
y_pred_orig = scaler.inverse_transform(y_pred)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_test_orig, label="Actual Data", linestyle="dashed")
plt.plot(y_pred_orig, label="RNN Prediction", color="red")
plt.legend()
plt.title("RNN-Based Time-Series Prediction on PNOz.dat")
plt.xlabel("Time Step")
plt.ylabel("Ozone Level")
plt.show()
