import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data_path = os.path.join("data", "PNOz.dat")

data = np.loadtxt(data_path)

print("Dataset loaded successfully! First few rows:")
print(data[:5])  # Print the first few rows to verify

# Normalize data
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1, 1))

# Prepare time-series input and output pairs
window_size = 10  # Define the number of past values to use for prediction
X, y = [], []

for i in range(len(data) - window_size):
    X.append(data[i : i + window_size].flatten())  # Input sequence
    y.append(data[i + window_size])  # Next value

X, y = np.array(X), np.array(y)

# Split into training and testing sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define and train the MLP model
mlp = MLPRegressor(
    hidden_layer_sizes=(10, 10), activation="tanh", solver="adam", max_iter=5000
)
mlp.fit(X_train, y_train.ravel())

# Make predictions
y_pred = mlp.predict(X_test)

# Rescale back to original range
y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(y_test_orig, label="Actual Data", linestyle="dashed")
plt.plot(y_pred_orig, label="MLP Prediction", color="red")
plt.legend()
plt.title("MLP Time-Series Prediction on PNOz.dat")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.show()
