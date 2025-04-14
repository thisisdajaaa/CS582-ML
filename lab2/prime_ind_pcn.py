import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from perceptron import pcn  # Import perceptron class from your previous implementation

# 1️⃣ Load Dataset from UCI Repository
def load_dataset():
    file_path = "data/iris/iris.data"  # Update this if necessary
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    
    df = pd.read_csv(file_path, header=None, names=column_names)

    return df


# 2️⃣ Preprocess Data (Normalize & Convert Labels)
def preprocess_data(df):
    df = df.copy()  # Ensure we're modifying a copy
    
    # Convert class labels to numerical values
    class_mapping = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }
    df["class"] = df["class"].map(class_mapping)  # Convert class labels to numbers
    
    # Handle missing values
    df = df.fillna(df.iloc[:, :-1].mean())  
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])  
    
    return df


# 3️⃣ Plot Dataset to Visualize Features
def plot_features(df):
    x1, x2, y = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, -1]
    plt.scatter(x1[y==0], x2[y==0], label="Class 0", color='red')
    plt.scatter(x1[y==1], x2[y==1], label="Class 1", color='blue')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Scatter Plot of Two Features")
    plt.show()

# 4️⃣ Train Perceptron on the Dataset
def train_perceptron(X, y):
    perceptron = pcn(X, y)
    perceptron.pcntrain(X, y, eta=0.1, nIterations=50)
    perceptron.confmat(X, y)
    return perceptron

# 5️⃣ Plot Decision Boundary
def plot_decision_boundary(perceptron, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid = np.concatenate((grid, -np.ones((grid.shape[0], 1))), axis=1)  # Add bias term
    Z = perceptron.pcnfwd(grid).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], color='red', label="Class 0")
    plt.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], color='blue', label="Class 1")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Perceptron Decision Boundary")
    plt.legend()
    plt.show()

# ✅ Main function to execute the entire pipeline
def main():
    df = load_dataset()
    df = preprocess_data(df)
    plot_features(df)
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values.reshape(-1, 1)
    perceptron = train_perceptron(X, y)
    plot_decision_boundary(perceptron, X, y)

# Run the script
if __name__ == "__main__":
    main()
