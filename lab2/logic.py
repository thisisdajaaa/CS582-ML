from perceptron import pcn  # Import the perceptron class
import numpy as np

def logic():
    """ Run AND and XOR logic functions """

    # Define AND dataset
    a = np.array([[0,0,0],
                  [0,1,0],
                  [1,0,0],
                  [1,1,1]])

    # Define XOR dataset
    b = np.array([[0,0,0],
                  [0,1,1],
                  [1,0,1],
                  [1,1,0]])

    # Train Perceptron on AND logic
    print("\nTraining Perceptron on AND gate")
    p = pcn(a[:,0:2], a[:,2:])
    p.pcntrain(a[:,0:2], a[:,2:], 0.25, 10)
    p.confmat(a[:,0:2], a[:,2:])

    # Train Perceptron on XOR logic
    print("\nTraining Perceptron on XOR gate")
    q = pcn(b[:,0:2], b[:,2:])
    q.pcntrain(b[:,0:2], b[:,2:], 0.25, 10)
    q.confmat(b[:,0:2], b[:,2:])

# ✅ Ensure the function runs when the script is executed
if __name__ == "__main__":
    logic()
