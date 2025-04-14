import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Given data
data = {
    "X": [10, 12, 13, 13, 10, 10, 11, 8, 8, 10, 13, 10, 9, 8, 10, 5, 11, 10, 9, 10, 8, 12, 11, 13],
    "Y": [11, 13, 15, 10, 12, 7, 12, 10, 9, 12, 13, 7, 12, 6, 13, 8, 12, 9, 11, 11, 9, 9, 8, 6]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Compute statistical parameters
Mx = np.mean(df["X"])
My = np.mean(df["Y"])
Sx = np.std(df["X"], ddof=1)  # Sample standard deviation
Sy = np.std(df["Y"], ddof=1)  # Sample standard deviation
r = np.corrcoef(df["X"], df["Y"])[0, 1]  # Pearson correlation coefficient

# Compute regression line parameters
m = r * (Sy / Sx)  # Slope
c = My - m * Mx  # Intercept

# Display results
results = {
    "Mean X": Mx,
    "Mean Y": My,
    "Std Dev X": Sx,
    "Std Dev Y": Sy,
    "Correlation (r)": r,
    "Slope (m)": m,
    "Intercept (c)": c
}

# Convert results to DataFrame
results_df = pd.DataFrame([results])

# Print results instead of using ace_tools
print("\nStatistical Parameters:\n", results_df)

# Plot the regression line
plt.scatter(df["X"], df["Y"], color="blue", label="Data Points")
x_line = np.linspace(min(df["X"]), max(df["X"]), 100)
y_line = m * x_line + c
plt.plot(x_line, y_line, color="red", linestyle="dashed", label=f"y = {m:.2f}x + {c:.2f}")

# Labels and title
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Linear Regression Plot")
plt.legend()
plt.grid()

# Show plot
plt.show()
