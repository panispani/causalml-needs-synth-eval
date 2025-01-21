import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis


# g(U) = sin(2 * pi * f * U)
def g(U, f):
    return np.sin(2 * np.pi * f * U)


# Define the range of U and the values of f
U = np.linspace(0, 1, 500)
frequencies = [1 / 4, 1 / 3, 1 / 2, 3 / 2, 1.0, 2.0, 5.0, 10.0]

# Plot the function for each value of f
plt.figure(figsize=(12, 8))
for f in frequencies:
    plt.plot(U, g(U, f), label=f"f = {f}")


# Customize the plot
# Define a colormap for the gradient
colors = viridis(np.linspace(0, 1, len(frequencies)))
plt.figure(figsize=(12, 8))
for f, color in zip(frequencies, colors):
    plt.plot(U, g(U, f), label=f"f = {f}", color=color)
plt.title(r"Sketch of $g(U) = \sin(2 \pi f U)$ for Different Values of $f$")
plt.xlabel("U")
plt.ylabel(r"$g(U)$")
plt.legend()
plt.grid(True)
# plt.show()

# Save the plot as a PNG file
output_path = "sin_function_plot.png"
plt.savefig(output_path)
plt.close()
