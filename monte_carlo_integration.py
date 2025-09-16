import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return x ** 2

# Numerical solution using the trapezoidal rule
def numerical_integration(a, b, n):
    x = np.linspace(a, b, n)
    y = f(x)
    return (b - a) / (2 * n) * (y[0] + 2 * np.sum(y[1:n-1]) + y[n-1])

# Min-max detection
def min_max_detection(a, b):
    x = np.linspace(a, b, 100)
    y = f(x)
    return np.min(y), np.max(y)

# Monte Carlo method for integration
def monte_carlo_integration(a, b, num_samples):
    x_random = np.random.uniform(a, b, num_samples)
    y_random = f(x_random)
    return (b - a) * np.mean(y_random)

# Visualization
def visualize_integration(a, b, num_samples):
    x = np.linspace(a, b, 100)
    plt.plot(x, f(x), label='f(x) = x^2', color='blue')
    
    x_random = np.random.uniform(a, b, num_samples)
    y_random = f(x_random)
    
    plt.scatter(x_random, y_random, color='red', alpha=0.5, label='Random Samples')
    plt.title('Monte Carlo Integration Visualization')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()
    plt.show()

# Parameters
a = 0
b = 3
n = 100
num_samples = 1000

# Execute the methods
numerical_result = numerical_integration(a, b, n)
min_value, max_value = min_max_detection(a, b)
monte_carlo_result = monte_carlo_integration(a, b, num_samples)

# Print results
print(f'Numerical Integration Result: {numerical_result}')
print(f'Min-Max Values: {min_value}, {max_value}')
print(f'Monte Carlo Integration Result: {monte_carlo_result}')

# Visualize the results
visualize_integration(a, b, num_samples)