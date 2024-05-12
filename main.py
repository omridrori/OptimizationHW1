# Define a sample objective function (Rosenbrock)
from src.utils import plot_optimization_analysis


def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Sample methods data with paths and values
methods_data = [
    {
        'label': 'Method 1',
        'path': [(-1.5, -1), (-1, 0), (-0.5, 0.75), (0, 1)],
        'values': [500, 200, 100, 50, 25]
    },
    {
        'label': 'Method 2',
        'path': [(-1, -1), (0, 0), (0.5, 0.5)],
        'values': [450, 300, 150, 75, 30]
    }
]

# Generate the plot
fig, axes = plot_optimization_analysis(rosenbrock, [-2, 2], [-1, 3], methods_data)