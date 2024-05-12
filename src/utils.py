import numpy as np
import matplotlib.pyplot as plt


def plot_optimization_analysis(f, xlimits, ylimits, methods_data, levels=50, title="Optimization Analysis"):
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot the contours
    x = np.linspace(xlimits[0], xlimits[1], 400)
    y = np.linspace(ylimits[0], ylimits[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(lambda x, y: f(np.array([x, y])))(X, Y)

    cp = axes[0].contour(X, Y, Z, levels, colors='black')
    axes[0].clabel(cp, inline=True, fontsize=8)
    axes[0].set_title('Contour Plot')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')

    # If paths are provided, plot them on the contour plot
    if methods_data:
        for data in methods_data:
            if 'path' in data:
                path = data['path']
                axes[0].plot(*zip(*path), marker='o', label=data['label'])
        axes[0].legend()

    # Plot function values vs iteration
    for data in methods_data:
        if 'values' in data:
            axes[1].plot(data['values'], label=data['label'], marker='o')

    axes[1].set_title('Function Value vs Iteration')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Function Value')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
    return fig, axes