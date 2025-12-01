"""
TDA Visualization Utilities

This file gives the functions for plotting persistence diagrams, Betti curves, and comparisons.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_persistence_diagram(diagram, title="Persistence Diagram", save_path=None, color='steelblue', alpha=0.6, max_points=1000):
    """
    Plot a persistence diagram (birth vs death).
    """
    finite_diagram = diagram[diagram[:, 1] < np.inf]

    if len(finite_diagram) == 0:
        print("No finite points to plot.")
        return

    # Subsample if too large
    if len(finite_diagram) > max_points:
        idx = np.random.choice(len(finite_diagram), max_points, replace=False)
        finite_diagram = finite_diagram[idx]
    births = finite_diagram[:, 0]
    deaths = finite_diagram[:, 1]
    pers = deaths - births

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(births, deaths, c=pers, cmap='viridis',
                    s=30, alpha=alpha, edgecolors='black', linewidth=0.4)

    # Birth = Death diagonal
    max_val = max(births.max(), deaths.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5)

    plt.colorbar(scatter, ax=ax, label="Persistence")

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
