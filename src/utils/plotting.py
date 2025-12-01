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


def plot_betti_curve(epsilons, betti_numbers, title="Betti Curve", save_path=None, color='darkgreen', linewidth=2):
    """
    Plot a Betti curve (features alive vs epsilon).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epsilons, betti_numbers, color=color, linewidth=linewidth)
    ax.fill_between(epsilons, betti_numbers, color=color, alpha=0.3)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Betti Number")
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_persistence_image(image, title="Persistence Image", save_path=None, cmap='hot'):
    """
    Plot a persistence image as a heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(image, cmap=cmap, origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax, label="Weight")
    ax.set_xlabel("Birth")
    ax.set_ylabel("Persistence")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def compare_persistence_diagrams(diagram1, diagram2, label1="Clean", label2="Adversarial", title="Persistence Diagram Comparison", save_path=None):
    """
    Plot two persistence diagrams side by side.
    """
    finite_diagram1 = diagram1[diagram1[:, 1] < np.inf]
    finite_diagram2 = diagram2[diagram2[:, 1] < np.inf]

    if len(finite_diagram1) == 0 and len(finite_diagram2) == 0:
        print("No finite points in either diagram.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left diagram ---
    if len(finite_diagram1) > 0:
        b1, e1 = finite_diagram1[:, 0], finite_diagram1[:, 1]
        p1 = e1 - b1
        sc1 = ax1.scatter(b1, e1, c=p1, cmap="Blues", s=30, alpha=0.6,
                          edgecolors="black", linewidth=0.4)
        m1 = max(b1.max(), e1.max())
        ax1.plot([0, m1], [0, m1], "r--", linewidth=1.5)
        plt.colorbar(sc1, ax=ax1, label="Persistence")

    ax1.set_title(label1)
    ax1.set_xlabel("Birth")
    ax1.set_ylabel("Death")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_aspect("equal")

    # --- Right diagram ---
    if len(finite_diagram2) > 0:
        b2, e2 = finite_diagram2[:, 0], finite_diagram2[:, 1]
        p2 = e2 - b2
        sc2 = ax2.scatter(b2, e2, c=p2, cmap="Reds", s=30, alpha=0.6,
                          edgecolors="black", linewidth=0.4)
        m2 = max(b2.max(), e2.max())
        ax2.plot([0, m2], [0, m2], "r--", linewidth=1.5)
        plt.colorbar(sc2, ax=ax2, label="Persistence")

    ax2.set_title(label2)
    ax2.set_xlabel("Birth")
    ax2.set_ylabel("Death")
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.set_aspect("equal")
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def compare_betti_curves(epsilons1, betti1, epsilons2, betti2, label1="Clean", label2="Adversarial", title="Betti Curve Comparison", save_path=None):
    """
    Plot two Betti curves for comparison.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(epsilons1, betti1, color="blue", label=label1)
    ax.fill_between(epsilons1, betti1, color="blue", alpha=0.2)
    ax.plot(epsilons2, betti2, color="red", label=label2)
    ax.fill_between(epsilons2, betti2, color="red", alpha=0.2)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Betti Number")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_statistics_comparison(stats1, stats2, label1="Clean", label2="Adversarial", title="Topological Statistics Comparison", save_path=None):
    """
    Compare persistence statistics between two datasets using boxplots.
    """
    metrics = ["n_features", "total_persistence", "mean_persistence", "max_persistence", "std_persistence"]
    labels = ["Num Features", "Total Pers.", "Mean Pers.", "Max Pers.", "Std Pers."]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (metric, lab) in enumerate(zip(metrics, labels)):
        ax = axes[i]
        data1 = stats1[metric]
        data2 = stats2[metric]
        
        # Box plots
        bp = ax.boxplot([data1, data2],
                        positions=[1, 2],
                        widths=0.6,
                        patch_artist=True,
                        showmeans=True)

        # Color the boxes
        bp["boxes"][0].set_facecolor("lightblue")
        bp["boxes"][1].set_facecolor("lightcoral")
        ax.set_xticks([1, 2])
        ax.set_xticklabels([label1, label2])
        ax.set_ylabel(lab)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Remove last empty subplot
    fig.delaxes(axes[5])

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()