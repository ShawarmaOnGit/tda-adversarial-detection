"""
Persistent Homology Computation: Computes topological features (H0, H1) using Ripser

H0 = Connected components (clusters)
H1 = Loops/cycles (holes in the data)
"""
import numpy as np
from ripser import ripser
import pickle
from tqdm import tqdm

def compute_persistence(points, maxdim=1, verbose=True):
    """
    Compute persistent homology of point cloud (points)
    points: numpy array of shape (N, D) - point cloud
    maxdim: Maximum homology dimension (0=components, 1=loops)
        
    Returns: diagrams - A dictionary with 'dgms' containing H0, H1 diagrams

    Plan:
      - Input: the point cloud (could be CNN features or raw pixels)
      - Risper builds connections between the points at increasing distances
      - It tracks when clusters merge (H0)
      - It tracks when loops appear and disappear (H1)
      - It summarizes this in a diagram
    """
    if verbose:
        print(f"Computing persistence for {points.shape[0]} points in {points.shape[1]}D:")
    
    result = ripser(points, maxdim=maxdim, verbose=False)
    if verbose:
        print(f"Persistence computed")
        print(f"H0 features: {len(result['dgms'][0])}")
        print(f"H1 features: {len(result['dgms'][1])}")
    return result


def save_diagrams(diagrams, filepath):
    """
    Save persistence diagrams to disk.
    diagrams: The output from ripser()
    """
    with open(filepath, 'wb') as f:
        pickle.dump(diagrams, f)
    print(f"Saved diagrams in {filepath}")


def load_diagrams(filepath):
    """
    Load persistence diagrams from disk.
    filepath in .pkl format
    diagrams: Loaded diagrams
    """
    with open(filepath, 'rb') as f:
        diagrams = pickle.load(f)
    print(f"Loaded diagrams from {filepath}")
    return diagrams