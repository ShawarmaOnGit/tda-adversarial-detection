"""
TDA Feature Engineering: Convert persistence diagrams into persistence images, Betti curves, and summaries

Main methods:
1. Persistence Images: 2D histogram representation of diagrams
2. Betti Curves: Number of topological features at each scale
3. Statistical Summaries: Total persistence, mean lifetime, etc.
"""
import numpy as np
from tqdm import tqdm

class PersistenceImageGenerator:
    """
    Converts persistence diagrams into 2D images (persistence images).
    
    Notes:
      - Persistence diagram are scatter plot of (birth, death) points
      - Persistence image is like a 2D histogram/heatmap of these points
      - Each point is weighted by its lifetime, which is death - birth
      - Gaussians smoothed around each point for continuity
    """
    def __init__(self, resolution=20, birth_range=None, pers_range=None, sigma=0.1):
        """
        resolution: Grid size (20x20 = 400 features)
        birth_range: (min, max) for birth time axis
        pers_range: (min, max) for persistence (death - birth) axis
        sigma: Gaussian smoothing parameter
        """
        self.resolution = resolution
        self.birth_range = birth_range
        self.pers_range = pers_range
        self.sigma = sigma
        self.fitted = False
    

    def fit(self, diagrams_list, verbose=True):
        """
        This function looks at a bunch of persistence diagrams and figures out:
          - the smallest and largest birth values
          - the largest persistence value (death - birth)
        It saves these as the ranges so the persistence images know how wide/tall the x/y-axis should be
        In short, it learns the coordinate ranges needed to build persistence images.

        diagrams_list: List of persistence diagrams (each is Nx2 array)
        """
        if verbose:
            print(f"\nFitting to {len(diagrams_list)} diagrams...")
        
        all_births = []
        all_persistences = []
        
        for diagram in diagrams_list:
            finite_diagram = diagram[diagram[:, 1] < np.inf]
            if len(finite_diagram) == 0:
                continue
            births = finite_diagram[:, 0]
            deaths = finite_diagram[:, 1]
            persistences = deaths - births
            
            all_births.extend(births)
            all_persistences.extend(persistences)
        
        if len(all_births) == 0:
            raise ValueError("No finite points found in diagrams")
        
        if self.birth_range is None:
            self.birth_range = (min(all_births), max(all_births))
        
        if self.pers_range is None:
            self.pers_range = (0, max(all_persistences))
        
        self.fitted = True
        
        if verbose:
            print(f"Birth range: [{self.birth_range[0]:.4f}, {self.birth_range[1]:.4f}]")
            print(f"Persistence range: [{self.pers_range[0]:.4f}, {self.pers_range[1]:.4f}]")
            print(f"Image resolution: {self.resolution}x{self.resolution}")