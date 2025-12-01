"""
Persistent Homology Computation: Computes topological features (H0, H1) using Ripser

H0 = Connected components (clusters)
H1 = Loops/cycles (holes in the data)
"""
import numpy as np
from ripser import ripser
import pickle
from tqdm import tqdm