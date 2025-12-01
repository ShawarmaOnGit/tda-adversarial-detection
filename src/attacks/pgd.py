"""
Projected Gradient Descent (PGD) Attack:
Iterative version of FGSM that applies multiple small perturbations while
staying within an epsilon ball around the original image.

PGD is stronger than FGSM because:
- Multiple iterations refine the perturbation
- Projection step ensures perturbation stays within epsilon bound
- Can escape shallow local minima that FGSM gets stuck in
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm


class PGDAttack:
    """
    Projected Gradient Descent (PGD) adversarial attack.
    
    PGD is an iterative version of FGSM that:
    1. Takes multiple small steps in the gradient direction
    2. Projects back to epsilon ball after each step
    3. Much stronger than single-step FGSM
    """
    
    def __init__(self, model, epsilon=0.03, alpha=0.01, num_steps=10):
        """
        Initialize PGD attack.
        
        Args:
            model: TensorFlow/Keras model to attack
            epsilon: Maximum perturbation (L∞ bound)
            alpha: Step size per iteration
            num_steps: Number of attack iterations
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        
        print(f"PGD Attack initialized")
        print(f"  Epsilon: {epsilon}")
        print(f"  Alpha (step size): {alpha}")
        print(f"  Steps: {num_steps}")
