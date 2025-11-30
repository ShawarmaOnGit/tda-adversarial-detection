"""
Fast Gradient Sign Method (FGSM) Attack:
Generates adversarial examples by adding small perturbations in the direction
of the gradient of the loss function.
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm

class FGSMAttack:
    def __init__(self, model, epsilon=0.03):
        """
        We want the input to change as little as possible --> limit input pixel changes to epsilon
        We want to change the output as much as possible --> maximize the loss
        """
        self.model = model
        self.epsilon = epsilon
        print(f"FGSM Attack successfully initialized")
        print(f"Epsilon: {epsilon}")