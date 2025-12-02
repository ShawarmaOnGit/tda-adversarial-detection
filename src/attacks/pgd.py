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
        
        
        
        
    def generate_single(self, image, label):
       """
       Generate single adversarial example using PGD.
       Returns:
         - adversarial_image: Perturbed image
         - perturbation: Final perturbation
       """
       image_batch = tf.expand_dims(image, 0)
       label_batch = tf.expand_dims(label, 0)
      
       # adversarial example
       adv_batch, pert_batch = self._pgd_batch(image_batch, label_batch)
       return adv_batch[0], pert_batch[0]
  
  
  
  
  
  
  
    def generate_batch(self, images, labels, verbose=True):
       """
       Generate adversarial examples for entire batch using PGD.
       images: Batch of images (N, H, W, C)
       """
       if verbose:
           print(f"\nGenerating PGD adversarial examples (ε={self.epsilon}, steps={self.num_steps})...")
      
       images = tf.constant(images, dtype=tf.float32)
       labels = tf.constant(labels, dtype=tf.int64)
       adversarial_images, perturbations = self._pgd_batch(images, labels, verbose=verbose)
      
       # success rate
       original_predictions = self.model.predict(images, verbose=0).argmax(axis=1)
       adversarial_predictions = self.model.predict(adversarial_images, verbose=0).argmax(axis=1)
      
       original_correct = (original_predictions == labels.numpy()).sum()
       adversarial_correct = (adversarial_predictions == labels.numpy()).sum()
      
       if original_correct > 0:
           success_rate = 1 - (adversarial_correct / original_correct)
       else:
           success_rate = 0
      
       if verbose:
           print(f"\nSuccessfully generated {len(images)} adversarial examples")
           print(f"Original accuracy:     {original_correct / len(images):.2%}")
           print(f"Adversarial accuracy:  {adversarial_correct / len(images):.2%}")
           print(f"Attack success rate:   {success_rate:.2%}")
      
       return adversarial_images.numpy(), perturbations.numpy(), success_rate
