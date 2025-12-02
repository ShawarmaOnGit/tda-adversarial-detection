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
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        
        print(f"PGD Attack initialized")
        print(f"Epsilon: {epsilon}")
        print(f"Alpha (step size): {alpha}")
        print(f"Steps: {num_steps}")
        
        
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
   
   
    def _pgd_batch(self, images, labels, verbose=False):
       """
       Internal PGD implementation.
      
       Args:
           images: Batch of images (TensorFlow tensor)
           labels: True labels (TensorFlow tensor)
           verbose: Show iteration progress
          
       Returns:
           adversarial_images: Perturbed images
           perturbations: Final perturbations
       """
       
       perturbation = tf.random.uniform(shape=images.shape, minval=-self.epsilon, maxval=self.epsilon, dtype=tf.float32)
      
       # Start with perturbed images
       adv_images = tf.clip_by_value(images + perturbation, 0, 1)
      
       # Iterative attack
       iterator = range(self.num_steps)
       if verbose:
           iterator = tqdm(iterator, desc="PGD iterations", leave=False)
      
       for step in iterator:
           adv_images_var = tf.Variable(adv_images)
           with tf.GradientTape() as tape:
               tape.watch(adv_images_var)
               # Forward pass
               predictions = self.model(adv_images_var, training=False)
               # Calculate loss
               loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
          
           gradients = tape.gradient(loss, adv_images_var)
           adv_images = adv_images_var + self.alpha * tf.sign(gradients)
          
           perturbation = adv_images - images
           perturbation = tf.clip_by_value(perturbation, -self.epsilon, self.epsilon)
           adv_images = images + perturbation
           adv_images = tf.clip_by_value(adv_images, 0, 1)
       final_perturbation = adv_images - images
      
       return adv_images, final_perturbation
  

def generate_pgd_dataset(model, images, labels, epsilon=0.03, alpha=0.01, num_steps=10, batch_size=128, verbose=True):
    """
    Convenience function to generate PGD adversarial dataset.
    """
    attacker = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_steps=num_steps)
    
    all_adversarial = []
    total_success = 0
    n_batches = 0
    if verbose:
        iterator = tqdm(range(0, len(images), batch_size), desc="Generating PGD attacks")
    else:
        iterator = range(0, len(images), batch_size)
    
    for i in iterator:
        batch_images = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        adv_batch, _, success = attacker.generate_batch(batch_images, batch_labels, verbose=False)
        
        all_adversarial.append(adv_batch)
        total_success += success
        n_batches += 1
    
    adversarial_images = np.vstack(all_adversarial)
    avg_success_rate = total_success / n_batches
    
    if verbose:
        print(f"\nPGD dataset complete")
        print(f"Total samples: {len(adversarial_images)}")
        print(f"Average attack success rate: {avg_success_rate:.2%}")
    
    return adversarial_images, avg_success_rate
   
   
# Quick test
if __name__ == "__main__":
   print("\nTesting PGD Attack...\n")
   import sys
   sys.path.append('../../')
   from src.data.cifar10 import load_cifar10
  
   # Load a dataset
   (train_images, train_labels), _, _, class_names = load_cifar10(validation_split=0.1)
   test_images = train_images[:100]
   test_labels = train_labels[:100]
   print(f"Test data: {test_images.shape}\n")

   model = tf.keras.applications.ResNet50(
       weights='imagenet',
       include_top=True
   )
   print("ResNet50 loaded successfully\n")
  
   # Resize images
   print("Resizing images to 224x224...")
   test_images_resized = tf.image.resize(test_images, (224, 224)).numpy()
   print(f"Resized to: {test_images_resized.shape}\n")
  
   # PGD attack
   attacker = PGDAttack(model, epsilon=0.03, alpha=0.01, num_steps=10)
  
  
   # Test single image
   print("\n[Test 1] Single image attack...")
   adv_image, perturbation = attacker.generate_single(
       test_images_resized[0],
       test_labels[0]
   )
   print(f"Generated adversarial example")
   print(f"Perturbation range: [{perturbation.numpy().min():.4f}, {perturbation.numpy().max():.4f}]")
  
   # Test batch
   print("\n[Test 2] Batch attack (10 images)...")
   adv_batch, pert_batch, success_rate = attacker.generate_batch(
       test_images_resized[:10],
       test_labels[:10],
       verbose=True
   )
  
   print("All tests completed successfully")