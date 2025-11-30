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


    def _fgsm_batch(self, images, labels):
        """
        Does the math for the attack
        images = batch of images (TensorFlow tensor)
        labels = true labels (TensorFlow tensor)
        """
        images = tf.Variable(images)        # So TF can take gradients with respect to
        with tf.GradientTape() as tape:
            tape.watch(images)              # If GT is security camera, this is telling which to follow
            predictions = self.model(images, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        gradients = tape.gradient(loss, images)    # Derivative of loss with respect to images
        
        perturbations = self.epsilon * tf.sign(gradients)
        adversarial_images = images + perturbations
        adversarial_images = tf.clip_by_value(adversarial_images, 0, 1)
        
        return adversarial_images, perturbations


    def generate_single(self, image, label):
        """
        Attacks ONE image
        adv_batch shape:  (1, H, W, C) --> One image
        pert_batch shape: (1, H, W, C)
        """
        image_batch = tf.expand_dims(image, 0)     # TF expects a batch of images. (H, W, C) --> (1, H, W, C) AND bcomes a TF tensor
        label_batch = tf.expand_dims(label, 0)     # Same as above, now for labels
        adv_batch, pert_batch = self._fgsm_batch(image_batch, label_batch)
        return adv_batch[0], pert_batch[0]         # The first element inside the batch, which is the only image
    
        # Extracting the first and only image from the batch makes (1, H, W, C) --> (H, W, C)


    def generate_batch(self, images, labels, verbose=True):
        """
        Generate adversarial examples for entire batch.
        """
        if verbose:
            print(f"\nGenerating FGSM adversarial examples (ε={self.epsilon})...")
        
        images = tf.constant(images, dtype=tf.float32)     # Unlike tf.expand_dims, we must force this to a tensor
        labels = tf.constant(labels, dtype=tf.int32)
        adversarial_images, perturbations = self._fgsm_batch(images, labels)

        original_predictions = self.model.predict(images, verbose=0).argmax(axis=1)
        adversarial_predictions = self.model.predict(adversarial_images, verbose=0).argmax(axis=1)
        original_correct = np.sum(original_predictions == labels.numpy())
        adversarial_correct = np.sum(adversarial_predictions == labels.numpy())    # After the attack

        if original_correct > 0:
            success_rate = (original_correct - adversarial_correct) / original_correct
        else:
            success_rate = 0

        if verbose:
            print(f"\nGenerated {len(images)} adversarial examples")
            print(f"Original accuracy:     {original_correct / len(images):.2%}")
            print(f"Adversarial accuracy:  {adversarial_correct / len(images):.2%}")
            print(f"Attack success rate:   {success_rate:.2%}")

        return adversarial_images.numpy(), perturbations.numpy(), success_rate