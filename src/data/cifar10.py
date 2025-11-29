import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_cifar10(validation_split=0.1):
    (training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.cifar10.load_data()
    
    training_images = training_images / 255.
    testing_images = testing_images / 255.

    # Validation Split:
    training_images, validation_images, training_labels, validation_labels = train_test_split(
        training_images, training_labels, test_size=validation_split, random_state=111, shuffle=True, stratify=training_labels
    )

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print(f"\nFinal splits:")
    print(f"  Training:   {training_images.shape[0]} samples")
    print(f"  Validation: {validation_images.shape[0]} samples")
    print(f"  Testing:    {testing_images.shape[0]} samples")
    
    return ((training_images, training_labels), (validation_images, validation_labels), (testing_images, testing_labels), class_names)


"""
CIFAR-10
 ├── Training set (50,000)
 │     ├── Training split (45,000)
 │     └── Validation split (5,000)
 │
 └── Testing set (10,000)
        └── After the model is fully trained
"""


