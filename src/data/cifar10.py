import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_cifar10(validation_split=0.1):
    (training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.cifar10.load_data()
    
    training_images = training_images / 255.
    testing_images = testing_images / 255.

    training_labels = training_labels.flatten()
    testing_labels = testing_labels.flatten()

    # Validation Split:
    training_images, validation_images, training_labels, validation_labels = train_test_split(
        training_images, training_labels, test_size=validation_split, random_state=111, shuffle=True, stratify=training_labels
    )

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print(f"\nFinal splits:")
    print(f"Training:   {training_images.shape[0]} samples")
    print(f"Validation: {validation_images.shape[0]} samples")
    print(f"Testing:    {testing_images.shape[0]} samples")
    
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


def visualize_samples(images, labels, class_names, n_samples):
    indices = np.random.choice(len(images), n_samples, replace=False)   # Random image position (a list)
    grid_size = int(np.sqrt(n_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx])
        axes[i].set_title(class_names[labels[idx]], fontsize=9)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


# Check whether dataset is balanced
def get_class_distribution(y, class_names):
    print("\nClass Distribution:")
    for i, name in enumerate(class_names):
        count = np.sum(y == i)
        percentage = 100 * count / len(y)
        print(f"{name}: {count} samples ({percentage:.2f}%)")

# Will help answer: is the dataset balanced? Did the train/validation/test split keep class proportions? Did stratification work?


# Quick test:
if __name__ == "__main__":
    (training_images, training_labels), (validation_images, validation_labels), (testing_images, testing_labels), class_names = load_cifar10()

    get_class_distribution(training_labels, class_names)

    visualize_samples(training_images,training_labels,class_names,9)

    print("---------")