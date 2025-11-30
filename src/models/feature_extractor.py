# Extracts high-level features from CIFAR-10 using pretrained models (ResNet50 by default):

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class FeatureExtractor:
    models = {
        "resnet50": {
            "model": tf.keras.applications.ResNet50,
            "preprocess": tf.keras.applications.resnet.preprocess_input,
            "dim": 2048
        },
        "resnet101": {
            "model": tf.keras.applications.ResNet101,
            "preprocess": tf.keras.applications.resnet.preprocess_input,
            "dim": 2048
        },
        "vgg16": {
            "model": tf.keras.applications.VGG16,
            "preprocess": tf.keras.applications.vgg16.preprocess_input,
            "dim": 512
        }
    }

    def __init__(self, model_name="resnet50"):
        model_parameters = self.models[model_name]
        self.model_name = model_name
        self.feature_dim = model_parameters["dim"]
        self.preprocess_fn = model_parameters["preprocess"]

        self.model = model_parameters["model"](    # tf.keras.applications.ResNet50 by default
            include_top=False,                     # Since we're doing feature extraction, not classification
            weights="imagenet",
            pooling="avg"
        )

        print(f"\nModel: {model_name}")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Total parameters: {self.model.count_params():,}")



    def extract_features(self, images, batch_size=128, resize_to=224, verbose=True):
        all_features = []
        iterator = range(0, len(images), batch_size)

        if verbose:
            iterator = tqdm(iterator, desc="\nFeature extraction in progress... ", unit="batch")

        for i in iterator:
            batch = images[i:i+batch_size]
            batch = tf.image.resize(batch, (resize_to, resize_to))
            batch = self.preprocess_fn(batch)
            feature_vectors = self.model.predict(batch, verbose=0)   # Run the CNN for feature extraction (batch_size, feature_dim)
            all_features.append(feature_vectors)
        features = np.vstack(all_features)   # Combine to one big array, (N, feature_dim) s.t. N = total images

        if verbose:
            print(f"\nExtracted features from {features.shape[0]:,} images")
            print(f"Feature shape: {features.shape}")
            print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")

        return features



    def save_features(self, features, labels, filepath):
        np.savez_compressed(filepath, features=features, labels=labels)   # Load next time with 'data = np.load("file.npz")'
        print(f"Features saved to {filepath}")


    
    @staticmethod
    def load_features(filepath, verbose=True):
        data = np.load(filepath)
        if verbose:
            print(f"Loaded {filepath} — features: {data['features'].shape}, labels: {data['labels'].shape}")

        return data["features"], data["labels"]
    


class DimensionalityReducer:
    """
    Goal: Reduce feature dimensionality using PCA.
    High-dimensional features (2048-dim) are computationally expensive for TDA. 
    So, we use PCA to reduce dimensionality while also preserving most variance.
    """
    def __init__(self, n_components=50, random_state=42):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.fitted = False
    

    def fit_transform(self, features, verbose=True):
        if verbose:
            print("\nApplying PCA...")
            print(f"Original feature dimension: {features.shape[1]}")
            print(f"Target dimension: {self.n_components}")
        reduced = self.pca.fit_transform(features)
        self.fitted = True
        return reduced


    def transform(self, features):
        if not self.fitted:
            raise ValueError("Call fit_transform() before transform().")
        return self.pca.transform(features)
    

    def plot_variance_explained(self, save_path=None):
        if not self.fitted:
            raise ValueError("PCA must be fitted before plotting.")

        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        num_components = len(explained_variance)

        # Find components at each threshold to get an understanding whether or not correct number was chosen
        threshold_90 = np.searchsorted(cumulative_variance, 0.90) + 1
        threshold_95 = np.searchsorted(cumulative_variance, 0.95) + 1
        threshold_99 = np.searchsorted(cumulative_variance, 0.99) + 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Individual variance
        ax1.bar(range(1, len(explained_variance) + 1), explained_variance, 
                color='steelblue', alpha=0.7, edgecolor='navy')
        ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Variance Explained', fontsize=12, fontweight='bold')
        ax1.set_title('Variance Explained by Each Component', fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3, linestyle='--')
        
        # Cumulative variance
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                 'o-', color='darkgreen', linewidth=2, markersize=4)
        
        # Each threshold
        # 90%
        ax2.axhline(y=0.90, color='red', linestyle='--', linewidth=1)
        ax2.text(num_components * 0.85, 0.91, f"90% ({threshold_90} comps)", color='red')

        # 95%
        ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=1)
        ax2.text(num_components * 0.85, 0.96, f"95% ({threshold_95} comps)", color='red')

        # 99%
        ax2.axhline(y=0.99, color='red', linestyle='--', linewidth=1)
        ax2.text(num_components * 0.85, 1.00, f"99% ({threshold_99} comps)", color='red')

        ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Variance Explained', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved variance plot to: {save_path}")
        
        plt.show()


    # Get indices of top N principal components by variance explained:
    def get_top_components(self, n=3):
        if not self.fitted:
            raise ValueError("PCA must be fitted first.")
        variances = self.pca.explained_variance_ratio_
        indices = np.argsort(variances)[::-1][:n]
        
        return indices, variances[indices]
    

# Quick test:
if __name__ == "__main__":
    print("Testing Feature Extractor Module...\n")
    
    import sys
    sys.path.append('../../')
    from src.data.cifar10 import load_cifar10
    
    (training_images, training_labels), _, _, class_names = load_cifar10(validation_split=0.1)
    
    # Only 500 samples for this test:
    test_images = training_images[:500]
    test_labels = training_labels[:500]

    print(f"\nTest data: {test_images.shape}\n")
    
    # Feature extraction
    extractor = FeatureExtractor(model_name='resnet50')
    features = extractor.extract_features(test_images, batch_size=32)
    
    # PCA reduction
    reducer = DimensionalityReducer(n_components=50)
    reduced_features = reducer.fit_transform(features)

    # PCA variance explained plot
    print("\nTesting PCA Variance Plot...\n")
    reducer.plot_variance_explained()
    
    # Test save/load
    extractor.save_features(features, test_labels, '../../results/features/test_features.npz')
    loaded_features, loaded_labels = FeatureExtractor.load_features(
        '../../results/features/test_features.npz'
    )
    
    print(f"\nOriginal features: {features.shape}")
    print(f"Reduced features: {reduced_features.shape}")
    print(f"Loaded features: {loaded_features.shape}")