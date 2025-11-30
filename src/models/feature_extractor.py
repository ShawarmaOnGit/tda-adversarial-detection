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
            "preprocess": tf.keras.applications.resnet50.preprocess_input,
            "dim": 2048
        },
        "resnet101": {
            "model": tf.keras.applications.ResNet101,
            "preprocess": tf.keras.applications.resnet101.preprocess_input,
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

        print(f"Model: {model_name}")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Total parameters: {self.model.count_params():,}")



    def extract_features(self, images, batch_size=128, resize_to=224, verbose=True):
        all_features = []
        iterator = range(0, len(images), batch_size)

        if verbose:
            iterator = tqdm(iterator, desc="Feature extraction in progress... ", unit="batch")

        for i in iterator:
            batch = images[i:i+batch_size]
            batch = tf.image.resize(batch, (resize_to, resize_to))
            batch = self.preprocess_fn(batch)
            feature_vectors = self.model.predict(batch, verbose=0)   # Run the CNN for feature extraction (batch_size, feature_dim)
            all_features.append(feature_vectors)
        features = np.vstack(all_features)   # Combine to one big array, (N, feature_dim) s.t. N = total images

        if verbose:
            print(f"Extracted features from {features.shape[0]:,} images")
            print(f"Feature shape: {features.shape}")
            print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")

        return features
