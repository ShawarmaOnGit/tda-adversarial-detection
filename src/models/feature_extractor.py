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

        print(f"Model: {model_name.upper()} with (feature_dim={self.feature_dim})")
