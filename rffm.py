import json
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass

from extract_features import create_extractor, rgb_to_y_channel, extract_features_from_image

@dataclass
class RFFMScores:
    """Container for RFFM scores with subscores."""
    overall: float  # Overall RFFM score [0, 1]
    sharpness: float  # Sharpness subscore
    artifacts: float  # Artifacts subscore
    texture: float  # Texture subscore
    intensity: float  # Intensity subscore
    
    def to_dict(self) -> dict:
        """Convert single image scores to dictionary."""
        return {
            'rffm': float(self.overall),
            'rffm_sharpness': float(self.sharpness),
            'rffm_artifacts': float(self.artifacts),
            'rffm_texture': float(self.texture),
            'rffm_intensity': float(self.intensity),
        }
    
    def to_array(self) -> np.ndarray:
        """Get overall RFFM score for a single image."""
        return np.array([
            self.overall,
            self.sharpness,
            self.artifacts,
            self.texture,
            self.intensity
        ])


class RFFM:
    
    def __init__(self):
        with open("config.json") as f:
            self.config = json.load(f)
        self.feature_names = self.config['feature_names']
        self.feature_groups = self.config['feature_groups']
        self.feature_weights = np.array(self.config['feature_weights'])
        self.d_ref = np.array(self.config['d_ref'])
        self.group_weights = self.config['group_weights']
        self.extractor = create_extractor(self.feature_names)
        self.epsilon = 1e-10
        
    
    def get_features(self, features: OrderedDict) -> np.ndarray:
        """Convert extracted features to numpy array in the order of self.feature_names."""
        feature_array = np.zeros(len(self.feature_names))
        for i, feat_name in enumerate(self.feature_names):
            if feat_name not in features:
                raise ValueError(f"Feature '{feat_name}' not found in extracted features.")
            value = features[feat_name]
            if not np.isfinite(value):
                value = 0.0
            feature_array[i] = value
        return feature_array
    
    
    def get_grouped_features(self, values: np.ndarray, group_name: str, features_indices: list) -> np.array:
        features_g = self.feature_groups[group_name]
        values_g = np.zeros(len(features_indices))
        for i, feat_name in enumerate(features_indices):
            index = features_g[feat_name]
            values_g[i] = values[index]
        return values_g
        
        
    def dist(self, deltas: np.ndarray, weights: np.ndarray, d_ref: np.ndarray) -> np.ndarray:
        """
        dist = (1 - Σw·d / Σw·d_ref)
        
        Args:
            deltas: Normalized deltas (n_images, n_features)
            weights: Feature weights (n_features,)
            d_ref: Reference normalization (n_features,)
        
        Returns:
            distance: float
        """
        numerator = np.sum(weights * deltas)
        denominator = np.sum(weights * d_ref)
        distance = (1.0 - numerator / denominator)
        return np.clip(distance, 0, 1)


    def rffm(self, deltas: np.ndarray) -> RFFMScores:
        subscores = {}
        for group_name, feature_indices in self.feature_groups.items():
            if len(feature_indices) == 0:
                subscores[group_name] = 0.0
                continue
            deltas_g = self.get_grouped_features(deltas, group_name, feature_indices)
            weights_g = self.get_grouped_features(self.feature_weights, group_name, feature_indices)
            d_ref_g = self.get_grouped_features(self.d_ref, group_name, feature_indices)
            weights_g = weights_g / weights_g.sum()
            rffm_g = self.dist(deltas_g, weights_g, d_ref_g)
            subscores[group_name] = rffm_g
        overall = 0.0
        for group_name, rffm_g in subscores.items():
            alpha_g = self.group_weights.get(group_name, 0.0)
            overall += alpha_g * rffm_g
        overall = np.clip(overall, 0, 1)
        return RFFMScores(
            overall=overall,
            sharpness=subscores.get('sharpness', 0),
            artifacts=subscores.get('artifacts', 0),
            texture=subscores.get('texture', 0),
            intensity=subscores.get('intensity', 0)
        )


    def score(self, pred: np.ndarray, target: np.ndarray):
        """
        High-level API to compute RFFM scores.
        
        Args:
            pred: Predicted images (n_images, n_features)
            target: Target images (n_images, n_features)
        Returns:
            If formula_type is 'grouped_*': RFFMScores object with subscores
            Otherwise: np.ndarray of overall scores
        """    
        pred_y = rgb_to_y_channel(pred)
        target_y = rgb_to_y_channel(target)
        pred_features = self.get_features(extract_features_from_image(pred_y, self.extractor))
        target_features = self.get_features(extract_features_from_image(target_y, self.extractor))
        deltas = np.abs(target_features - pred_features) / (np.abs(target_features) + self.epsilon)
        return self.rffm(deltas)