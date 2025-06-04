import torch
import torch.nn as nn
from ml.models.feature_extractor.eegnet import EEGNet
from ml.models.pooling import mean, attention, transformer

import torch
import torch.nn as nn

class HierarchicalClassifier(nn.Module):
    def __init__(self, feature_extractor, pooling_layer, classifier):
        super(HierarchicalClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.pooling_layer = pooling_layer
        self.classifier = classifier

    def forward(self, x, attention_mask=None, return_attn_weights=False, return_features=False):
        # Pass through the feature extractor
        features = self.feature_extractor(x)
        
        # Pass through the pooling layer
        pooled_features = self.pooling_layer(features)
        
        # Pass through the classifier
        out = self.classifier(pooled_features)
        
        if return_attn_weights:
            out["attention_weights"] = attention_mask  # Just an example
        if return_features:
            out["features"] = pooled_features
        
        return out
        
class HierarchicalClassifier(nn.Module):
    def __init__(self,
                 feature_extractor,
                 pooling_layer,
                 classifier
                 ):
        """
        Args:
            eegnet_args (dict): kwargs to initialize EEGNet
            pooling_type (str): "mean", "attention", or "transformer"
            pooling_args (dict): kwargs for the pooling layer
            hidden_dim (int): hidden size for classification layers
            num_classes (tuple): number of classes at each hierarchy level
        """
        super().__init__()
        self.eegnet = EEGNet(**eegnet_args)
        feature_dim = self.eegnet.feature_dim
        pooling_args = pooling_args or {}

        if pooling_type == "mean":
            self.pool = MeanPooling()
        elif pooling_type == "attention":
            self.pool = AttentionPooling(input_dim=feature_dim, **pooling_args)
        elif pooling_type == "transformer":
            self.pool = TransformerPooling(input_dim=feature_dim, **pooling_args)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

        self.classifier1 = nn.Linear(feature_dim, num_classes[0])
        self.classifier2 = nn.Linear(feature_dim, num_classes[1])
        self.classifier3 = nn.Linear(feature_dim, num_classes[2])

    def forward(self, x, attention_mask=None, return_attn_weights=False, return_features=False):
        """
        Args:
            x: [B, E, C, T] - batch of recordings
            attention_mask: [B, E] - mask for valid epochs
            return_attn_weights (bool): if True, also returns attention weights
            return_features (bool): if True, also returns the pooled feature vector

        Returns:
            Dict containing logits for each level, and optionally features/attention weights.
        """
        B, E, C, T = x.shape
        x = x.view(B * E, 1, C, T)  # Flatten to [B*E, 1, C, T]
        epoch_features = self.eegnet(x)  # [B*E, D]
        D = epoch_features.shape[-1]
        epoch_features = epoch_features.view(B, E, D)  # [B, E, D]

        # Check if using Attention pooling - if so, set so that attention can be returned
        attn_weights = None
        if isinstance(self.pool, AttentionPooling):
            pooled, attn_weights = self.pool(epoch_features, mask=attention_mask, return_weights=True)
        else:
            pooled = self.pool(epoch_features, mask=attention_mask)

        out = {
            "level1_logits": self.classifier1(pooled),
            "level2_logits": self.classifier2(pooled),
            "level3_logits": self.classifier3(pooled)
        }

        if return_attn_weights:
            out["attention_weights"] = attn_weights
        if return_features:
            out["features"] = pooled  # [B, D]

        return out