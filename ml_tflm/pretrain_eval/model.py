import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from ml_tflm.pre_training.model_pretrain import build_vanilla_eegnet, build_projector

class AttentionPooling(layers.Layer):
    def __init__(self, input_dim, hidden_dim=32, activation='tanh', temperature=1.0, dropout_rate=0.5, l2_weight=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(hidden_dim, activation=activation,
                                   kernel_regularizer=regularizers.l2(l2_weight))
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(1, use_bias=False,
                                   kernel_regularizer=regularizers.l2(l2_weight))
        self.temperature = temperature

    def call(self, inputs, mask=None, return_weights=False, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        scores = self.dense2(x)
        scores = tf.squeeze(scores, axis=-1)

        if mask is not None:
            large_neg = -1e9
            scores = tf.where(mask, scores, large_neg * tf.ones_like(scores))

        scores /= self.temperature
        weights = tf.nn.softmax(scores, axis=-1)

        weights_expanded = tf.expand_dims(weights, axis=-1)
        pooled = tf.reduce_sum(inputs * weights_expanded, axis=1)

        if return_weights:
            return pooled, weights
        return pooled

class EEGNetFlatClassifier(Model):
    def __init__(self,
                 eegnet,
                 projector,
                 pooling_args=None,
                 num_classes=4,
                 classifier_dropout_rate=0.5,
                 l2_weight=1e-3):
        super().__init__()
        self.eegnet = eegnet
        self.projector = projector
        self.eegnet.trainable = False
        self.projector.trainable = False

        pooling_args = pooling_args or {}
        if 'dropout_rate' not in pooling_args:
            pooling_args['dropout_rate'] = classifier_dropout_rate
        if 'l2_weight' not in pooling_args:
            pooling_args['l2_weight'] = l2_weight

        dummy_input = tf.zeros([1, 21, 128, 1])
        feature_dim = self.eegnet(dummy_input, training=False).shape[-1]

        self.pool = AttentionPooling(input_dim=feature_dim, **pooling_args)

        # Define classifier as a Sequential model to expose as self.classifier
        self.classifier = tf.keras.Sequential([
            layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(l2_weight)),
            layers.Dropout(classifier_dropout_rate),
            layers.Dense(num_classes, kernel_regularizer=regularizers.l2(l2_weight))
        ])

    def call(self, x, attention_mask=None, return_attn_weights=False, return_features=False, training=False):
        B, E, C, T = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [B * E, C, T, 1])
        epoch_features = self.eegnet(x, training=training)
        epoch_projection = self.projector(epoch_features, training=training)
        D = tf.shape(epoch_projection)[-1]
        epoch_projection = tf.reshape(epoch_projection, [B, E, D])

        pooled, attn_weights = self.pool(epoch_projection, mask=attention_mask, return_weights=True, training=training)
        logits = self.classifier(pooled, training=training)

        out = {"logits": logits}
        if return_attn_weights:
            out["attention_weights"] = attn_weights
        if return_features:
            out["features"] = pooled

        return out
    
def get_classifier(model_dict):
    pooling_args = {}

    return EEGNetFlatClassifier(model_dict["feature_extractor"], model_dict["projector"], pooling_args=pooling_args)

def configure_model(feature_args, projector_args):
    """
    Creates and returns the feature extractor and projector models.

    Args:
        feature_args (dict): Arguments for building the feature extractor.
        projector_args (dict): Arguments for building the projector.

    Returns:
        dict: A dictionary with 'feature_extractor' and 'projector' models.
    """
    feature_extractor = build_vanilla_eegnet(**feature_args)
    projector = build_projector(**projector_args)
    
    return {
        "feature_extractor": feature_extractor,
        "projector": projector
    }

if __name__ == "__main__":
    model_dict = configure_model({"bottleneck_dim": 16}, {"input_dim": 16})
    classifier = get_classifier(model_dict)

    print(0)