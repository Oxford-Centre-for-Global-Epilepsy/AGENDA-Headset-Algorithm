import tensorflow as tf
from tensorflow.keras import layers

# A multi-head classifier for hierarchical classification tasks.
# Each Dense layer corresponds to a level in the hierarchy and produces separate logits.
class HierarchicalClassifierHead(tf.keras.layers.Layer):
    def __init__(self, num_classes=(2, 2, 2), hidden_dim=0, l2_weight=1e-5):
        """
        Args:
            num_classes (tuple): Output class counts per level (e.g., (2, 2, 2))
            hidden_dim (int): Shared hidden layer size. Set < 0 to disable.
            l2_weight (float): L2 regularization for bottleneck (if used)
        """
        super().__init__()
        self.use_bottleneck = hidden_dim > 0

        if self.use_bottleneck:
            self.bottleneck = layers.Dense(
                hidden_dim,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
            )

        self.heads = [layers.Dense(n) for n in num_classes]

    def call(self, x):
        if self.use_bottleneck:
            x = self.bottleneck(x)
        return {
            "level1_logits": self.heads[0](x),
            "level2_logits": self.heads[1](x),
            "level3_logits": self.heads[2](x)
        }


# A single-head classifier for flat (non-hierarchical) classification tasks.
class FlatClassifierHead(tf.keras.layers.Layer):
    def __init__(self, num_classes=4, l2_weight=1e-5):
        """
        Args:
            num_classes (int): Total number of output classes.
            l2_weight (float): L2 regularization strength applied to the Dense layer.
        """
        super().__init__()
        self.classifier = tf.keras.layers.Dense(
            num_classes, 
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
            name="flat_classifier_dense"
        )

    def call(self, x):
        # Output dictionary containing flat logits
        return {
            "logits": self.classifier(x)
        }

def GetClassifierHead(type, num_classes, hidden_dim=0, l2_weight=1e-5):
    """
    Constructs a classifier head based on the specified type.

    Args:
        type (str): Type of classifier head to create. Supported: "flat", "hierarchical".
        num_classes (int or tuple): Number of output classes. 
            - For "flat", this should be an integer.
            - For "hierarchical", this should be a tuple indicating class count per level.
        l2_weight (float): L2 regularization weight for the Dense layers.

    Returns:
        tf.keras.Layer: An instance of FlatClassifierHead or HierarchicalClassifierHead.

    Raises:
        ValueError: If the given type is not supported.
    """
    if type == 'flat':
        return FlatClassifierHead(num_classes=num_classes, l2_weight=l2_weight)
    elif type == 'hierarchical':
        return HierarchicalClassifierHead(num_classes=num_classes, hidden_dim=hidden_dim, l2_weight=l2_weight)
    else:
        raise ValueError(f"Unsupported classifier head type: '{type}'. "
                         f"Expected 'flat' or 'hierarchical'.")


if __name__ == "__main__":
    # Define dummy input tensor: batch of 4 samples, each with 128 features
    dummy_input = tf.random.normal([4, 128])  # [batch_size, feature_dim]

    print("\n=== Testing FlatClassifierHead ===")
    flat_head = FlatClassifierHead(num_classes=4)
    flat_output = flat_head(dummy_input)
    print("Flat classifier output keys:", flat_output.keys())
    print("Flat classifier logits shape:", flat_output["logits"].shape)

    print("\n=== Testing HierarchicalClassifierHead ===")
    hier_head = HierarchicalClassifierHead(num_classes=(2, 2, 2))
    hier_output = hier_head(dummy_input)
    for key, value in hier_output.items():
        print(f"{key} shape:", value.shape)
