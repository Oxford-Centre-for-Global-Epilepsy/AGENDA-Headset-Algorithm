import tensorflow as tf
import numpy as np
import csv

class StructureAwareLoss(tf.keras.losses.Loss):
    def __init__(self, label_config, temperature=1, distance_csv_path=None, class_histogram=None):
        super().__init__()

        # Load label configuration
        self.label_map = label_config['label_map']
        self.inverse_label_map = label_config['inverse_label_map']
        self.label_prior = label_config['label_prior']

        # Construct an internal label map for tensor indexing
        self.label_map_internal = {key: i for i, key in enumerate(self.label_map.keys())}


        # Construct the soft class vector based on the label prior
        self.soft_class_vector = self.construct_soft_class(self.label_prior)

        # Load or use default distance matrix
        if distance_csv_path:
            with open(distance_csv_path, 'r') as f:
                reader = csv.reader(f)
                distance_matrix = np.array([[float(x) for x in row] for row in reader])
        else:
            distance_matrix = np.array([
                [0.0, 3.0, 3.0, 3.0],
                [3.0, 0.0, 2.0, 2.0],
                [3.0, 2.0, 0.0, 1.0],
                [3.0, 2.0, 1.0, 0.0],
            ])

        # Compute the soft target vectors from input
        self.distance_matrix = self.normalize_distance_matrix(distance_matrix, temperature)
        self.soft_target_vectors = self.precompute_target_vector()

        # Use unnormalized class histogram if not provided
        if class_histogram is None:
            class_histogram = {
                "neurotypical": 1,
                "epileptic": 1,
                "focal": 1,
                "generalized": 1,
                "left": 1,
                "right": 1
            }
 
        # Normalize class weights based on the histogram
        self.class_weights = self.normalize_class_weights(class_histogram)

        # Convert class weights to tensor
        self.tensor_conversion()

    def call(self, y_true, logits):
        """
        Compute the structure-aware loss using precomputed soft targets and class weights.

        Args:
            y_true (tf.Tensor): shape [B, 3], hierarchical labels
            logits (tf.Tensor): shape [B, C], raw model outputs

        Returns:
            tf.Tensor: scalar loss value
        """

        # Step 1: Squeeze y_true to remove last dimension if present
        y_true = tf.squeeze(y_true, axis=-1)
        y_true = tf.cast(y_true, tf.int32)

        # Step 2: Get soft target distributions for each sample
        soft_targets = tf.gather(self.soft_target_tensor, y_true)  # shape [B, C]

        tf.print(soft_targets)
        tf.print(logits)
        
        # Step 3: Compute log-softmax of logits
        log_probs = tf.nn.log_softmax(logits, axis=1)  # shape [B, C]

        # Step 4: KL divergence between soft targets and predicted log-probs
        kl_div = -tf.reduce_sum(soft_targets * log_probs, axis=1)  # shape [B]

        # Step 5: Apply sample-wise class weights
        sample_weights = tf.gather(self.class_weights_tensor, y_true)  # shape [B]
        weighted_loss = kl_div * sample_weights  # shape [B]

        # Step 6: Return batch mean
        return tf.reduce_mean(weighted_loss)

    def normalize_class_weights(self, class_histogram):
        """
        Compute normalized inverse-frequency weights per input class label.

        Args:
            class_histogram (dict): {class_label (str): count (int)}.

        Returns:
            dict: {class_label (str): normalized weight (float)}.
        """
        # Step 1: Compute raw inverse-frequency weights
        raw_weights = {}
        for label, count in class_histogram.items():
            if count > 0:
                raw_weights[label] = 1.0 / count
            else:
                raw_weights[label] = 0.0  # Avoid division by zero

        # Step 2: Normalize weights so they sum to 1
        total_weight = sum(raw_weights.values())
        if total_weight == 0:
            raise ValueError("All class frequencies are zero — cannot normalize.")

        normalized_weights = {
            label: weight / total_weight for label, weight in raw_weights.items()
        }

        return normalized_weights  # dict[str, float], length = 6

    def normalize_distance_matrix(self, distance_matrix, temperature):
        Z = np.max(distance_matrix) / temperature
        distance_matrix_normalized = tf.convert_to_tensor(distance_matrix / Z, dtype=tf.float32)

        return distance_matrix_normalized

    def construct_soft_class(self, label_prior):
        """
        Construct the soft class vector based on the label prior.
        label_prior: dictionary of binary class prior at each decision point
            e.g., {'epileptic': 0.5, 'focal': 0.5, 'right': 0.5}
        """

        # Construct the soft class vector from the label prior
        soft_class_vector = {
            "neurotypical": tf.convert_to_tensor([1.0, 0.0, 0.0, 0.0], dtype=tf.float32),
            "epileptic": tf.convert_to_tensor([0.0, 
                                               1.0-label_prior['focal'], 
                                               label_prior['focal']*(1.0-label_prior['right']), 
                                               label_prior['focal']*label_prior['right']], dtype=tf.float32),
            "focal": tf.convert_to_tensor([0.0, 
                                           0.0, 
                                           1.0-label_prior['right'], 
                                           label_prior['right']], dtype=tf.float32),
            "generalized": tf.convert_to_tensor([0.0, 1.0, 0.0, 0.0], dtype=tf.float32),
            "left": tf.convert_to_tensor([0.0, 0.0, 1.0, 0.0], dtype=tf.float32),
            "right": tf.convert_to_tensor([0.0, 0.0, 0.0, 1.0], dtype=tf.float32)
        }

        return soft_class_vector
    
    def precompute_target_vector(self):
        """
        Precompute the target vector for the soft class vector.
        """
        bolzmann_matrix = tf.exp(-self.distance_matrix)

        # Create a new dict to hold the soft target vectors
        soft_target_vectors = {}

        # Precompute the target vector from bolzmann matrix
        for key, vector in self.soft_class_vector.items():
            # Compute the joint soft target distribution (in matrix form)
            soft_target_matrix = bolzmann_matrix * tf.expand_dims(vector, axis=1)
            soft_target_matrix = soft_target_matrix / tf.reduce_sum(soft_target_matrix)

            # Collapse the matrix into soft target vector
            soft_target_vector = tf.reduce_sum(soft_target_matrix, axis=0)

            soft_target_vectors[key] = soft_target_vector

        return soft_target_vectors
            
    def tensor_conversion(self):
        """
        Convert all pre-computed dictionaries into tensor form.
        """

        # Convert dict to tensor: shape [C(classification), C(label output)]
        self.soft_target_tensor = tf.stack(
            [self.soft_target_vectors[label] for label in sorted(self.label_map_internal, key=self.label_map_internal.get)],
            axis=0
        )

        # Convert class weights dict to tensor: shape [C(input class)]
        self.class_weights_tensor = tf.convert_to_tensor(
            [
                self.class_weights.get(label, 0.0)  # fallback to 0.0 if label is missing
                for label in sorted(self.label_map_internal, key=self.label_map_internal.get)
            ],
            dtype=tf.float32
        )

    def get_soft_class_vector(self, label=None):
        """
        Get the soft class vector for a specific label.
        If label is None, return the full soft class vector dict.
        """
        if label is None:
            return self.soft_class_vector
        else:
            if label in self.soft_class_vector:
                return self.soft_class_vector[label]
            else:
                raise ValueError(f"Label '{label}' not found in soft class vector.")

    def get_target_vector(self, label=None):
        """
        Get the target vector for a specific label.
        If label is None, return the full soft target tensor.
        """
        if label is None:
            return self.soft_target_tensor
        else:
            index = self.label_map_internal.get(label)
            if index is not None:
                return self.soft_target_tensor[index]
            else:
                raise ValueError(f"Label '{label}' not found in label map.")

def masked_cross_entropy(logits, targets, mask, class_weights=None):
    """
    Computes masked cross-entropy loss for binary/multiclass classification.
    
    Args:
        logits: Tensor of shape [B, C]
        targets: Tensor of shape [B], integer class labels
        mask: Boolean Tensor of shape [B]
        class_weights: Optional Tensor of shape [C], weights for each class
    
    Returns:
        Scalar tensor representing the masked cross-entropy loss
    """
    mask = tf.cast(mask, dtype=tf.bool)
    if tf.reduce_sum(tf.cast(mask, tf.float32)) == 0.0:
        return tf.constant(0.0)

    logits = tf.boolean_mask(logits, mask)
    targets = tf.boolean_mask(targets, mask)

    if class_weights is not None:
        weights = tf.gather(class_weights, targets)
        loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
        return tf.reduce_mean(loss * weights)
    else:
        loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
        return tf.reduce_mean(loss)

class HierarchicalLoss(tf.keras.losses.Loss):
    def __init__(self, weights=(1.0, 1.0, 1.0), level1_weights=None, level2_weights=None, level3_weights=None):
        super().__init__()
        self.weights = weights
        self.level1_weights = level1_weights
        self.level2_weights = level2_weights
        self.level3_weights = level3_weights

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: A dict with:
                'targets': Tensor of shape [B, 3]
                'label_mask': Tensor of shape [B, 3] (1 = valid, 0 = invalid)
            y_pred: A dict with:
                'level1_logits': [B, 2],
                'level2_logits': [B, 2],
                'level3_logits': [B, 2]
        Returns:
            Scalar tensor loss
        """
        targets = y_true['targets']
        label_mask = tf.cast(y_true['label_mask'], dtype=tf.bool)

        # Level 1: already binary
        loss1 = masked_cross_entropy(
            y_pred["level1_logits"],
            targets[:, 0],
            label_mask[:, 0],
            class_weights=self.level1_weights
        )

        # Level 2: remap 2=focal → 0, 3=generalized → 1
        level2_target = tf.where(targets[:, 1] == 3, 1, 0)
        loss2 = masked_cross_entropy(
            y_pred["level2_logits"],
            level2_target,
            label_mask[:, 1],
            class_weights=self.level2_weights
        )

        # Level 3: remap 4=left → 0, 5=right → 1
        level3_target = tf.where(targets[:, 2] == 5, 1, 0)
        loss3 = masked_cross_entropy(
            y_pred["level3_logits"],
            level3_target,
            label_mask[:, 2],
            class_weights=self.level3_weights
        )

        return self.weights[0] * loss1 + self.weights[1] * loss2 + self.weights[2] * loss3

if __name__ == "__main__":
    # Dummy label config
    label_config = {
        "label_map": {
            "neurotypical": 0,
            "epileptic": 1,
            "focal": 2,
            "generalized": 3,
            "left": 4,
            "right": 5
        },
        "inverse_label_map": {
            0: "neurotypical",
            1: "epileptic",
            2: "focal",
            3: "generalized",
            4: "left",
            5: "right"
        },
        "label_prior": {
            "epileptic": 0.5,
            "focal": 0.5,
            "right": 0.5
        }
    }

    # Instantiate the loss
    loss_fn = StructureAwareLoss(label_config=label_config, temperature=5)

    print("=== Full soft class vector ===")
    print(loss_fn.get_soft_class_vector())

    print("\n=== Full soft target tensor ===")
    print(loss_fn.get_target_vector())

    # Query individual labels
    labels_to_test = ["neurotypical", "epileptic", "focal", "right"]

    for label in labels_to_test:
        print(f"\n--- For label: {label} ---")
        print("Soft class vector:")
        print(loss_fn.get_soft_class_vector(label))
        print("Target vector:")
        print(loss_fn.get_target_vector(label))
