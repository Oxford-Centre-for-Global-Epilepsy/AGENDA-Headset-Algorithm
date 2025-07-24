import tensorflow as tf
import numpy as np
import csv

class StructureAwareLoss(tf.keras.losses.Loss):
    def __init__(self, label_config, clip_value=None, temperature=1, distance_csv_path=None, class_histogram=None, dist_2_2=None, dist_1_2=None):
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
        elif dist_1_2 and dist_2_2:
            distance_matrix = np.array([
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, dist_1_2, dist_1_2],
                [1.0, dist_1_2, 0.0, dist_2_2],
                [1.0, dist_1_2, dist_2_2, 0.0],
            ])
        else:
            distance_matrix = np.array([
                [0.0, 3.0, 3.0, 3.0],
                [3.0, 0.0, 2.0, 2.0],
                [3.0, 2.0, 0.0, 1.0],
                [3.0, 2.0, 1.0, 0.0],
            ])

        # Compute the soft target vectors from input
        self.temperature = temperature
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

        self.clip_value = tf.abs(clip_value)

    def call(self, y_true, output):
        """
        Compute the structure-aware loss using precomputed soft targets and class weights.

        Args:
            y_true (dict): {
                'targets': tf.Tensor of shape [B, 3], hierarchical labels
            }
            output (dict): contains 'logits': tf.Tensor of shape [B, C], raw model outputs

        Returns:
            tf.Tensor: scalar loss value
        """
        y_true = y_true["targets"]
        if len(y_true.shape) == 2 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        y_true = tf.cast(y_true, tf.int32)

        logits = output["logits"]

        # Step 2: Get soft target distributions for each sample
        soft_targets = tf.gather(self.soft_target_tensor, y_true)  # shape [B, C]

        # Step 3: Compute log-softmax of logits
        if self.clip_value is not None:
            logits = tf.clip_by_value(logits, clip_value_min=-self.clip_value, clip_value_max=self.clip_value) # Clip logits to preserve stability at confident predictions
        log_probs = tf.nn.log_softmax(logits, axis=1)  # shape [B, C]

        '''
        tf.print("Soft Target Vectors:", soft_targets, summarize=-1)
        probs = tf.nn.softmax(logits, axis=1)
        tf.print("Softmax probabilities:", probs, summarize=-1)
        '''
        
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
        distance_matrix = tf.convert_to_tensor(distance_matrix, dtype=tf.float32)
        return distance_matrix / temperature

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
            
    def anneal_temperature(self, new_temperature):
        """
        Lowers the temperature and recomputes the distance matrix, soft target vectors, and tensors.
        
        Args:
            new_temperature (float): New temperature value (must be > 0).
        """
        if new_temperature <= 0:
            raise ValueError("Temperature must be greater than 0.")

        # Update and normalize new distance matrix
        self.distance_matrix = self.normalize_distance_matrix(self.distance_matrix.numpy(), new_temperature)

        # Recompute soft target vectors
        self.soft_target_vectors = self.precompute_target_vector()

        # Convert to tensor form again
        self.tensor_conversion()

        # Update the memorized temperature value
        self.temperature = new_temperature

        print(f"[Anneal] Updated temperature to {new_temperature:.4f} and recomputed target vectors.")

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
    mask_sum = tf.reduce_sum(tf.cast(mask, tf.float32))

    def compute_loss():
        masked_logits = tf.boolean_mask(logits, mask)
        masked_targets = tf.boolean_mask(targets, mask)

        loss = tf.keras.losses.sparse_categorical_crossentropy(masked_targets, masked_logits, from_logits=True)

        if class_weights is not None:
            weights = tf.gather(class_weights, masked_targets)
            return tf.reduce_mean(loss * weights)
        else:
            return tf.reduce_mean(loss)

    return tf.cond(
        tf.equal(mask_sum, 0.0),
        lambda: tf.constant(0.0, dtype=tf.float32),
        compute_loss
    )

class HierarchicalLoss(tf.keras.losses.Loss):
    def __init__(self, weights=(1.0, 1.0, 1.0), level1_weights=None, level2_weights=None, level3_weights=None, label_config=None, class_histogram=None):
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

class ConditionalEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, label_to_heads = None, lambda_entropy=1.0, eps=1e-6, name="conditional_entropy_loss"):
        """
        Keras-compatible loss that applies attention entropy penalty only
        to specific heads, based on sample label.

        Args:
            lambda_entropy: scalar weight for the entropy term
            eps: small constant to prevent log(0)
        """
        super().__init__(name=name)
        self.lambda_entropy = lambda_entropy
        self.eps = eps

        # Your internal label → head mapping
        # 3: generalized → head 0
        # 4, 5: focal left/right → head 1
        self.label_to_heads = label_to_heads

        # Compute label-to-head mask matrix [num_labels, num_heads]
        self.num_heads = 2
        max_label = max(self.label_to_heads.keys()) + 1
        label_mask_matrix = []
        for label in range(max_label):
            active_heads = self.label_to_heads.get(label, [])
            row = [1.0 if h in active_heads else 0.0 for h in range(self.num_heads)]
            label_mask_matrix.append(row)
        self.label_mask = tf.constant(label_mask_matrix, dtype=tf.float32)  # shape [L, H]

    def call(self, y_true, attention_weights):
        """
        Args:
            y_true: Tensor of shape [B] or [B, 1], int class labels
            attention_weights: Tensor of shape:
                [B, T]   → single-head attention
                [B, H, T] → multi-head attention
        Returns:
            Scalar entropy penalty (float32)
        """
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)  # [B]

        shape = attention_weights.shape

        if len(shape) == 2:  # [B, T]
            attention_weights = tf.expand_dims(attention_weights, axis=1)  # [B, 1, T]
            label_mask = tf.gather(self.label_mask[:, :1], y_true)         # [B, 1]
        elif len(shape) == 3:  # [B, H, T]
            label_mask = tf.gather(self.label_mask, y_true)                # [B, H]
        else:
            raise ValueError(f"Expected attention_weights rank 2 or 3, but got shape {shape}")


        # Compute entropy per head
        entropy = -tf.reduce_sum(attention_weights * tf.math.log(attention_weights + self.eps), axis=-1)  # [B, H]

        # Apply mask
        masked_entropy = entropy * label_mask  # [B, H]
        num_active_heads = tf.reduce_sum(label_mask, axis=-1)  # [B]
        per_sample_entropy = tf.reduce_sum(masked_entropy, axis=-1) / (num_active_heads + self.eps)  # [B]

        """
        tf.print("y_true:", y_true)
        tf.print("label_mask:", label_mask)
        tf.print("entropy:", entropy)
        tf.print("masked_entropy:", masked_entropy)
        tf.print("num_active_heads:", num_active_heads)
        tf.print("per_sample_entropy:", per_sample_entropy)
        """
        
        # Return batch mean
        return self.lambda_entropy * tf.reduce_sum(per_sample_entropy)

if __name__ == "__main__":
    """
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

    '''
    print("=== Full soft class vector ===")
    print(loss_fn.get_soft_class_vector())

    print("\n=== Full soft target tensor ===")
    print(loss_fn.get_target_vector())
    '''
    
    # Query individual labels
    labels_to_test = ["neurotypical", "epileptic", "focal", "right"]

    
    for label in labels_to_test:
        print(f"\n--- For label: {label} ---")
        print("Soft class vector:")
        print(loss_fn.get_soft_class_vector(label))
        print("Target vector:")
        print(loss_fn.get_target_vector(label))
    """

    loss = ConditionalEntropyLoss(label_to_heads={3: [0], 4: [1], 5: [1]})

    # Example 1: multi-head input with focal (label 4 → head 1)
    y_true = tf.constant([4, 3, 0])  # focal, generalized, neurotypical
    attn = tf.constant([
        [[0.1, 0.9], [0.5, 0.5]],   # head 0, head 1
        [[0.8, 0.2], [0.6, 0.4]],
        [[0.5, 0.5], [0.5, 0.5]]
    ], dtype=tf.float32)  # shape [3, 2, 2]

    print("Loss:", loss(y_true, attn).numpy())

