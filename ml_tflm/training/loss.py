import tensorflow as tf
import numpy as np
import csv

class StructureAwareLoss(tf.keras.losses.Loss):
    def __init__(self, distance_csv_path=None, class_histogram=None):
        super().__init__()

        # Load or use default distance matrix
        if distance_csv_path:
            with open(distance_csv_path, 'r') as f:
                reader = csv.reader(f)
                distance_matrix = np.array([[float(x) for x in row] for row in reader])
        else:
            distance_matrix = np.array([
                [0.0, 9.0, 9.0, 9.0],
                [9.0, 0.0, 3.0, 3.0],
                [9.0, 3.0, 0.0, 1.0],
                [9.0, 3.0, 1.0, 0.0],
            ])
        self.D = tf.convert_to_tensor(distance_matrix, dtype=tf.float32)
        self.D = self.D / tf.reduce_max(self.D)  # normalize for stability

        # Class histogram handling
        if class_histogram is None:
            class_histogram = tf.ones([4], dtype=tf.float32)

        inv_freq = 1.0 / class_histogram
        self.class_weights = inv_freq / tf.reduce_sum(inv_freq)  # shape: [C]

    def call(self, y_true, logits):
        """
        y_true: [B] integer class labels
        logits: [B, C] raw output (before softmax)
        """
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)  # [B]
        logits = tf.convert_to_tensor(logits, dtype=tf.float32)  # [B, C]

        # Structure-aware soft target q_y: [B, C]
        D_y = tf.gather(self.D, y_true)                   # [B, C]
        q = tf.exp(-D_y)
        q = q / tf.reduce_sum(q, axis=1, keepdims=True)   # normalize to get prob

        # Log-softmax over predicted logits
        log_p = tf.nn.log_softmax(logits, axis=1)         # [B, C]

        # KL divergence: sum q_i log(q_i / p_i) = - sum q_i log(p_i) + const
        loss_per_sample = -tf.reduce_sum(q * log_p, axis=1)  # [B]

        # Weight per sample by class inverse frequency
        sample_weights = tf.gather(self.class_weights, y_true)  # [B]
        weighted_loss = loss_per_sample * sample_weights

        return tf.reduce_mean(weighted_loss)
