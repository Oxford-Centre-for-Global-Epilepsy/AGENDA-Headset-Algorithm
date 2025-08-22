import tensorflow as tf
import numpy as np

class NTXentLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1, name="nt_xent_loss"):
        super().__init__(name=name)
        self.temperature = temperature
    def call(self, y_true, y_pred):
        """
        Args:
            y_pred: dict with keys:
                - "features": [B, D]
                - "attn_weights" (optional): [B]
            y_true: [B] - integer subject IDs
        """
        features = tf.math.l2_normalize(y_pred["features"], axis=-1)  # [B, D]

        sim_matrix = tf.matmul(features, features, transpose_b=True) / self.temperature  # [B, B]

        labels = tf.reshape(y_true, [-1])
        label_eq = tf.cast(tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0)), tf.float32)
        mask_off_diag = 1.0 - tf.eye(tf.shape(labels)[0])
        positive_mask = label_eq * mask_off_diag  # [B, B]

        exp_sim = tf.exp(sim_matrix) * mask_off_diag
        denom = tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-8

        log_prob = sim_matrix - tf.math.log(denom)
        pos_log_prob = tf.reduce_sum(log_prob * positive_mask, axis=1)
        num_pos = tf.reduce_sum(positive_mask, axis=1) + 1e-8
        sample_losses = -pos_log_prob / num_pos  # [B]

        if "attn_weights" in y_pred:
            weights = tf.reshape(y_pred["attn_weights"], [-1])
            loss = tf.reduce_sum(weights * sample_losses)
        else:
            loss = tf.reduce_mean(sample_losses)

        return loss

class NTXentMaskedLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1, top_k_percent=0.1, class_histogram=None, name="nt_xent_entropy_masked_loss"):
        super().__init__(name=name)
        self.temperature = temperature
        self.top_k_percent = top_k_percent

        if class_histogram is not None:
            histogram = tf.constant(class_histogram, dtype=tf.float32)
            inv_histogram = 1.0 / (histogram + 1e-8)
            self.class_weights = inv_histogram * (tf.cast(tf.size(histogram), tf.float32) / tf.reduce_sum(inv_histogram))
        else:
            self.class_weights = None

    def call(self, y_pred, y_true):
        B = tf.shape(y_pred)[0]
        features = tf.math.l2_normalize(y_pred, axis=-1)

        sim_matrix = tf.matmul(features, features, transpose_b=True) / self.temperature
        mask_off_diag = 1.0 - tf.eye(B)
        sim_matrix_no_diag = sim_matrix * mask_off_diag

        sim_softmax = tf.nn.softmax(sim_matrix_no_diag, axis=1)
        entropy = -tf.reduce_sum(sim_softmax * tf.math.log(sim_softmax + 1e-8), axis=1)

        labels = tf.reshape(y_true, [-1])
        unique_labels, _ = tf.unique(labels)

        anchor_mask = tf.zeros_like(entropy, dtype=tf.float32)  # initialize to all zeros

        for label in unique_labels:
            label_indices = tf.where(tf.equal(labels, label))[:, 0]
            label_entropies = tf.gather(entropy, label_indices)

            k = tf.cast(
                tf.math.maximum(1, tf.cast(self.top_k_percent * tf.cast(tf.shape(label_indices)[0], tf.float32), tf.int32)),
                tf.int32,
            )
            _, topk_relative_indices = tf.math.top_k(-label_entropies, k=k, sorted=False)
            topk_absolute_indices = tf.gather(label_indices, topk_relative_indices)

            updates = tf.ones_like(topk_absolute_indices, dtype=tf.float32)
            anchor_mask += tf.scatter_nd(indices=tf.expand_dims(topk_absolute_indices, axis=1),
                                         updates=updates,
                                         shape=[B])

        # Expand anchor mask to both row and column axes
        anchor_row_mask = tf.expand_dims(anchor_mask, 1)  # shape [B, 1]
        anchor_col_mask = tf.expand_dims(anchor_mask, 0)  # shape [1, B]

        # Joint mask: both anchor and target must be confident
        joint_anchor_mask = anchor_row_mask * anchor_col_mask  # shape [B, B]

        # Same class, off-diagonal, and both confident
        label_eq = tf.cast(tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0)), tf.float32)
        positive_mask = label_eq * mask_off_diag * joint_anchor_mask


        log_softmax_sim = sim_matrix - tf.math.log(
            tf.reduce_sum(tf.exp(sim_matrix_no_diag), axis=1, keepdims=True) + 1e-8
        )
        pos_log_prob = tf.reduce_sum(log_softmax_sim * positive_mask, axis=1)
        num_pos = tf.reduce_sum(positive_mask, axis=1) + 1e-8
        base_loss = - (pos_log_prob / num_pos)

        if self.class_weights is not None:
            sample_weights = tf.gather(self.class_weights, tf.cast(labels, tf.int32))
            masked_loss = anchor_mask * base_loss * sample_weights
            final_loss = tf.reduce_sum(masked_loss) / (tf.reduce_sum(anchor_mask * sample_weights) + 1e-8)
        else:
            masked_loss = anchor_mask * base_loss
            final_loss = tf.reduce_sum(masked_loss) / (tf.reduce_sum(anchor_mask) + 1e-8)

        return final_loss
   
class InstanceSupConLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1, vicreg_weight=None, name="instance_supcon_loss"):
        super().__init__(name=name)
        self.temperature = temperature

    def call(self, y_true, y_pred):
        features = tf.math.l2_normalize(y_pred["features"], axis=-1)  # [B, D]
        sim_matrix = tf.matmul(features, features, transpose_b=True) / self.temperature
        B = tf.shape(features)[0]

        sample_index = tf.cast(tf.reshape(y_true["sample_index"], [-1]), tf.int32)
        mask_self = tf.eye(B, dtype=tf.bool)

        pos_mask = tf.equal(
            tf.expand_dims(sample_index, 1),
            tf.expand_dims(sample_index, 0)
        ) & ~mask_self

        neg_mask = ~mask_self

        # Stable log-softmax
        logits = sim_matrix - tf.reduce_max(sim_matrix, axis=1, keepdims=True)
        exp_logits = tf.exp(logits) * tf.cast(neg_mask, tf.float32)
        denom = tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-8
        log_prob = logits - tf.math.log(denom)

        loss_mat = -log_prob * tf.cast(pos_mask, tf.float32)
        pos_per_row = tf.reduce_sum(tf.cast(pos_mask, tf.float32), axis=1)
        loss = tf.reduce_sum(loss_mat, axis=1) / tf.maximum(pos_per_row, 1.0)
        return tf.reduce_mean(loss)

class LabelSupConLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1, vicreg_weight=1.0, name="label_supcon_loss"):
        """
        Args:
            temperature: Contrastive temperature.
            vicreg_weight: Weight of the VICReg regularization (set to 0 to disable).
        """
        super().__init__(name=name)
        self.temperature = temperature
        self.vicreg_weight = vicreg_weight

    def call(self, y_true, y_pred):
        # === Supervised Contrastive Loss ===
        features = tf.math.l2_normalize(y_pred["features"], axis=-1)  # [B, D]
        tf.debugging.check_numerics(features, "SupCon features contain NaN or Inf!")

        sim_matrix = tf.matmul(features, features, transpose_b=True) / self.temperature
        sim_matrix = tf.clip_by_value(sim_matrix, -50.0, 50.0)

        B = tf.shape(features)[0]
        internal_label = tf.cast(tf.reshape(y_true["internal_label"], [-1]), tf.int32)
        mask_self = tf.eye(B, dtype=tf.bool)

        pos_mask = tf.equal(
            tf.expand_dims(internal_label, 1),
            tf.expand_dims(internal_label, 0)
        ) & ~mask_self

        neg_mask = ~mask_self
        logits = sim_matrix - tf.reduce_max(sim_matrix, axis=1, keepdims=True)
        exp_logits = tf.exp(logits) * tf.cast(neg_mask, tf.float32)
        denom = tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-4
        log_prob = logits - tf.math.log(denom)
        log_prob = tf.clip_by_value(log_prob, -100.0, 100.0)

        loss_mat = -log_prob * tf.cast(pos_mask, tf.float32)
        pos_per_row = tf.reduce_sum(tf.cast(pos_mask, tf.float32), axis=1)
        valid_rows = tf.where(pos_per_row > 0.0)
        loss_per_sample = tf.reduce_sum(loss_mat, axis=1) / tf.maximum(pos_per_row, 1.0)
        loss_per_sample = tf.gather(loss_per_sample, valid_rows[:, 0])
        supcon_loss = tf.reduce_mean(loss_per_sample)

        # === VICReg Regularization ===
        vicreg_loss_term = 0.0
        if self.vicreg_weight > 0.0 and "vicreg_indices" in y_true:
            indices = y_true["vicreg_indices"]               # shape [B_orig, 2]
            indices = tf.cast(indices, tf.int32)
            z1 = tf.gather(features, indices[:, 0])          # [B_orig, D]
            z2 = tf.gather(features, indices[:, 1])          # [B_orig, D]
            vicreg_loss_term = vicreg_loss(z1, z2) * self.vicreg_weight

        return supcon_loss + vicreg_loss_term
    
class TopKContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1, vicreg_weight=1.0, top_k=128, name="topk_supcon_loss"):
        super().__init__(name=name)
        self.temperature = temperature
        self.vicreg_weight = vicreg_weight
        self.top_k = top_k

    def call(self, y_true, y_pred):
        features = tf.math.l2_normalize(y_pred["features"], axis=-1)  # [B, D]
        sim_matrix = tf.matmul(features, features, transpose_b=True) / self.temperature  # [B, B]
        sim_matrix = tf.clip_by_value(sim_matrix, -50.0, 50.0)

        B = tf.shape(features)[0]
        internal_label = tf.cast(tf.reshape(y_true["internal_label"], [-1]), tf.int32)
        mask_self = tf.eye(B, dtype=tf.bool)

        # Mask out self similarity by setting very low value
        sim_no_self = tf.where(mask_self, tf.fill(tf.shape(sim_matrix), -1e9), sim_matrix)

        # For each sample, get top_k indices by similarity
        topk_vals, topk_indices = tf.math.top_k(sim_no_self, k=tf.minimum(self.top_k, B - 1))  # [B, k]

        # Gather labels of top-k neighbors
        expanded_labels = tf.expand_dims(internal_label, 1)  # [B,1]
        topk_labels = tf.gather(internal_label, topk_indices)  # [B, k]

        # Create positive mask for top-k neighbors
        pos_mask_topk = tf.equal(expanded_labels, topk_labels)  # [B, k]
        neg_mask_topk = tf.logical_not(pos_mask_topk)

        # Numerator: only consider positive pairs in top-k
        # Prepare logits for numerator: topk_vals masked by pos_mask_topk (others -inf)
        numerator_logits = tf.where(pos_mask_topk, topk_vals, tf.fill(tf.shape(topk_vals), -1e9))

        # Calculate log softmax denominator over all top-k pairs (positive + negative)
        exp_logits = tf.exp(topk_vals)
        denom = tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-6

        log_prob = numerator_logits - tf.math.log(denom)  # [B, k], large neg where no pos

        # Mask out invalid rows (samples without any positive pairs in top-k)
        pos_counts = tf.reduce_sum(tf.cast(pos_mask_topk, tf.float32), axis=1)
        valid_samples_mask = pos_counts > 0

        # Calculate loss per sample: average over positive pairs only
        loss_per_sample = -tf.reduce_sum(log_prob * tf.cast(pos_mask_topk, tf.float32), axis=1) / (pos_counts + 1e-6)

        # Average only valid samples
        supcon_loss = tf.reduce_sum(tf.boolean_mask(loss_per_sample, valid_samples_mask)) / (tf.reduce_sum(tf.cast(valid_samples_mask, tf.float32)) + 1e-6)

        # VICReg as before...
        vicreg_loss_term = 0.0
        if self.vicreg_weight > 0.0 and "vicreg_indices" in y_true:
            indices = tf.cast(y_true["vicreg_indices"], tf.int32)
            z1 = tf.gather(features, indices[:, 0])
            z2 = tf.gather(features, indices[:, 1])
            vicreg_loss_term = vicreg_loss(z1, z2) * self.vicreg_weight

        return supcon_loss + vicreg_loss_term

class ClusterMaskedSupConLoss(tf.keras.losses.Loss):
    def __init__(self, cluster_centers=None, temperature=0.1, vicreg_weight=1.0, name="cluster_masked_supcon_loss"):
        """
        Args:
            cluster_centers: Optional Tensor of shape [num_clusters, feature_dim], fixed cluster centers.
                             If None, fall back to standard supervised contrastive loss.
            temperature: Contrastive temperature.
            vicreg_weight: Weight of VICReg regularization.
        """
        super().__init__(name=name)
        self.cluster_centers = cluster_centers
        self.temperature = temperature
        self.vicreg_weight = vicreg_weight

    def update_cluster_centers(self, new_centers):
        """
        Update cluster centers dynamically.

        Args:
            new_centers: Tensor of shape [num_clusters, feature_dim].
        """
        self.cluster_centers = new_centers

    def call(self, y_true, y_pred):
        # Normalize features
        features = tf.math.l2_normalize(y_pred["features"], axis=-1)  # [B, D]

        B = tf.shape(features)[0]

        if self.cluster_centers is not None:
            # Use cluster centers for masking
            sim_to_centers = tf.matmul(features, self.cluster_centers, transpose_b=True)
            assigned_clusters = tf.argmax(sim_to_centers, axis=1)  # [B]

            cluster_eq = tf.equal(
                tf.expand_dims(assigned_clusters, 1),
                tf.expand_dims(assigned_clusters, 0)
            )
            mask_self = tf.eye(B, dtype=tf.bool)
            pos_mask = tf.logical_and(cluster_eq, tf.logical_not(mask_self))  # [B, B]
        else:
            # Fall back to standard supervised contrastive mask based on labels
            internal_label = tf.cast(tf.reshape(y_true["internal_label"], [-1]), tf.int32)
            label_eq = tf.equal(
                tf.expand_dims(internal_label, 1),
                tf.expand_dims(internal_label, 0)
            )
            mask_self = tf.eye(B, dtype=tf.bool)
            pos_mask = tf.logical_and(label_eq, tf.logical_not(mask_self))  # [B, B]

        # Compute similarity matrix between features
        sim_matrix = tf.matmul(features, features, transpose_b=True) / self.temperature
        sim_matrix = tf.clip_by_value(sim_matrix, -50.0, 50.0)

        logits = sim_matrix - tf.reduce_max(sim_matrix, axis=1, keepdims=True)
        exp_logits = tf.exp(logits) * tf.cast(tf.logical_not(mask_self), tf.float32)
        denom = tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-6
        log_prob = logits - tf.math.log(denom)

        # Mask positives by pos_mask
        pos_logits = tf.where(pos_mask, log_prob, tf.fill(tf.shape(log_prob), -1e9))

        pos_per_sample = tf.reduce_sum(tf.cast(pos_mask, tf.float32), axis=1)
        valid_samples_mask = pos_per_sample > 0

        sum_pos_logits = tf.reduce_sum(tf.where(pos_mask, pos_logits, tf.zeros_like(pos_logits)), axis=1)
        count_pos = tf.reduce_sum(tf.cast(pos_mask, tf.float32), axis=1)

        valid_sum_pos_logits = tf.boolean_mask(sum_pos_logits, valid_samples_mask)
        valid_count_pos = tf.boolean_mask(count_pos, valid_samples_mask)

        loss_per_sample = -valid_sum_pos_logits / valid_count_pos
        supcon_loss = tf.reduce_mean(loss_per_sample)

        # VICReg regularization if available
        vicreg_loss_term = 0.0
        if self.vicreg_weight > 0.0 and "vicreg_indices" in y_true:
            indices = tf.cast(y_true["vicreg_indices"], tf.int32)
            z1 = tf.gather(features, indices[:, 0])
            z2 = tf.gather(features, indices[:, 1])
            vicreg_loss_term = vicreg_loss(z1, z2) * self.vicreg_weight

        return supcon_loss + vicreg_loss_term

def spread_points_on_sphere(num_clusters, feature_dim, iterations=100, lr=0.1, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)

    # Initialize random points on sphere
    points = tf.Variable(tf.math.l2_normalize(tf.random.normal([num_clusters, feature_dim]), axis=1))

    for _ in range(iterations):
        # Compute pairwise cosine similarities
        sims = tf.matmul(points, points, transpose_b=True)  # [num_clusters, num_clusters]
        # Zero diagonal to ignore self-similarity
        sims = sims - tf.linalg.diag(tf.linalg.diag_part(sims))

        # Repulsion force approx = gradient to decrease pairwise similarity
        grads = tf.matmul(sims, points)  # Weighted sum of neighbors
        grads = grads - tf.reduce_sum(sims, axis=1, keepdims=True) * points

        # Gradient step to push points apart
        points.assign(tf.math.l2_normalize(points - lr * grads, axis=1))

    return points

def vicreg_loss(z1, z2, lambda_invar=5.0, lambda_var=1.0, lambda_cov=1.0):
    """
    Computes VICReg loss between two batches of projected features.

    Args:
        z1: Tensor of shape [B, D], first view.
        z2: Tensor of shape [B, D], second view.
        lambda_invar: Weight for invariance loss (mean squared error).
        lambda_var: Weight for variance regularization.
        lambda_cov: Weight for covariance regularization.

    Returns:
        Scalar VICReg loss.
    """

    # === Invariance Loss ===
    invariance = tf.reduce_mean(tf.square(z1 - z2))  # MSE

    # === Variance Loss ===
    def variance_penalty(z):
        std = tf.math.reduce_std(z, axis=0) + 1e-4  # avoid zero std
        return tf.reduce_mean(tf.nn.relu(1.0 - std))

    var_z1 = variance_penalty(z1)
    var_z2 = variance_penalty(z2)
    variance = var_z1 + var_z2

    # === Covariance Loss ===
    def covariance_penalty(z):
        z_centered = z - tf.reduce_mean(z, axis=0, keepdims=True)
        cov = tf.matmul(z_centered, z_centered, transpose_a=True) / tf.cast(tf.shape(z)[0] - 1, tf.float32)
        D = tf.cast(tf.shape(z)[1], tf.float32)
        cov_off_diag = cov * (1.0 - tf.eye(tf.shape(z)[1], dtype=tf.float32))
        return tf.reduce_sum(tf.square(cov_off_diag)) / D

    cov_z1 = covariance_penalty(z1)
    cov_z2 = covariance_penalty(z2)
    covariance = cov_z1 + cov_z2

    # === Total VICReg Loss ===
    return (
        lambda_invar * invariance +
        lambda_var * variance +
        lambda_cov * covariance
    )
