import tensorflow as tf

def cast_prediction_flat(output_list, return_hierarchical=False):
    """
    Casts a list of model outputs to predicted class indices.

    Args:
        output_list (list[dict]): Each dict contains key "logits" (tf.Tensor of shape [B_i, num_classes])

    Returns:
        tf.Tensor: Predicted class indices (int32)
    """
    logits_batch = tf.concat([entry["logits"] for entry in output_list], axis=0)
    flat_pred = tf.argmax(logits_batch, axis=1, output_type=tf.int32)

    if not return_hierarchical:
        return flat_pred

    flat_probs = tf.nn.softmax(logits_batch, axis=1)

    # Probabilities from flat classifier: [neurotypical, generalized, focal-left, focal-right]
    p_neurotypical = flat_probs[:, 0]
    p_generalized = flat_probs[:, 1]
    p_focal_left = flat_probs[:, 2]
    p_focal_right = flat_probs[:, 3]

    # Normalize within each decision node
    p_focal_total = p_focal_left + p_focal_right + 1e-8
    p_focal_left_norm = p_focal_left / p_focal_total
    p_focal_right_norm = p_focal_right / p_focal_total

    p_epileptic_total = p_generalized + p_focal_total
    p_generalized_norm = p_generalized / (p_epileptic_total + 1e-8)
    p_focal_norm = p_focal_total / (p_epileptic_total + 1e-8)

    p_level1_total = p_neurotypical + p_epileptic_total
    p_neurotypical_norm = p_neurotypical / (p_level1_total + 1e-8)
    p_epileptic_norm = p_epileptic_total / (p_level1_total + 1e-8)

    # Walk the hierarchy top-down to predict each level
    level1_pred = tf.cast(p_epileptic_norm > p_neurotypical_norm, tf.int32)
    level2_pred = tf.cast(p_generalized_norm > p_focal_norm, tf.int32)
    level3_pred = tf.cast(p_focal_right_norm > p_focal_left_norm, tf.int32)

    return level1_pred, level2_pred, level3_pred

def cast_prediction_hierarchical(predictions, return_hierarchical=False):
    """
    Converts hierarchical model logits into compressed 4-class flat predictions:
        0 = neurotypical
        1 = generalized
        2 = focal-left
        3 = focal-right
        -1 = invalid (should not occur under normal hierarchy)

    Args:
        predictions (dict or list of dict): output(s) from hierarchical model:
            {
                "level1_logits": [B, 2],
                "level2_logits": [B, 2],
                "level3_logits": [B, 2]
            }

    Returns:
        Tensor of shape [B] with flat class IDs (int32), or -1 if invalid.
    """
    if isinstance(predictions, list):
        # Concatenate across batch dimension
        level1_logits = tf.concat([p["level1_logits"] for p in predictions], axis=0)
        level2_logits = tf.concat([p["level2_logits"] for p in predictions], axis=0)
        level3_logits = tf.concat([p["level3_logits"] for p in predictions], axis=0)
    else:
        level1_logits = predictions["level1_logits"]
        level2_logits = predictions["level2_logits"]
        level3_logits = predictions["level3_logits"]

    level1_pred = predict_from_logits(level1_logits)
    level2_pred = predict_from_logits(level2_logits)
    level3_pred = predict_from_logits(level3_logits)

    if return_hierarchical:
        return level1_pred, level2_pred, level3_pred

    flat_class_ids = tf.fill(tf.shape(level1_pred), -1)

    is_neurotypical = tf.equal(level1_pred, 0)
    is_generalized = tf.logical_and(tf.equal(level1_pred, 1), tf.equal(level2_pred, 1))
    is_focal_left = tf.logical_and(tf.equal(level1_pred, 1),
                        tf.logical_and(tf.equal(level2_pred, 0), tf.equal(level3_pred, 0)))
    is_focal_right = tf.logical_and(tf.equal(level1_pred, 1),
                         tf.logical_and(tf.equal(level2_pred, 0), tf.equal(level3_pred, 1)))

    flat_class_ids = tf.where(is_neurotypical, 0, flat_class_ids)
    flat_class_ids = tf.where(is_generalized, 1, flat_class_ids)
    flat_class_ids = tf.where(is_focal_left, 2, flat_class_ids)
    flat_class_ids = tf.where(is_focal_right, 3, flat_class_ids)

    return flat_class_ids

def cast_prediction_binary(output_list, return_hierarchical=False):
    """
    Casts a list of model outputs to binary predicted class indices.

    Args:
        output_list (list[dict]): Each dict contains key "logits" (tf.Tensor of shape [B, 2])
        return_hierarchical (bool): Ignored; always returns flat binary predictions.

    Returns:
        tf.Tensor: Predicted binary class indices (int32), shape [B]
    """
    logits_batch = tf.concat([entry["logits"] for entry in output_list], axis=0)
    pred = tf.argmax(logits_batch, axis=1, output_type=tf.int32)
    return pred

def predict_from_logits(logits, threshold=0.5):
    """
    Converts raw logits to predicted class labels.

    Supports:
    - [B, 2]: softmax-style logits → argmax
    - [B, 1] or [B]: binary logits → sigmoid + threshold

    Args:
        logits: Tensor of shape [B], [B, 1], or [B, 2]
        threshold: Threshold to use for binary case (default 0.5)

    Returns:
        Tensor of shape [B] with predicted class labels (int32)
    """
    shape = logits.shape

    if shape.rank == 2 and shape[-1] == 2:
        return tf.argmax(logits, axis=1, output_type=tf.int32)

    if (shape.rank == 2 and shape[-1] == 1) or shape.rank == 1:
        probs = tf.sigmoid(tf.squeeze(logits, axis=-1) if shape.rank == 2 else logits)
        return tf.cast(probs > threshold, tf.int32)

    raise ValueError(f"Unsupported logits shape: {shape}")

def cast_labels(labels, internal_index_map):
    """
    Casts labels to internal indices based on the label configuration.

    Args:
        labels (tf.Tensor): Tensor of string labels.
        label_config (dict): Label configuration containing 'label_map'.

    Returns:
        tf.Tensor: Tensor of internal indices.
    """

    if isinstance(labels, tf.Tensor):
        labels = labels.numpy().tolist()

    return tf.convert_to_tensor([internal_index_map[label] for label in labels], dtype=tf.int32)

# Caster registry for Hydra lookup
CASTER_REGISTRY = {
    "cast_prediction_flat": cast_prediction_flat,
    "cast_prediction_hierarchical": cast_prediction_hierarchical,
    "cast_prediction_binary": cast_prediction_binary
}
