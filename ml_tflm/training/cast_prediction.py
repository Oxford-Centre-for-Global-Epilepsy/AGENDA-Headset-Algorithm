import tensorflow as tf

def cast_prediction_flat(output_list):
    """
    Casts a list of model outputs to predicted class indices.

    Args:
        output_list (list[dict]): Each dict contains key "logits" (tf.Tensor of shape [B_i, num_classes])

    Returns:
        tf.Tensor: Predicted class indices (int32)
    """
    logits_batch = tf.concat([entry["logits"] for entry in output_list], axis=0)
    return tf.argmax(logits_batch, axis=1, output_type=tf.int32)


def cast_prediction_hierarchical(predictions):
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

    level1_pred = tf.argmax(level1_logits, axis=1, output_type=tf.int32)
    level2_pred = tf.argmax(level2_logits, axis=1, output_type=tf.int32)
    level3_pred = tf.argmax(level3_logits, axis=1, output_type=tf.int32)

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