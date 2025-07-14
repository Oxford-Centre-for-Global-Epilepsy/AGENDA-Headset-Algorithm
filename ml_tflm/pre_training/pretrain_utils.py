import tensorflow as tf
import random
from ml_tflm.pre_training.model_pretrain import build_projector, build_vanilla_eegnet, build_feature_extractor
from ml_tflm.pre_training.dataset_pretrain_aug import build_augmented_dataset
from tqdm import trange
from collections import deque
import numpy as np
from matplotlib.colors import ListedColormap

import h5py
from ml_tflm.pre_training.dataset_pretrain import EEGContrastiveBufferDatasetTF

import matplotlib
matplotlib.use('Agg')  # set backend before importing pyplot
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from collections import defaultdict

from sklearn.cluster import KMeans

import umap
import os

def contrastive_train_step(model_dict, optimizer_dict, loss_fn, batch):
    with tf.GradientTape() as tape:
        features = model_dict["feature_extractor"](batch["data"], training=True)
        projections = model_dict["projector"](features, training=True)
        y_pred = {
            "features": projections,
        }
        loss = loss_fn(batch["internal_label"], y_pred)

    grads = tape.gradient(
        loss,
        model_dict["feature_extractor"].trainable_variables + model_dict["projector"].trainable_variables
    )

    """
    for g in grads:
        tf.print(tf.norm(g))
    """
        
    optimizer_dict["feature_extractor"].apply_gradients(
        zip(grads[:len(model_dict["feature_extractor"].trainable_variables)],
            model_dict["feature_extractor"].trainable_variables)
    )
    optimizer_dict["projector"].apply_gradients(
        zip(grads[len(model_dict["feature_extractor"].trainable_variables):],
            model_dict["projector"].trainable_variables)
    )
    return loss

def contrastive_val_step(model_dict, loss_fn, batch):
    features = model_dict["feature_extractor"](batch["data"], training=False)
    projections = model_dict["projector"](features, training=False)
    y_pred = {
        "features": projections,
    }
    loss = loss_fn(batch["internal_label"], y_pred)    
    return loss

def grouped_supcon_train_step(model_dict, optimizer_dict, loss_fn, batch):
    with tf.GradientTape() as tape:
        # Forward pass
        features = model_dict["feature_extractor"](batch["data"], training=True)
        projections = model_dict["projector"](features, training=True)

        # === Ground truth ===
        y_true = {
            "internal_label": batch["internal_label"],
            "sample_index": batch["sample_index"],
            "vicreg_indices": batch["vicreg_indices"]
        }

        # === Model prediction ===
        y_pred = {
            "features": projections
        }

        # Compute loss
        loss = loss_fn(y_true, y_pred)

        total_loss = loss

    # Compute and apply gradients
    vars_feat = model_dict["feature_extractor"].trainable_variables
    vars_proj = model_dict["projector"].trainable_variables
    grads = tape.gradient(total_loss, vars_feat + vars_proj)

    optimizer_dict["feature_extractor"].apply_gradients(zip(grads[:len(vars_feat)], vars_feat))
    optimizer_dict["projector"].apply_gradients(zip(grads[len(vars_feat):], vars_proj))

    return total_loss

def grouped_supcon_val_step(model_dict, loss_fn, batch):
    features = model_dict["feature_extractor"](batch["data"], training=False)
    projections = model_dict["projector"](features, training=False)

    y_true = {
        "internal_label": batch["internal_label"],
        "sample_index": batch["sample_index"],
        "vicreg_indices": batch.get("vicreg_indices", None),  # optional
    }

    y_pred = {
        "features": projections,
    }

    loss = loss_fn(y_true, y_pred)

    metrics = compute_projection_metrics(projections, batch["internal_label"])
    return loss, metrics

def farthest_point_sampling(points, n_samples):
    """
    Greedy farthest point sampling from points (np.ndarray shape [N, D])
    to select n_samples points that are spread out.

    Returns indices of selected points.
    """
    N = points.shape[0]
    selected = [0]  # start from first point arbitrarily
    distances = np.full(N, np.inf)

    for _ in range(1, n_samples):
        last = selected[-1]
        # Compute distances to the last selected point
        dist_to_last = np.linalg.norm(points - points[last], axis=1)
        # Update distances to the closest selected point so far
        distances = np.minimum(distances, dist_to_last)
        # Select point with max distance to selected set
        next_index = np.argmax(distances)
        selected.append(next_index)

    return selected

def hybrid_supcon_train_step(model_dict, optimizer_dict, loss_fn_instance, loss_fn_class, batch):
    with tf.GradientTape() as tape:
        features = model_dict["feature_extractor"](batch["data"], training=True)

        proj_instance = model_dict["projector_instance"](features, training=True)
        proj_class = model_dict["projector_class"](features, training=True)

        y_true = {
            "internal_label": batch["internal_label"],
            "sample_index": batch["sample_index"]
        }

        loss_inst = loss_fn_instance(y_true, proj_instance)
        loss_cls = loss_fn_class(y_true, proj_class)
        loss = loss_inst + loss_cls

    # Collect gradients
    grads = tape.gradient(
        loss,
        model_dict["feature_extractor"].trainable_variables +
        model_dict["projector_instance"].trainable_variables +
        model_dict["projector_class"].trainable_variables
    )

    # Apply gradients
    n_feat = len(model_dict["feature_extractor"].trainable_variables)
    n_proj_i = len(model_dict["projector_instance"].trainable_variables)

    optimizer_dict["feature_extractor"].apply_gradients(
        zip(grads[:n_feat], model_dict["feature_extractor"].trainable_variables)
    )
    optimizer_dict["projector_instance"].apply_gradients(
        zip(grads[n_feat:n_feat + n_proj_i], model_dict["projector_instance"].trainable_variables)
    )
    optimizer_dict["projector_class"].apply_gradients(
        zip(grads[n_feat + n_proj_i:], model_dict["projector_class"].trainable_variables)
    )

    return {
        "total": loss,
        "instance": loss_inst,
        "class": loss_cls
    }

def hybrid_supcon_val_step(model_dict, loss_fn_instance, loss_fn_class, batch):
    features = model_dict["feature_extractor"](batch["data"], training=False)

    proj_instance = model_dict["projector_instance"](features, training=False)
    proj_class = model_dict["projector_class"](features, training=False)

    y_true = {
        "internal_label": batch["internal_label"],
        "sample_index": batch["sample_index"]
    }

    loss_inst = loss_fn_instance(y_true, proj_instance)
    loss_cls = loss_fn_class(y_true, proj_class)
    total_loss = loss_inst + loss_cls

    return {
        "total": total_loss,
        "instance": loss_inst,
        "class": loss_cls
    }

def compute_projection_metrics(projections, labels):
    """
    Compute separation metrics for projected features:
    - intra-class distance
    - inter-class distance
    - inter/intra ratio
    - projection std (collapse indicator)

    Args:
        projections: [B, D] tensor
        labels: [B] int tensor of class ids

    Returns:
        Dictionary with metric names and scalar values.
    """
    num_classes = tf.reduce_max(labels) + 1
    features_norm = tf.math.l2_normalize(projections, axis=1)

    # Intra-class distances
    intra_dists = []
    for c in tf.range(num_classes):
        mask = tf.where(labels == c)[:, 0]
        masked_feats = tf.gather(features_norm, mask)
        if tf.shape(masked_feats)[0] > 1:
            pairwise = tf.matmul(masked_feats, masked_feats, transpose_b=True)
            dists = 1.0 - pairwise
            intra_dists.append(tf.reduce_mean(dists))

    # Inter-class distances
    inter_dists = []
    for c1 in tf.range(num_classes):
        for c2 in tf.range(c1 + 1, num_classes):
            mask1 = tf.where(labels == c1)[:, 0]
            mask2 = tf.where(labels == c2)[:, 0]
            f1 = tf.gather(features_norm, mask1)
            f2 = tf.gather(features_norm, mask2)
            if tf.shape(f1)[0] > 0 and tf.shape(f2)[0] > 0:
                pairwise = tf.matmul(f1, f2, transpose_b=True)
                dists = 1.0 - pairwise
                inter_dists.append(tf.reduce_mean(dists))

    intra_mean = tf.reduce_mean(intra_dists)
    inter_mean = tf.reduce_mean(inter_dists)
    separation_ratio = inter_mean / (intra_mean + 1e-6)

    # Feature variance
    proj_std = tf.math.reduce_std(projections, axis=0)
    proj_var = tf.reduce_mean(proj_std)

    return {
        "intra_dist": intra_mean,
        "inter_dist": inter_mean,
        "sep_ratio": separation_ratio,
        "proj_std": proj_var,
    }

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

def configure_dual_model(feature_args, projector_instance_args, projector_class_args):
    """
    Creates and returns the feature extractor and two separate projectors:
    one for instance-level contrast and one for class-level contrast.

    Args:
        feature_args (dict): Arguments for building the feature extractor.
        projector_instance_args (dict): Args for the instance-level projector.
        projector_class_args (dict): Args for the class-level projector.

    Returns:
        dict: Dictionary with 'feature_extractor', 'projector_instance', and 'projector_class'.
    """
    feature_extractor = build_vanilla_eegnet(**feature_args)
    projector_instance = build_projector(**projector_instance_args)
    projector_class = build_projector(**projector_class_args)

    return {
        "feature_extractor": feature_extractor,
        "projector_instance": projector_instance,
        "projector_class": projector_class
    }

def load_eeg_contrastive_datasets(
    h5_file_path,
    dataset_name,
    label_config,
    val_frac=0.2,
    buffer_size_train=32,
    buffer_size_val=16,
    batch_size=16,
    omit_channels=None,
    seed=42
):
    """
    Splits EEG subjects into train and validation groups and loads buffer-based contrastive datasets.

    Returns:
        train_buffer (EEGContrastiveBufferDatasetTF)
        val_buffer (EEGContrastiveBufferDatasetTF)
    """
    with h5py.File(h5_file_path, "r") as f:
        subject_ids = sorted(list(f[dataset_name].keys()))

    # Shuffle and split
    random.seed(seed)
    random.shuffle(subject_ids)
    n_total = len(subject_ids)
    n_val = round(n_total * val_frac)
    val_ids = subject_ids[:n_val]
    train_ids = subject_ids[n_val:]

    if len(train_ids) == 0 or len(val_ids) == 0:
        raise ValueError("Not enough data for training or validation split. "
                         f"{len(train_ids)=}, {len(val_ids)=}, {n_total=}")

    train_buffer = EEGContrastiveBufferDatasetTF(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config,
        buffer_size=buffer_size_train,
        batch_size=batch_size,
        omit_channels=omit_channels,
        subject_ids=train_ids
    )

    val_buffer = EEGContrastiveBufferDatasetTF(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config,
        buffer_size=buffer_size_val,
        batch_size=batch_size,
        omit_channels=omit_channels,
        subject_ids=val_ids
    )

    return train_buffer, val_buffer

def split_augmented_eeg_datasets(
    h5_file_path,
    dataset_name,
    label_config,
    augment_logits,
    val_frac=0.2,
    buffer_size_train=32,
    buffer_size_val=16,
    batch_size=16,
    num_views=2,
    omit_channels=None,
    seed=42
):
    """
    Splits subject IDs into train/val and returns EEGAugmentedBufferDatasetTF objects for each.

    Returns:
        train_buffer, val_buffer
    """
    with h5py.File(h5_file_path, "r") as f:
        subject_ids = sorted(list(f[dataset_name].keys()))

    # Shuffle and split
    random.seed(seed)
    random.shuffle(subject_ids)
    n_total = len(subject_ids)
    n_val = round(n_total * val_frac)
    val_ids = subject_ids[:n_val]
    train_ids = subject_ids[n_val:]

    if len(train_ids) == 0 or len(val_ids) == 0:
        raise ValueError("Not enough data for training or validation split.")

    train_buffer = build_augmented_dataset(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config,
        augment_logits=augment_logits,
        batch_size=batch_size,
        buffer_size=buffer_size_train,
        num_views=num_views,
        omit_channels=omit_channels,
        subject_ids=train_ids
    )

    val_buffer = build_augmented_dataset(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config,
        augment_logits=augment_logits,
        batch_size=batch_size,
        buffer_size=buffer_size_val,
        num_views=num_views,
        omit_channels=omit_channels,
        subject_ids=val_ids
    )

    return train_buffer, val_buffer

def train_contrastive(
    model_dict,
    optimizer_dict,
    loss_fn,
    train_buffer,
    val_buffer,
    epochs=50,
    steps_per_epoch=100,
    val_steps=10,
    verbose=True,
    smoothing_window=3
):
    best_val_loss = float("inf")
    best_weights = {
        "feature_extractor": model_dict["feature_extractor"].get_weights(),
        "projector": model_dict["projector"].get_weights(),
    }

    val_loss_history = deque(maxlen=smoothing_window)

    for epoch in range(epochs):
        # At start of epoch
        prev_weights_feat = model_dict["feature_extractor"].get_weights()
        prev_weights_proj = model_dict["projector"].get_weights()

        # === Reload ===
        train_buffer._reload_buffer()
        # val_buffer._reload_buffer()

        # === Training ===
        train_loss = 0.0
        for _ in trange(steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs} - Training", disable=not verbose):
            train_batch = next(train_buffer.generator())
            loss = contrastive_train_step(model_dict, optimizer_dict, loss_fn, train_batch)
            train_loss += loss.numpy()
        train_loss /= steps_per_epoch

        # === Validation ===
        val_loss = 0.0
        for _ in trange(val_steps, desc=f"Epoch {epoch+1}/{epochs} - Validation", disable=not verbose):
            val_batch = next(val_buffer.generator())
            loss = contrastive_val_step(model_dict, loss_fn, val_batch)
            val_loss += loss.numpy()
        val_loss /= val_steps

        val_loss_history.append(val_loss)
        smoothed_val_loss = sum(val_loss_history) / len(val_loss_history)

        if verbose:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f} (Smoothed = {smoothed_val_loss:.4f})")

        # Save best weights based on smoothed validation loss
        if smoothed_val_loss < best_val_loss:
            best_val_loss = smoothed_val_loss
            best_weights = {
                "feature_extractor": model_dict["feature_extractor"].get_weights(),
                "projector": model_dict["projector"].get_weights(),
            }

        # After epoch
        delta_feat = sum(
            tf.norm(tf.convert_to_tensor(w_new) - tf.convert_to_tensor(w_old)).numpy()
            for w_old, w_new in zip(prev_weights_feat, model_dict["feature_extractor"].get_weights())
        )
        delta_proj = sum(
            tf.norm(tf.convert_to_tensor(w_new) - tf.convert_to_tensor(w_old)).numpy()
            for w_old, w_new in zip(prev_weights_proj, model_dict["projector"].get_weights())
        )

        print(f"Epoch {epoch+1}: Feature ΔW = {delta_feat:.6f}, Projector ΔW = {delta_proj:.6f}")

        if epoch % 5 == 0:  # Visualize every 5 epochs
            vis_batch = next(val_buffer.generator())
            features = model_dict["feature_extractor"](vis_batch["data"], training=False)
            projections = model_dict["projector"](features, training=False).numpy()
            labels = vis_batch["internal_label"]

            tsne = TSNE(n_components=2, perplexity=10, init='pca', learning_rate='auto')
            proj_2d = tsne.fit_transform(projections)

            plt.figure(figsize=(6, 5))
            scatter = plt.scatter(proj_2d[:, 0], proj_2d[:, 1], c=labels, cmap='tab10', s=10)
            plt.colorbar(scatter)
            plt.title(f"t-SNE of Projections (Epoch {epoch+1})")
            plt.tight_layout()
            plt.savefig(f"diversity_plot_epoch{epoch}.png")


    # Restore best weights before returning
    model_dict["feature_extractor"].set_weights(best_weights["feature_extractor"])
    model_dict["projector"].set_weights(best_weights["projector"])

    return best_val_loss

def train_grouped_supcon(
    model_dict,
    optimizer_dict,
    loss_fn,
    train_buffer,
    val_buffer,
    epochs=50,
    steps_per_epoch=100,
    val_steps=10,
    verbose=True,
    smoothing_window=3,
    cluster_centers=None
):
    best_val_loss = float("inf")
    best_weights = {
        "feature_extractor": model_dict["feature_extractor"].get_weights(),
        "projector": model_dict["projector"].get_weights(),
    }

    val_loss_history = deque(maxlen=smoothing_window)

    for epoch in range(epochs):
        # At start of epoch
        prev_weights_feat = model_dict["feature_extractor"].get_weights()
        prev_weights_proj = model_dict["projector"].get_weights()

        # === Reload training buffer ===
        train_buffer._reload_buffer()

        # === Training ===
        train_loss = 0.0
        for _ in trange(steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs} - Training", disable=not verbose):
            train_batch = next(train_buffer.generator())
            loss = grouped_supcon_train_step(model_dict, optimizer_dict, loss_fn, train_batch)
            train_loss += loss.numpy()
        train_loss /= steps_per_epoch

        # === Validation ===
        val_loss = 0.0
        val_metrics_accum = defaultdict(float)

        for _ in trange(val_steps, desc=f"Epoch {epoch+1}/{epochs} - Validation", disable=not verbose):
            val_batch = next(val_buffer.generator())
            loss, metrics = grouped_supcon_val_step(model_dict, loss_fn, val_batch)
            val_loss += loss.numpy()
            for k, v in metrics.items():
                val_metrics_accum[k] += v

        val_loss /= val_steps
        val_loss_history.append(val_loss)
        smoothed_val_loss = sum(val_loss_history) / len(val_loss_history)

        avg_val_metrics = {k: v / val_steps for k, v in val_metrics_accum.items()}

        if verbose:
            metric_str = ", ".join([f"{k} = {v:.4f}" for k, v in avg_val_metrics.items()])
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f} (Smoothed = {smoothed_val_loss:.4f}), {metric_str}")

        # Save best weights based on smoothed validation loss
        if smoothed_val_loss < best_val_loss:
            best_val_loss = smoothed_val_loss
            best_weights = {
                "feature_extractor": model_dict["feature_extractor"].get_weights(),
                "projector": model_dict["projector"].get_weights(),
            }

        # After epoch
        delta_feat = sum(
            tf.norm(tf.convert_to_tensor(w_new) - tf.convert_to_tensor(w_old)).numpy()
            for w_old, w_new in zip(prev_weights_feat, model_dict["feature_extractor"].get_weights())
        )
        delta_proj = sum(
            tf.norm(tf.convert_to_tensor(w_new) - tf.convert_to_tensor(w_old)).numpy()
            for w_old, w_new in zip(prev_weights_proj, model_dict["projector"].get_weights())
        )

        print(f"Epoch {epoch+1}: Feature ΔW = {delta_feat:.6f}, Projector ΔW = {delta_proj:.6f}")

        if (epoch+1) % 1 == 0:
            vis_batches = [next(val_buffer.generator()) for _ in range(10)]
            plot_umap_projection(model_dict, vis_batches, epoch+1, projector_key="projector", cluster_centers=cluster_centers)

    # Restore best weights before returning
    model_dict["feature_extractor"].set_weights(best_weights["feature_extractor"])
    model_dict["projector"].set_weights(best_weights["projector"])

    return best_val_loss

def train_hybrid_supcon(
    model_dict,
    optimizer_dict,
    loss_fn_instance,
    loss_fn_class,
    train_buffer,
    val_buffer,
    epochs=50,
    steps_per_epoch=100,
    val_steps=10,
    verbose=True,
    smoothing_window=3
):
    best_val_loss = float("inf")
    best_weights = {
        "feature_extractor": model_dict["feature_extractor"].get_weights(),
        "projector_instance": model_dict["projector_instance"].get_weights(),
        "projector_class": model_dict["projector_class"].get_weights(),
    }

    val_loss_history = deque(maxlen=smoothing_window)

    for epoch in range(epochs):
        prev_weights_feat = model_dict["feature_extractor"].get_weights()
        prev_weights_proj_i = model_dict["projector_instance"].get_weights()
        prev_weights_proj_c = model_dict["projector_class"].get_weights()

        # === Reload training buffer ===
        train_buffer._reload_buffer()

        # === Training ===
        train_loss_total = 0.0
        for _ in trange(steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs} - Training", disable=not verbose):
            train_batch = next(train_buffer.generator())
            losses = hybrid_supcon_train_step(model_dict, optimizer_dict, loss_fn_instance, loss_fn_class, train_batch)
            train_loss_total += losses["total"].numpy()
        train_loss_total /= steps_per_epoch

        # === Validation ===
        val_loss_total = 0.0
        for _ in trange(val_steps, desc=f"Epoch {epoch+1}/{epochs} - Validation", disable=not verbose):
            val_batch = next(val_buffer.generator())
            losses = hybrid_supcon_val_step(model_dict, loss_fn_instance, loss_fn_class, val_batch)
            val_loss_total += losses["total"].numpy()
        val_loss_total /= val_steps

        val_loss_history.append(val_loss_total)
        smoothed_val_loss = sum(val_loss_history) / len(val_loss_history)

        if verbose:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss_total:.4f}, Val Loss = {val_loss_total:.4f} (Smoothed = {smoothed_val_loss:.4f})")

        if smoothed_val_loss < best_val_loss:
            best_val_loss = smoothed_val_loss
            best_weights = {
                "feature_extractor": model_dict["feature_extractor"].get_weights(),
                "projector_instance": model_dict["projector_instance"].get_weights(),
                "projector_class": model_dict["projector_class"].get_weights(),
            }

        delta_feat = sum(
            tf.norm(tf.convert_to_tensor(w_new) - tf.convert_to_tensor(w_old)).numpy()
            for w_old, w_new in zip(prev_weights_feat, model_dict["feature_extractor"].get_weights())
        )
        delta_proj_i = sum(
            tf.norm(tf.convert_to_tensor(w_new) - tf.convert_to_tensor(w_old)).numpy()
            for w_old, w_new in zip(prev_weights_proj_i, model_dict["projector_instance"].get_weights())
        )
        delta_proj_c = sum(
            tf.norm(tf.convert_to_tensor(w_new) - tf.convert_to_tensor(w_old)).numpy()
            for w_old, w_new in zip(prev_weights_proj_c, model_dict["projector_class"].get_weights())
        )

        print(f"Epoch {epoch+1}: Feature ΔW = {delta_feat:.6f}, InstanceProj ΔW = {delta_proj_i:.6f}, ClassProj ΔW = {delta_proj_c:.6f}")

        if (epoch + 1) % 5 == 0:
            vis_batches = [next(val_buffer.generator()) for _ in range(20)]
            plot_umap_projection(model_dict, vis_batches, epoch+1, projector_key="projector_class")

    # Restore best weights
    model_dict["feature_extractor"].set_weights(best_weights["feature_extractor"])
    model_dict["projector_instance"].set_weights(best_weights["projector_instance"])
    model_dict["projector_class"].set_weights(best_weights["projector_class"])

    return best_val_loss

def evaluate_model_loss(model_dict, loss_fn, val_dataset, steps=50):
    """
    Evaluate SupCon loss of the current model on the validation set.

    Args:
        model_dict: {
            "feature_extractor": tf.keras.Model,
            "projector": tf.keras.Model
        }
        loss_fn: instance of SupConLoss
        val_dataset: tf.data.Dataset from your buffer.get_tf_dataset()
        steps: number of batches to evaluate

    Returns:
        Average SupCon loss over the specified number of validation batches.
    """
    total_loss = 0.0
    for i, batch in enumerate(val_dataset.take(steps)):
        features = model_dict["feature_extractor"](batch["data"], training=False)
        projections = model_dict["projector"](features, training=False)
        y_pred = {
            "features": projections,
        }
        loss = loss_fn(batch["internal_label"], y_pred)
        total_loss += loss.numpy()

    avg_loss = total_loss / steps
    print(f"[Eval] Avg SupCon val loss over {steps} steps: {avg_loss:.4f}")
    return avg_loss

def plot_umap_projection(
    model_dict,
    batches,
    epoch,
    label_key="internal_label",
    save_path_prefix="diversity_plot_epoch",
    title_prefix="UMAP of Projections",
    projector_key="projector_class",
    cluster_centers=None,
    model_save_dir="feature_extractor_checkpoints"  # new arg for model saving folder
):
    all_projections = []
    all_labels = []

    for batch in batches:
        sample_index = np.array(batch["sample_index"])
        labels = np.array(batch[label_key])
        data = batch["data"]

        # Deduplicate within batch
        _, unique_indices = np.unique(sample_index, return_index=True)
        data = tf.gather(data, unique_indices)
        labels = labels[unique_indices]

        # Forward pass
        features = model_dict["feature_extractor"](data, training=False)
        projections = np.array(model_dict[projector_key](features, training=False))

        all_projections.append(projections)
        all_labels.append(labels)

    final_proj = np.vstack(all_projections)
    final_labels = np.hstack(all_labels)

    reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, metric='cosine')
    proj_2d = reducer.fit_transform(final_proj)

    if cluster_centers is not None:
        cluster_centers_np = cluster_centers.numpy() if hasattr(cluster_centers, 'numpy') else cluster_centers
        centers_2d = reducer.transform(cluster_centers_np)

    custom_colors = ["#6A67CE", "#FF6F91", "#FFC75F"]
    cmap = ListedColormap(custom_colors)

    # Create directory for this epoch if not exists
    epoch_dir = os.path.join(model_save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(proj_2d[:, 0], proj_2d[:, 1], c=final_labels, cmap=cmap, s=10, vmin=0, vmax=2)
    cbar = plt.colorbar(scatter, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Class 0", "Class 1", "Class 2"])

    if cluster_centers is not None:
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], marker="X", s=100, c="black", label="Cluster Centers")
        plt.legend()

    plt.title(f"{title_prefix} (Epoch {epoch})")
    plt.tight_layout()
    plt.savefig(os.path.join(epoch_dir, f"{save_path_prefix}{epoch}.png"))
    plt.close()

    # Save the feature extractor
    feature_extractor_path = os.path.join(epoch_dir, "feature_extractor.keras")
    model_dict["feature_extractor"].save(feature_extractor_path)

    # Save the projector
    projector_path = os.path.join(epoch_dir, "projector.keras")
    model_dict["projector"].save(projector_path)