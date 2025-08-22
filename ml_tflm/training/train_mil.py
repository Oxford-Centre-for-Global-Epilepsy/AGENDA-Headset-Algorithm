import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False
})

from ml_tflm.training.loss import BinaryLoss

# #####################
# ##### VARIABLES #####
# #####################

# ===== MODEL =====

EEGNET_F1 = 16
EEGNET_D = 2
EEGNET_F2 = 4

CLS_HID = 16

# ===== DATA =====

BATCH_SIZE = 32
SEGMENT_LEN = 256

# ===== TRAIN =====

EPOCHS = 40
TRAIN_STEPS = 100
LEARNING_RATE = 1e-4



# ######################
# ##### COMPONENTS #####
# ######################

from ml_tflm.models_tf.feature_extractor import EEGNet

class ClassifierHead(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim=0, l2_weight=1e-4):
        """
        Segment-level classifier head for MIL (2-class).

        Args:
            input_dim (int): Dimension of the segment embedding (D)
            hidden_dim (int): Optional bottleneck (set to 0 to disable)
        """
        super().__init__()
        self.use_bottleneck = hidden_dim > 0

        if self.use_bottleneck:
            self.bottleneck = tf.keras.layers.Dense(
                hidden_dim,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
            )

        self.out = tf.keras.layers.Dense(2)  # Two-class logits

    def call(self, x, training=False):
        if self.use_bottleneck:
            x = self.bottleneck(x)
        return self.out(x)

def masked_mean_pooling(logits, mask):
    mask = tf.cast(mask, tf.float32)             # [B, E]
    mask = tf.expand_dims(mask, axis=-1)         # [B, E, 1]

    logits_sum = tf.reduce_sum(logits * mask, axis=1)  # [B, K]
    mask_sum = tf.reduce_sum(mask, axis=1) + 1e-8       # [B, 1]

    return logits_sum / mask_sum  # [B, K]

def masked_separation_pooling(logits, mask):
    """
    Attention pooling based on logit separation score.

    Args:
        logits: [B, E, K] — segment-level logits
        mask:   [B, E]    — boolean mask (True = valid)

    Returns:
        pooled_logits: [B, K]
    """
    # Compute separation score: |z1 - z0| for binary classification
    sep_score = tf.abs(logits[..., 1] - logits[..., 0])  # [B, E]

    # Apply mask: invalid segments get very negative scores
    minus_inf = tf.constant(-1e9, dtype=logits.dtype)
    sep_score = tf.where(mask, sep_score, minus_inf)

    # Normalize scores with softmax across segments
    attn_weights = tf.nn.softmax(sep_score, axis=1)      # [B, E]
    attn_weights = tf.expand_dims(attn_weights, axis=-1) # [B, E, 1]

    # Weighted sum of logits
    pooled_logits = tf.reduce_sum(logits * attn_weights, axis=1)  # [B, K]

    return pooled_logits



# ########################
# ##### TRAIN HELPER #####
# ########################

from ml_tflm.training.train_utils import load_label_config, prepare_eeg_datasets

def get_dataset():
    print("-> Loading dataset...")

    train_val_sets, *_ = prepare_eeg_datasets(
        h5_file_path="ml_tflm/dataset/agenda_data_03/combined_south_africa_monopolar_standard_10_20.h5",
        dataset_name="combined_south_africa_monopolar_standard_10_20",
        label_config=load_label_config("ml_tflm/training/label_map.JSON"),
        batch_size=BATCH_SIZE,
        val_frac=0.2,
        test_frac=0.0,
        stratify=True,
        mirror_flag=False,
        internal_label_cap={0:240, 3:120, 4:60, 5:60},
        omit_channels=None,
        k_fold=False,
        chunk_size=SEGMENT_LEN
    )

    train_ds = train_val_sets[0][0]
    val_ds = train_val_sets[0][1]

    return train_ds, val_ds


# #######################
# ##### TRAIN STEPS #####
# #######################

# @tf.function
def train_step(batch, feature_extractor, classifier_head, pooling_fn, loss_fn, optimizer):
    x, y, mask = batch["data"], batch["labels"], batch["attention_mask"]  # x: [B, E, C, T], mask: [B, E]
    y = {"targets": y}

    # Flatten segments
    B, E = tf.shape(x)[0], tf.shape(x)[1]
    x_flat = tf.reshape(x, [B * E, x.shape[2], x.shape[3], 1])  # [B*E, C, T, 1]

    with tf.GradientTape() as tape:
        # Step 1: feature extraction
        z_flat = feature_extractor(x_flat, training=True)     # [B*E, D]
        logits_flat = classifier_head(z_flat, training=True)  # [B*E, K]

        # Step 2: reshape to [B, E, K]
        logits = tf.reshape(logits_flat, [B, E, -1])  # [B, E, K]

        # Step 3: masked pooling
        pooled_logits = pooling_fn(logits, mask)  # [B, K]
        y_pred = {'logits': pooled_logits}

        # Step 4: loss
        loss = loss_fn(y, y_pred)

        if feature_extractor.losses:
            loss += tf.add_n(feature_extractor.losses)
        if classifier_head.losses:
            loss += tf.add_n(classifier_head.losses)

    variables = feature_extractor.trainable_variables + classifier_head.trainable_variables
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))

    preds = tf.argmax(pooled_logits, axis=-1)
    return loss, preds, y

# @tf.function
def val_step(batch, feature_extractor, classifier_head, pooling_fn, loss_fn):
    x, y, mask = batch["data"], batch["labels"], batch["attention_mask"]  # x: [B, E, C, T], mask: [B, E]
    y = {"targets": y}

    # Flatten segments
    B, E = tf.shape(x)[0], tf.shape(x)[1]
    x_flat = tf.reshape(x, [B * E, x.shape[2], x.shape[3], 1])  # [B*E, C, T, 1]

    # Feature extraction and classification
    z_flat = feature_extractor(x_flat, training=False)        # [B*E, D]
    logits_flat = classifier_head(z_flat, training=False)     # [B*E, K]

    # Reshape back to [B, E, K]
    logits = tf.reshape(logits_flat, [B, E, -1])  # [B, E, K]

    # Pool segment logits → bag-level logits
    pooled_logits = pooling_fn(logits, mask)  # [B, K]
    y_pred = {'logits': pooled_logits}

    # Compute loss
    loss = loss_fn(y, y_pred)

    if feature_extractor.losses:
        loss += tf.add_n(feature_extractor.losses)
    if classifier_head.losses:
        loss += tf.add_n(classifier_head.losses)

    preds = tf.argmax(pooled_logits, axis=-1)
    return loss, preds, y


# ################
# ##### MAIN #####
# ################

def main():
    # --- Get Dataset ---
    train_ds, val_ds = get_dataset()

    # --- Get Model ---
    print("-> Loading models...")
    sample = next(iter(train_ds))
    _, _, NUM_CHANNELS, SEGMENT_LEN = sample["data"].shape

    feature_extractor = EEGNet(
                            num_channels=NUM_CHANNELS, num_samples=SEGMENT_LEN, 
                            F1=EEGNET_F1, D=EEGNET_D, F2=EEGNET_F2,
                            norm_layer="BATCH",
                            dropout_rate=0.5
                            )
    classifier = ClassifierHead(
                            input_dim=0,
                            hidden_dim=CLS_HID
    )

    # --- Evaluators ---
    pool_fn = masked_mean_pooling
    loss_fn = BinaryLoss()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # --- Train Loop ---
    for epoch in range(EPOCHS):
        if epoch == 10:
            pool_fn = masked_separation_pooling

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # --- Train Step(s) ---
        step_iter = enumerate(train_ds.take(TRAIN_STEPS))
        step_iter = tqdm(step_iter, total=TRAIN_STEPS, desc="Training", unit="batch")
        
        train_losses = []
        skipped_batches = 0

        for _, batch in step_iter:
            try:
                loss, _, _ = train_step(batch, feature_extractor, classifier, pool_fn, loss_fn, optimizer)
                train_losses.append(loss.numpy())
            except tf.errors.ResourceExhaustedError:
                skipped_batches += 1
                tf.keras.backend.clear_session()
                continue

        if train_losses:
            print(f" -> Train Loss: {sum(train_losses) / len(train_losses):.4f}")
        else:
            print(" -> Train Loss: N/A (All batches skipped)")
        if skipped_batches > 0:
            print(f" -> Skipped {skipped_batches} batch(es) due to OOM")

        # --- Val Step(s) ---
        val_iter = enumerate(val_ds)
        val_iter = tqdm(val_iter, desc="Validation", unit="batch")

        val_losses = []
        all_preds = []
        all_labels = []

        for _, batch in val_iter:
            try:
                loss, preds, y = val_step(batch, feature_extractor, classifier, pool_fn, loss_fn)
                val_losses.append(loss.numpy())
                all_preds.append(preds.numpy())
                all_labels.append(y["targets"][:, 0].numpy())
            except tf.errors.ResourceExhaustedError:
                tf.keras.backend.clear_session()
                continue

        # --- Print Val Loss ---
        if val_losses:
            avg_loss = sum(val_losses) / len(val_losses)
            print(f" -> Val Loss: {avg_loss:.4f}")
        else:
            print(" -> Val Loss: N/A (All batches skipped)")

        # --- Print Confusion Matrix ---
        if all_preds and all_labels:
            y_true = np.concatenate(all_labels, axis=0)
            y_pred = np.concatenate(all_preds, axis=0)

            try:
                cm = confusion_matrix(y_true, y_pred)
                labels = [f"Class {i}" for i in range(cm.shape[0])]
                cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in labels],
                                        columns=[f"Pred {l}" for l in labels])
                print("\nConfusion Matrix:")
                print(cm_df.to_string())
            except Exception as e:
                print(f" -> Could not compute confusion matrix: {e}")
        else:
            print(" -> No predictions available for confusion matrix.")

    print("=== Training Complete ===")

if __name__ == "__main__":
    main()
