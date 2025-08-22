import tensorflow as tf
from tqdm import tqdm

tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False
})


# ##### FOR SOME REASONS: This is not working yet #####


# ############################
# ##### MODEL COMPONENTS #####
# ############################

# ===== FEATURE EXTRACTOR =====
from ml_tflm.models_tf.feature_extractor import EEGNet

# ===== BERT STYLE TRANSFORMER =====
class BERTStyleEncoder(tf.keras.Model):
    def __init__(self, embed_dim=64, num_heads=1, ff_dim=128, dropout=0.1, num_layers=1, max_len=128):
        super().__init__()
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        self.encoders = []
        for _ in range(num_layers):
            self.encoders.append([
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
                tf.keras.layers.Dropout(dropout),
            ])

    def call(self, x, mask=None, training=False):
        B = tf.shape(x)[0]
        E = tf.shape(x)[1]

        # Positional embedding [E, D] → broadcast to [1, E, D]
        positions = tf.range(start=0, limit=E, delta=1)
        pos_embed = self.pos_embedding(positions)
        x = x + tf.expand_dims(pos_embed, axis=0)  # [B, E, D]

        # Prepare attention mask for MHA (expecting shape [B, 1, 1, E])
        attention_mask = None
        if mask is not None:
            # Ensure mask is boolean and in correct shape
            mask = tf.cast(mask, dtype=tf.bool)        # [B, E]
            attention_mask = mask[:, tf.newaxis, tf.newaxis, :]  # [B, 1, 1, E]

        # Run through encoder blocks
        for ln1, mha, drop1, ln2, d1, d2, drop2 in self.encoders:
            attn_out = mha(
                query=ln1(x),
                key=x,
                value=x,
                attention_mask=attention_mask,
                training=training
            )
            x = x + drop1(attn_out, training=training)
            ffn_out = d2(d1(ln2(x)))
            x = x + drop2(ffn_out, training=training)

        return x  # [B, E, D]

# ===== MSP HEAD =====
class MSPHead(tf.keras.layers.Layer):
    def __init__(self, embed_dim, channels=21, segment_len=128, hidden_dim=128):
        super().__init__()
        self.channels = channels
        self.segment_len = segment_len
        self.hidden = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(channels * segment_len)
        ])

    def call(self, x):
        # x: [B, E, D] -> [B, E, C*T] -> reshape to [B, E, C, T]
        x = self.hidden(x)
        return tf.reshape(x, [-1, tf.shape(x)[1], self.channels, self.segment_len])

class ReversedEEGNetDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim=128, channels=21, segment_len=128):
        super().__init__()
        self.channels = channels
        self.segment_len = segment_len
        self.initial_shape = (1, 1, embed_dim)

        self.proj = tf.keras.layers.Dense(embed_dim, activation='relu')

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Reshape(self.initial_shape),  # [B*E, 1, 1, D]
            tf.keras.layers.Conv2DTranspose(64, (1, 4), strides=(1, 2), padding="same", activation='relu'),
            tf.keras.layers.Conv2DTranspose(32, (4, 1), strides=(2, 1), padding="same", activation='relu'),
            tf.keras.layers.Conv2DTranspose(16, (2, 4), strides=(2, 2), padding="same", activation='relu'),
            tf.keras.layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), padding="same", activation='tanh'),  # [B*E, C, T, 1]
        ])

    def call(self, x):
        # x: [B, E, D]
        B, E, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        x = tf.reshape(x, [-1, D])  # [B*E, D]
        x = self.proj(x)
        x = self.decoder(x)        # [B*E, C, T, 1]
        x = tf.squeeze(x, -1)      # [B*E, C, T]
        return tf.reshape(x, [B, E, self.channels, self.segment_len])  # [B, E, C, T]



# ################
# ##### LOSS #####
# ################

def reconstruction_loss(y_true, y_pred, mask):
    """
    Computes the reconstruction loss over masked segments only.

    Args:
        y_true: [B, E, C, T] — ground truth EEG segments
        y_pred: [B, E, C, T] — predicted EEG segments
        mask:   [B, E]       — boolean mask where 1 = masked, 0 = unmasked

    Returns:
        scalar loss
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)  # [B, E]

    # Debug: Log mean and std of prediction
    # tf.print("y_pred mean:", tf.reduce_mean(y_pred), "std:", tf.math.reduce_std(y_pred))

    # Compute squared error per segment
    squared_error = tf.reduce_mean(tf.square(y_true - y_pred + 1e-6), axis=[2, 3])  # [B, E]

    # Apply mask
    masked_error = squared_error * mask  # [B, E]

    # Compute mean over all masked positions (avoid divide-by-zero)
    total_masked = tf.reduce_sum(mask)
    loss = tf.reduce_sum(masked_error) / tf.maximum(total_masked, 1.0)

    return loss



# ###################
# ##### DATASET #####
# ###################

# ===== DATASET PREPARATION =====
from ml_tflm.training.train_utils import prepare_eeg_datasets, load_label_config

# ===== MASKING LOGIC =====
def apply_random_mask(x, mask_ratio=0.3):
    """
    Applies random binary masks to input EEG sequences.

    Args:
        x: [B, E, C, T] — input EEG segments
        mask_ratio: float — fraction of segments to mask

    Returns:
        masked_x: [B, E, C, T] — input with masked segments zeroed
        target:   [B, E, C, T] — original (ground truth) data for all segments
        mask:     [B, E]       — binary mask indicating which segments were masked
    """
    B, E = tf.shape(x)[0], tf.shape(x)[1]

    # Random mask: for each segment, draw 1 with probability `mask_ratio`
    mask = tf.cast(tf.random.uniform([B, E]) < mask_ratio, x.dtype)  # [B, E]
    mask_exp = tf.reshape(mask, [B, E, 1, 1])  # [B, E, 1, 1] to broadcast

    # Store original as target
    target = tf.identity(x)

    # Apply mask: zero out masked segments
    masked_x = x * (1.0 - mask_exp)

    return masked_x, target, tf.cast(mask, tf.bool)



# ########################
# ##### TRAINER TOOL #####
# ########################

# ===== TRAIN STEP =====
@tf.function
def train_step(batch, model_components, optimizer, mask_ratio=0.3):
    """
    Single training step.

    Args:
        batch: Dict with 'data': [B, E, C, T]
        model_components: Dict with keys "feature_extractor", "transformer_encoder", "msp_head"
        optimizer: tf.keras.optimizers.Optimizer

    Returns:
        loss value
    """
    x_raw = batch["data"]  # [B, E, C, T]
    x_masked, y_true, mask = apply_random_mask(x_raw, mask_ratio=mask_ratio)

    B = tf.shape(x_masked)[0]
    E = tf.shape(x_masked)[1]
    C = tf.shape(x_masked)[2]
    T = tf.shape(x_masked)[3]

    feature_extractor = model_components["feature_extractor"]
    encoder = model_components["transformer_encoder"]
    msp_head = model_components["msp_head"]

    with tf.GradientTape() as tape:
        # Reshape for feature extractor
        x_flat = tf.reshape(x_masked, [B * E, C, T, 1])               # [B*E, C, T, 1]
        x_embed_flat = feature_extractor(x_flat, training=True)      # [B*E, D]
        D = tf.shape(x_embed_flat)[-1]
        x_embed = tf.reshape(x_embed_flat, [B, E, D])                 # [B, E, D]
        encoded = encoder(x_embed, mask=mask, training=True)         # [B, E, D]
        y_pred = msp_head(encoded, training=True)                    # [B, E, C, T]
        loss = reconstruction_loss(y_true, y_pred, mask)             # scalar

    variables = (
        feature_extractor.trainable_variables +
        encoder.trainable_variables +
        msp_head.trainable_variables
    )
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return loss

# ===== VAL STEP =====
@tf.function
def val_step(batch, model_components, mask_ratio=0.3):
    """
    Single validation step.

    Args:
        batch: Dict with 'data': [B, E, C, T]
        model_components: Dict with keys "feature_extractor", "transformer_encoder", "msp_head"

    Returns:
        loss value
    """
    x_raw = batch["data"]  # [B, E, C, T]
    x_masked, y_true, mask = apply_random_mask(x_raw, mask_ratio=mask_ratio)

    B = tf.shape(x_masked)[0]
    E = tf.shape(x_masked)[1]
    C = tf.shape(x_masked)[2]
    T = tf.shape(x_masked)[3]

    feature_extractor = model_components["feature_extractor"]
    encoder = model_components["transformer_encoder"]
    msp_head = model_components["msp_head"]

    # Reshape for feature extractor
    x_flat = tf.reshape(x_masked, [B * E, C, T, 1])               # [B*E, C, T, 1]
    x_embed_flat = feature_extractor(x_flat, training=False)     # [B*E, D]
    D = tf.shape(x_embed_flat)[-1]
    x_embed = tf.reshape(x_embed_flat, [B, E, D])                 # [B, E, D]

    encoded = encoder(x_embed, mask=mask, training=False)        # [B, E, D]
    y_pred = msp_head(encoded, training=False)                   # [B, E, C, T]
    loss = reconstruction_loss(y_true, y_pred, mask)
    return loss



# ################
# ##### MAIN #####
# ################

def main():
    print("=== Starting Masked Segment Prediction (MSP) Pretraining ===")

    # === Hardcoded Configuration ===
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    STEPS_PER_EPOCH = 200
    MASK_RATIO = 0.1
    EPOCH_LEN = 128 # Note: this refers to the sequence length of continuous EEG segments

    # === Instantiating Components ===
    # --- Load Datasets ---
    print("-> Loading dataset...")
    train_val_sets, *_ = prepare_eeg_datasets(  # ← Already imported from your `train_utils`
        h5_file_path="ml_tflm/dataset/agenda_data_03/combined_south_africa_monopolar_standard_10_20.h5",
        dataset_name="combined_south_africa_monopolar_standard_10_20",
        label_config=load_label_config("ml_tflm/training/label_map.JSON"),
        batch_size=BATCH_SIZE,
        val_frac=0.1,
        test_frac=0.0,
        stratify=True,
        mirror_flag=False,
        internal_label_cap={240, 120, 60, 60},
        omit_channels=None,
        k_fold=False,
        chunk_size=EPOCH_LEN
    )

    train_ds = train_val_sets[0][0]
    val_ds = train_val_sets[0][1]

    # --- Get Shape Info from One Batch ---
    sample = next(iter(train_ds))
    _, _, NUM_CHANNELS, SEGMENT_LEN = sample["data"].shape

    # --- Build Model Components ---
    feature_extractor = EEGNet(num_channels=NUM_CHANNELS, num_samples=SEGMENT_LEN, 
                               norm_layer=tf.keras.layers.BatchNormalization, 
                               dropout_rate=0.5)
    transformer_encoder = BERTStyleEncoder(embed_dim=64, num_heads=1, ff_dim=128, dropout=0.1, num_layers=1, max_len=EPOCH_LEN)
    msp_head = MSPHead(embed_dim=64, channels=NUM_CHANNELS, segment_len=SEGMENT_LEN, hidden_dim=1024)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    model_components = {
        "feature_extractor": feature_extractor,
        "transformer_encoder": transformer_encoder,
        "msp_head": msp_head
    }

    # === Training Loop ===
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # --- Train Step(s) ---
        step_iter = enumerate(train_ds.take(STEPS_PER_EPOCH))
        step_iter = tqdm(step_iter, total=STEPS_PER_EPOCH, desc="Training", unit="batch")
        
        train_losses = []
        skipped_batches = 0

        for _, batch in step_iter:
            try:
                loss = train_step(batch, model_components, optimizer, MASK_RATIO)
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
        for _, batch in val_iter:
            try:
                loss = val_step(batch, model_components, MASK_RATIO)
                val_losses.append(loss.numpy())
            except tf.errors.ResourceExhaustedError:
                tf.keras.backend.clear_session()
                continue

        if val_losses:
            print(f" -> Val Loss: {sum(val_losses) / len(val_losses):.4f}")
        else:
            print(" -> Val Loss: N/A (All batches skipped)")

    print("=== Training Complete ===")

if __name__ == "__main__":
    main()