import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from ml_tflm.pre_training.dataset_pretrain import load_eeg_contrastive_and_patient_datasets
from ml_tflm.pre_training.model_pretrain import configure_biphasic_model
from ml_tflm.pre_training.dataset_pretrain import EEGPatientBatchDatasetTF, EEGContrastiveBufferDatasetTF
from ml_tflm.pre_training.loss_pretrain import NTXentLoss, SupConLoss


@tf.function
def train_step_phase1(model, batch, loss_fn, optimizer):
    model.set_mode("phase1")
    x = batch["data"]
    y = batch["internal_label"]

    with tf.GradientTape() as tape:
        outputs = model(x, training=True)
        y_pred = {
            "features": outputs["z_proj"],
            "attn_weights": outputs.get("attn_weights", tf.ones_like(y, dtype=tf.float32))
        }
        loss = loss_fn(y, y_pred)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def train_step_phase2(model, batch, loss_fn, optimizer):
    model.set_mode("phase2")
    x = batch["data"]
    y = batch["internal_label"]
    mask = batch["epoch_mask"]

    with tf.GradientTape() as tape:
        outputs = model(x, attention_mask=mask, training=True)
        y_pred = {
            "features": outputs["z_pooled"]
        }

        # Base SupCon loss
        loss = loss_fn(y, y_pred)

        # Entropy range regularization on attention weights
        attn_entropy_penalty = attention_entropy_range(outputs["attn_weights"], low=0.5, high=1.0)
        loss += 0.03 * attn_entropy_penalty

        # L2 regularization (collected from model.losses)
        loss += tf.add_n(model.losses)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

@tf.function
def val_step_phase1(model, batch, loss_fn):
    model.set_mode("phase1")
    x = batch["data"]
    y = batch["internal_label"]

    outputs = model(x, training=False)
    y_pred = {
        "features": outputs["z_proj"],
        "attn_weights": outputs.get("attn_weights", tf.ones_like(y, dtype=tf.float32))
    }
    loss = loss_fn(y, y_pred)

    return loss

@tf.function
def val_step_phase2(model, batch, loss_fn):
    model.set_mode("phase2")
    x = batch["data"]
    y = batch["internal_label"]
    mask = batch["epoch_mask"]

    outputs = model(x, attention_mask=mask, training=True)
    y_pred = {
        "features": outputs["z_pooled"]
    }

    # Base SupCon loss
    loss = loss_fn(y, y_pred)

    # Entropy range regularization on attention weights
    attn_entropy_penalty = attention_entropy_range(outputs["attn_weights"], low=0.5, high=1.0)
    loss += 0.03 * attn_entropy_penalty

    # L2 regularization (collected from model.losses)
    loss += tf.add_n(model.losses)

    return loss

@tf.function
def eval_attention_entropy(model, dataset, num_batches=10):
    model.set_mode("phase2")
    entropies = []
    for i, batch in enumerate(dataset):
        if i >= num_batches:
            break
        x = batch["data"]
        mask = batch["epoch_mask"]
        outputs = model(x, attention_mask=mask, training=False, return_attn_weights=True)
        weights = outputs["attn_weights"]  # [B, E]

        # Compute entropy for each sample
        entropy = -tf.reduce_sum(weights * tf.math.log(weights + 1e-8), axis=1)
        entropies.append(entropy)

    return tf.concat(entropies, axis=0)

def plot_patient_embedding_distribution(model, dataset, num_batches=20):
    from sklearn.manifold import TSNE

    model.set_mode("phase2")
    pooled = []
    labels = []

    for i, batch in enumerate(dataset):
        if i >= num_batches:
            break
        x = batch["data"]
        y = batch["internal_label"]
        mask = batch["epoch_mask"]
        outputs = model(x, attention_mask=mask, training=False, return_features=True)
        pooled.append(outputs["z_pooled"].numpy())
        labels.append(y.numpy())

    pooled = np.concatenate(pooled, axis=0)
    labels = np.concatenate(labels, axis=0)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(pooled)

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=str(label), alpha=0.6)
    plt.title("t-SNE of Patient Pooled Embeddings")
    plt.legend(title="Label")
    plt.tight_layout()
    plt.show()

def overfit_single_batch(model, loss_fn, dataset_phase1, dataset_phase2, optimizer, num_steps=50):
    """
    print("=== Phase 1 Overfit Test ===")
    batch = next(iter(dataset_phase1))
    labels = batch["internal_label"]    # [B]
    tf.print("Labels:", labels)

    for step in range(num_steps):
        loss = train_step_phase1(model, batch, loss_fn, optimizer)
        if step % 10 == 0:
            print(f"[Phase1] Step {step+1}/{num_steps} - Loss: {float(loss):.4f}")
    """
            
    print("\n=== Phase 2 Overfit Test ===")
    batch = next(iter(dataset_phase2))  # Get a single batch from phase 2 dataset
    labels = batch["internal_label"]    # [B]
    tf.print("Labels:", labels)

    # Generate synthetic structured features
    fake_features = make_structured_proj(labels)  # [B, 200, 32]
    attention_mask = tf.ones((fake_features.shape[0], fake_features.shape[1]))  # shape: [B, E]

    # Set model to attention_test mode
    model.set_mode("attention_test")
    model.clip_attention = True
    model.temperature = 0.4

    for step in range(num_steps):
        with tf.GradientTape() as tape:
            outputs = model(fake_features, attention_mask=attention_mask, training=True)
            # Loss computed only using attention weights and known labels
            y_pred = {
                    "features": outputs["z_pooled"]
                }
            loss = loss_fn(labels, y_pred)
            loss += 0.03*attention_entropy_range(outputs["attn_weights"])
            loss += tf.add_n(model.losses)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 10 == 0 or step == num_steps - 1:
            print(f"[Phase2] Step {step+1}/{num_steps} - Loss: {float(loss):.4f}")

    # Extract top 20 attention scores per patient
    attn_weights = outputs["attn_weights"].numpy()  # shape: [B, E]
    B, E = attn_weights.shape

    print("\nTop 20 Attention Scores per Patient:")
    for i in range(B):
        top20 = sorted(attn_weights[i], reverse=True)[:20]
        print(f"Patient {i}: {['{:.4f}'.format(w) for w in top20]}")

def make_structured_proj(labels, num_epochs=200, num_informative=10, feature_dim=32, num_active_dims=8):
    """
    Generates synthetic projection vectors for attention testing.
    
    Args:
        labels: Tensor of shape [B], containing class labels (int: 0 to 3).
        num_epochs: Number of epochs per patient (E).
        num_informative: Number of informative (class-correlated) vectors per patient.
        feature_dim: Dimensionality of each feature vector (D).
        num_active_dims: Number of "hot" dimensions in each informative vector (e.g. 8-hot).
        
    Returns:
        Tensor of shape [B, E, D] with synthetic features.
    """
    B = tf.shape(labels)[0]
    E = num_epochs
    D = feature_dim
    
    # Predefined informative templates for 4 classes
    templates = tf.constant([
        [1]*num_active_dims + [0]*(D - num_active_dims),         # class 0
        [0]*8 + [1]*num_active_dims + [0]*(D - 2*num_active_dims),  # class 1
        [0]*16 + [1]*num_active_dims + [0]*(D - 3*num_active_dims), # class 2
        [0]*24 + [1]*num_active_dims + [0]*(D - 4*num_active_dims)  # class 3
    ], dtype=tf.float32)  # [4, D]
    
    # Lookup templates for each label
    informative_vectors = tf.gather(templates, labels)           # [B, D]
    informative_vectors = tf.repeat(tf.expand_dims(informative_vectors, 1), num_informative, axis=1)  # [B, 10, D]
    
    # Add small random jitter to each informative vector to simulate variability
    informative_vectors += tf.random.normal(tf.shape(informative_vectors), mean=0.0, stddev=0.05)
    
    # Generate random noise for rest
    num_noise = E - num_informative
    noise_vectors = tf.random.normal([B, num_noise, D], mean=0.0, stddev=1.0)
    
    # Concatenate and shuffle along epoch dimension
    full = tf.concat([informative_vectors, noise_vectors], axis=1)  # [B, E, D]
    
    # Shuffle each patient's 200 vectors independently
    indices = tf.argsort(tf.random.uniform([B, E]), axis=-1)
    batch_indices = tf.tile(tf.expand_dims(tf.range(B), axis=1), [1, E])
    gather_indices = tf.stack([batch_indices, indices], axis=-1)
    shuffled = tf.gather_nd(full, gather_indices)
    
    return shuffled

def attention_entropy_range(attn_weights, low=0.5, high=1., reduction='mean'):
    """
    Encourages entropy of attention weights to stay within [low, high]
    """
    eps = 1e-8
    entropy = -tf.reduce_sum(attn_weights * tf.math.log(attn_weights + eps), axis=1)

    # Apply soft penalties for outside the range
    low_violation = tf.nn.relu(low - entropy)
    high_violation = tf.nn.relu(entropy - high)
    loss = tf.square(low_violation) + tf.square(high_violation)

    if reduction == 'mean':
        return tf.reduce_mean(loss)
    return loss

def run_overfit_attention_pretraining():
    # === Hardcoded paths and configs ===
    h5_path = "ml_tflm/dataset/agenda_data_01/combined_south_africa_monopolar_standard_10_20.h5"
    dataset_name = "combined_south_africa_monopolar_standard_10_20"
    label_config = {
        "label_map": {"neurotypical": 0, "generalized": 1, "left": 2, "right": 3},
        "inverse_label_map": {0: "neurotypical", 1: "generalized", 2: "left", 3: "right"},
    }

    # === Load Datasets ===
    ds_phase1 = EEGContrastiveBufferDatasetTF(
        h5_file_path=h5_path,
        dataset_name=dataset_name,
        label_config=label_config,
        batch_size=8
    ).get_tf_dataset()

    ds_phase2 = EEGPatientBatchDatasetTF(
        h5_file_path=h5_path,
        dataset_name=dataset_name,
        label_config=label_config,
        batch_size=8
    ).get_tf_dataset()

    # === Configure Model ===
    feature_args = {"bottleneck_dim": 16}
    projector_args = {"input_dim": 16, "projection_dim": 32}
    attention_args = {"input_dim": 32}

    model = configure_biphasic_model(feature_args, projector_args, attention_args)
    model.set_mode("build")
    _ = model(tf.zeros((1, 5, 21, 128)), attention_mask=tf.ones((1, 5)), training=True)

    # === Optimizer & Warm-up ===
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    grads = [tf.zeros_like(v) for v in model.trainable_variables]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # === Loss Function ===
    loss_fn = SupConLoss(temperature=0.16)

    # === Overfit Single Batch ===
    overfit_single_batch(model, loss_fn, ds_phase1, ds_phase2, optimizer, num_steps=2000)

def train_contrastive_eeg():
    # === Path & Dataset Setup ===
    h5_file_path = "ml_tflm/dataset/agenda_data_01/combined_south_africa_monopolar_standard_10_20.h5"
    dataset_name = "combined_south_africa_monopolar_standard_10_20"
    label_config = {
        "label_map": {
            "neurotypical": 0,
            "generalized": 1,
            "left": 2,
            "right": 3
        },
        "inverse_label_map": {
            0: "neurotypical",
            1: "generalized",
            2: "left",
            3: "right"
        }
    }

    # === Load Datasets ===
    train_buffer, val_buffer, train_patient, val_patient = load_eeg_contrastive_and_patient_datasets(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config,
        val_frac=0.2,
        buffer_size_train=32,
        buffer_size_val=16,
        batch_size_buffer=512,
        batch_size_patient=16,
        omit_channels=None,
        seed=42
    )

    ds_phase2 = train_patient.get_tf_dataset()
    ds_val1 = val_buffer.get_tf_dataset()
    ds_val2 = val_patient.get_tf_dataset()

    # === Model Setup ===
    feature_args = {"bottleneck_dim": 16}
    projector_args = {"input_dim": 16, "projection_dim": 32}
    attention_args = {"input_dim": 32}

    model = configure_biphasic_model(feature_args, projector_args, attention_args)
    model.set_mode("build")
    _ = model(tf.zeros((1, 5, 21, 128)), attention_mask=tf.ones((1, 5)), training=True)
    model.clip_attention = True
    model.temperature = 0.4

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    grads = [tf.zeros_like(v) for v in model.trainable_variables]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_fn = SupConLoss(temperature=0.16)

    # === Training Loop ===
    print("=== Starting Pretraining ===")
    for epoch in range(10):
        print(f"\nEpoch {epoch+1}/10")

        # === Reload Phase 1 Buffer ===
        train_buffer._reload_buffer()
        ds_phase1 = train_buffer.get_tf_dataset()

        # === Phase 1: Contrastive Feature Learning ===
        phase1_train_losses = []
        for batch in tqdm(iter(ds_phase1.take(100)), desc="Phase 1"):
            loss = train_step_phase1(model, batch, loss_fn, optimizer)
            phase1_train_losses.append(loss.numpy())
        print(f"Phase 1 Train Loss: {np.mean(phase1_train_losses):.4f}")

        val_losses_1 = []
        for batch in ds_val1.take(25):
            val_loss = val_step_phase1(model, batch, loss_fn)
            val_losses_1.append(val_loss.numpy())
        print(f"Phase 1 Val Loss: {np.mean(val_losses_1):.4f}")

        # === Phase 2: Attention Pooling Learning ===
        phase2_train_losses = []
        for batch in tqdm(iter(ds_phase2.take(20)), desc="Phase 2"):
            loss = train_step_phase2(model, batch, loss_fn, optimizer)
            phase2_train_losses.append(loss.numpy())
        print(f"Phase 2 Train Loss: {np.mean(phase2_train_losses):.4f}")

        val_losses_2 = []
        for batch in ds_val2.take(10):
            val_loss = val_step_phase2(model, batch, loss_fn)
            val_losses_2.append(val_loss.numpy())
        print(f"Phase 2 Val Loss: {np.mean(val_losses_2):.4f}")

    print("\n=== Pretraining Completed ===")

if __name__ == "__main__":
    train_contrastive_eeg()
