import tensorflow as tf
import tensorflow_model_optimization as tfmot
from ml_tflm.training.train_utils import get_checkpoint_manager, maybe_restore_checkpoint
import numpy as np

class Trainer:
    def __init__(self, model, loss_fn, optimizer, evaluator,
                 train_dataset, val_dataset, model_input_lookup, model_target_lookup, test_dataset=None,
                 save_ckpt=False, ckpt_interval=1, ckpt_save_dir="./checkpoints",
                 load_ckpt=False, ckpt_load_dir=None,
                 attention_warmup_epoch=None,
                 anneal_interval=None, anneal_coeff=0.5):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.model_input_lookup = model_input_lookup
        self.model_target_lookup = model_target_lookup
        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss_metric = tf.keras.metrics.Mean(name="val_loss")

        self.ckpt = None
        self.ckpt_manager = None

        if load_ckpt or save_ckpt:
            self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        if load_ckpt:
            if ckpt_load_dir is None:
                raise ValueError("Checkpoint load requested, but ckpt_load_dir is None.")

            _, load_manager = get_checkpoint_manager(self.model, self.optimizer, ckpt_load_dir)
            restored = maybe_restore_checkpoint(self.ckpt, load_manager)

            if restored:
                print(f"Restored checkpoint from {load_manager.latest_checkpoint}")
            else:
                print("No checkpoint found — training will start from scratch.")

        if save_ckpt:
            self.ckpt_interval = ckpt_interval
            self.ckpt_save_dir = ckpt_save_dir
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_save_dir, max_to_keep=5)

        self.metric_history = []

        self.attention_warmup_epoch = attention_warmup_epoch
        self.enable_attention = False
        if attention_warmup_epoch:
            self.enable_attention = False
            self._attention_warmup_done = False
        else:
            self.enable_attention = True
            self._attention_warmup_done = True

        self.anneal_interval = anneal_interval
        self.anneal_coeff = anneal_coeff

    def retrace_all(self):
        try:
            sample_batch = next(iter(self.train_dataset))

            # Force a forward pass to ensure new variables (e.g., pool) are created
            model_inputs = {k: sample_batch[v] for k, v in self.model_input_lookup.items()}
            _ = self.model(**model_inputs, use_attention=self.enable_attention, training=True)

            # Rebuild optimizer with current trainable variables
            optimizer_config = self.optimizer.get_config()
            self.optimizer = type(self.optimizer).from_config(optimizer_config)
            self.optimizer.build(self.model.trainable_variables)

            # Retrace tf.functions
            self.train_step.get_concrete_function(sample_batch)
            self.val_step.get_concrete_function(sample_batch)

            print("[Retrace] Successfully forced retracing and optimizer rebuild.")
        except Exception as e:
            print(f"[Retrace] Failed to retrace tf.functions: {e}")

    @tf.function
    def train_step(self, batch):
        model_inputs = {k: batch[v] for k, v in self.model_input_lookup.items()}
        model_targets = {k: batch[v] for k, v in self.model_target_lookup.items()}

        with tf.GradientTape() as tape:
            outputs = self.model(**model_inputs, use_attention=self.enable_attention, return_attn_weights=True, training=True)
            loss = self.loss_fn(y_pred=outputs, y_true=model_targets)
            loss += tf.add_n(self.model.losses)

            # === Add attention entropy regularization if applicable ===
            if "attention_weights" in outputs:
                attn = outputs["attention_weights"]  # shape: [B, E]
                entropy = -tf.reduce_sum(attn * tf.math.log(attn + 1e-8), axis=-1)  # shape: [B]
                mean_entropy = tf.reduce_mean(entropy)  # scalar
                loss += 0.005 * mean_entropy  # weight is tunable

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_loss_metric.update_state(loss)

    @tf.function
    def val_step(self, batch):
        model_inputs = {k: batch[v] for k, v in self.model_input_lookup.items()}
        model_targets = {k: batch[v] for k, v in self.model_target_lookup.items()}

        outputs = self.model(**model_inputs, use_attention=self.enable_attention, return_attn_weights=True, training=False)
        loss = self.loss_fn(y_pred=outputs, y_true=model_targets)
        self.val_loss_metric.update_state(loss)

    def eval_step(self):
        all_preds, all_targets = [], []
        attention_entropies = []
        attention_means = []

        for i, batch in enumerate(self.val_dataset):
            try:
                model_inputs = {k: batch[v] for k, v in self.model_input_lookup.items()}
                y = batch["internal_label"]

                outputs = self.model(**model_inputs,
                                    use_attention=self.enable_attention,
                                    training=False,
                                    return_attn_weights=True,
                                    return_features=True)

                all_preds.append(outputs)
                all_targets.extend(y.numpy().tolist())

                if "attention_weights" in outputs:
                    attention = outputs["attention_weights"]  # [B, E]
                    entropy = -tf.reduce_sum(attention * tf.math.log(attention + 1e-6), axis=-1)  # [B]
                    attention_entropies.extend(entropy.numpy().tolist())
                    attention_means.append(tf.reduce_mean(attention, axis=0).numpy())  # [E]

            except tf.errors.ResourceExhaustedError:
                print(f"[Eval Step | Batch {i}] Skipping batch due to memory error.")
                tf.keras.backend.clear_session()
                import gc
                gc.collect()
                continue

        # Aggregate predictions and evaluate
        target_tensor = tf.convert_to_tensor(all_targets, dtype=tf.int32)
        results = self.evaluator.evaluate(all_preds, target_tensor)

        # Add attention stats
        if attention_entropies:
            results["attention_entropy_mean"] = float(np.mean(attention_entropies))
            results["attention_entropy_std"] = float(np.std(attention_entropies))

        """
        if attention_means:
            results["attention_mean"] = np.mean(np.stack(attention_means), axis=0).tolist()
        """
            
        return results

    def train_loop(self, epochs, steps_per_epoch=None, progress_bar=True):
        for epoch in range(1, epochs + 1):
            print(f"\n=== Epoch {epoch}/{epochs} ===")
            self.enable_attention = (self.attention_warmup_epoch is not None and epoch >= self.attention_warmup_epoch)
            if self.enable_attention and not self._attention_warmup_done:
                print("[Check] Pool layer trainable:", self.model.pool.trainable_variables)
                self.retrace_all()
                
                self._attention_warmup_done = True

            old_weights = {
                "feature_extractor": [tf.identity(w) for w in self.model.eegnet.trainable_variables],
                "pool": [tf.identity(w) for w in self.model.pool.trainable_variables],
                "classifier": [tf.identity(w) for w in self.model.classifier.trainable_variables],
            }

            step_iter = enumerate(self.train_dataset.take(steps_per_epoch)) if steps_per_epoch else enumerate(self.train_dataset)
            if progress_bar:
                from tqdm import tqdm
                step_iter = tqdm(step_iter, total=steps_per_epoch)

            for step, batch in step_iter:
                try:
                    self.train_step(batch)
                except tf.errors.ResourceExhaustedError:
                    print(f"[Epoch {epoch} | Step {step}] Skipping batch due to memory exhaustion.")
                    tf.keras.backend.clear_session()
                    continue


            val_iter = tqdm(enumerate(self.val_dataset), desc="Validation")
            for step, batch in val_iter:
                try:
                    self.val_step(batch)
                except tf.errors.ResourceExhaustedError:
                    print(f"[Epoch {epoch} | Val Step {step}] Skipping batch due to memory error.")
                    tf.keras.backend.clear_session()
                    import gc
                    gc.collect()
                    continue

            print(f"Train Loss = {self.train_loss_metric.result():.4f}, "
                  f"Val Loss = {self.val_loss_metric.result():.4f}")

            metrics = self.eval_step()
            print("Val F1: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(
                metrics["f1"], metrics["accuracy"], metrics["precision"], metrics["recall"]
            ))

            for level in ["level1", "level2", "level3"]:
                if metrics.get(f"{level}_f1") is not None:
                    print(f"{level.upper()} - F1: {metrics[f'{level}_f1']:.4f}, "
                          f"Acc: {metrics[f'{level}_acc']:.4f}, "
                          f"Prec: {metrics[f'{level}_precision']:.4f}, "
                          f"Recall: {metrics[f'{level}_recall']:.4f}")

            # Print entropy stats if available
            if "attention_entropy_mean" in metrics:
                print("Attention Entropy — Mean: {:.4f}, Std: {:.4f}".format(
                    metrics["attention_entropy_mean"], metrics["attention_entropy_std"]
                ))

            # Optionally print mean attention vector (e.g., mean importance per epoch index)
            if "attention_mean" in metrics:
                mean_attn = np.array(metrics["attention_mean"])
                print("Mean Attention Weights (across eval set):", np.round(mean_attn, 3))

            if self.ckpt_manager and (epoch % self.ckpt_interval == 0):
                save_path = self.ckpt_manager.save()
                print(f"Checkpoint saved at {save_path}")

            self.metric_history.append({
                "val_loss": self.val_loss_metric.result().numpy(),
                **metrics
            })

            new_weights = {
                "feature_extractor": [tf.identity(w) for w in self.model.eegnet.trainable_variables],
                "pool": [tf.identity(w) for w in self.model.pool.trainable_variables],
                "classifier": [tf.identity(w) for w in self.model.classifier.trainable_variables],
            }

            print_weight_updates(old_weights, new_weights)

            # Anneal the loss function if necessary
            if self.anneal_interval:
                if epoch%self.anneal_interval == 0:
                    self.loss_fn.anneal_temperature(self.loss_fn.temperature * self.anneal_coeff)


    def get_metrics(self, mode="best"):
        if mode == "all":
            return self.metric_history

        if not self.metric_history:
            return None

        best_epoch = max(self.metric_history, key=lambda m: m.get("f1", -1))
        return best_epoch

def print_weight_updates(old_weights_dict, new_weights_dict):
    print("Weight update norms:")
    for part_name, old_vars in old_weights_dict.items():
        new_vars = new_weights_dict.get(part_name, None)

        if not old_vars or not new_vars:
            print(f"  {part_name}: no trainable variables to report")
            continue

        if len(old_vars) != len(new_vars):
            print(f"  {part_name}: variable count mismatch (old: {len(old_vars)}, new: {len(new_vars)})")
            continue

        diffs = []
        for old, new in zip(old_vars, new_vars):
            diff = tf.norm(new - old).numpy()
            diffs.append(diff)

        if not diffs:
            print(f"  {part_name}: no weight differences found")
            continue

        max_diff = np.max(diffs)
        mean_diff = np.mean(diffs)
        print(f"  {part_name}: max update {max_diff:.6f}, mean update {mean_diff:.6f}")
