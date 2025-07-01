import tensorflow as tf
import tensorflow_model_optimization as tfmot
from ml_tflm.training.train_utils import get_checkpoint_manager, maybe_restore_checkpoint

class Trainer:
    def __init__(self, model, loss_fn, optimizer, evaluator,
                 train_dataset, val_dataset, model_input_lookup, model_target_lookup, test_dataset=None,
                 save_ckpt=False, ckpt_interval=1, ckpt_save_dir="./checkpoints",
                 load_ckpt=False, ckpt_load_dir=None):
           
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

        # Create the checkpoint object regardless of loading or saving
        if load_ckpt or save_ckpt:
            self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        # If loading is enabled, try to restore from ckpt_load_dir
        if load_ckpt:
            if ckpt_load_dir is None:
                raise ValueError("Checkpoint load requested, but ckpt_load_dir is None.")
            
            _, load_manager = get_checkpoint_manager(self.model, self.optimizer, ckpt_load_dir)
            restored = maybe_restore_checkpoint(self.ckpt, load_manager)

            if restored:
                print(f"Restored checkpoint from {load_manager.latest_checkpoint}")
            else:
                print("No checkpoint found — training will start from scratch.")

        # If saving is enabled, create save manager using ckpt_save_dir
        if save_ckpt:
            self.ckpt_interval = ckpt_interval
            self.ckpt_save_dir = ckpt_save_dir
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_save_dir, max_to_keep=5)

    @tf.function
    def train_step(self, batch):
        model_inputs = {k: batch[v] for k, v in self.model_input_lookup.items()}
        model_targets = {k: batch[v] for k, v in self.model_target_lookup.items()}

        # Forward and loss
        with tf.GradientTape() as tape:
            outputs = self.model(**model_inputs, training=True)
            loss = self.loss_fn(y_pred = outputs, y_true = model_targets)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_loss_metric.update_state(loss)

    @tf.function
    def val_step(self, batch):
        model_inputs = {k: batch[v] for k, v in self.model_input_lookup.items()}
        model_targets = {k: batch[v] for k, v in self.model_target_lookup.items()}

        # Forward pass and loss computation
        outputs = self.model(**model_inputs, training=False)
        loss = self.loss_fn(y_pred = outputs, y_true = model_targets)

        self.val_loss_metric.update_state(loss)

    def eval_step(self):
        all_preds, all_targets = [], []
        for batch in self.val_dataset:
            model_inputs = {k: batch[v] for k, v in self.model_input_lookup.items()}

            y = batch["internal_label"]

            output = self.model(**model_inputs, training=False)
            all_preds.append(output)
            all_targets.extend(y.numpy().tolist())

        target_tensor = tf.convert_to_tensor(all_targets, dtype=tf.int32)
        return self.evaluator.evaluate(all_preds, target_tensor)

    def train_loop(self, epochs, steps_per_epoch=None, progress_bar=True):
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch} / {epochs}")
            self.train_loss_metric.reset_state()
            self.val_loss_metric.reset_state()

            step_iter = enumerate(self.train_dataset.take(steps_per_epoch)) if steps_per_epoch else enumerate(self.train_dataset)
            if progress_bar:
                from tqdm import tqdm
                step_iter = tqdm(step_iter, total=steps_per_epoch)

            for step, batch in step_iter:
                self.train_step(batch)

            for batch in self.val_dataset:
                self.val_step(batch)

            print(f"Train Loss = {self.train_loss_metric.result():.4f}, "
                  f"Val Loss = {self.val_loss_metric.result():.4f}")
            
            metrics = self.eval_step()
            print("Val F1: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(
                metrics["f1"], metrics["accuracy"], metrics["precision"], metrics["recall"]
            ))

            if self.ckpt_manager and (epoch % self.ckpt_interval == 0):
                save_path = self.ckpt_manager.save()
                print(f"Checkpoint saved at {save_path}")

