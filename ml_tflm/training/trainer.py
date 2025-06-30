import tensorflow as tf
import numpy as np
import tqdm
import ml_tflm.training.train_utils as utils

class Trainer:
    def __init__(self, model, loss_fn, optimizer, evaluator,
                 train_dataset, val_dataset, test_dataset=None,
                 dataset_keys=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.dataset_keys = dataset_keys or {
            "inputs": "data",
            "mask": "attention_mask",
            "targets": "internal_label"
        }

        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss_metric = tf.keras.metrics.Mean(name="val_loss")

    @tf.function
    def train_step(self, batch):
        x = batch[self.dataset_keys["inputs"]]
        mask = batch.get(self.dataset_keys["mask"], None)
        y = batch[self.dataset_keys["targets"]]

        with tf.GradientTape() as tape:
            outputs = self.model(x, attention_mask=mask, training=True)
            loss = self.loss_fn(y, outputs["logits"])
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_loss_metric.update_state(loss)

    @tf.function
    def val_step(self, batch):
        x = batch[self.dataset_keys["inputs"]]
        mask = batch.get(self.dataset_keys["mask"], None)
        y = batch[self.dataset_keys["targets"]]

        outputs = self.model(x, attention_mask=mask, training=False)
        loss = self.loss_fn(y, outputs["logits"])
        self.val_loss_metric.update_state(loss)

    def eval_step(self):
        all_preds, all_targets = [], []
        for batch in self.val_dataset:
            x = batch[self.dataset_keys["inputs"]]
            mask = batch.get(self.dataset_keys["mask"], None)
            y = batch[self.dataset_keys["targets"]]

            logits = self.model(x, attention_mask=mask, training=False)["logits"]
            all_preds.append(logits)
            all_targets.extend(y.numpy().tolist())

        pred_tensor = tf.concat(all_preds, axis=0)
        target_tensor = tf.convert_to_tensor(all_targets, dtype=tf.int32)
        return self.evaluator.evaluate(pred_tensor, target_tensor)

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
