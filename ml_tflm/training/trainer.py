import tensorflow as tf
import tensorflow_model_optimization as tfmot

class Trainer:
    def __init__(self, model, loss_fn, optimizer, evaluator,
                 train_dataset, val_dataset, model_input_lookup, model_target_lookup, test_dataset=None, QAT=False):
        if QAT == True:
            model = tfmot.quantization.keras.quantize_annotate_model(model)
           
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

