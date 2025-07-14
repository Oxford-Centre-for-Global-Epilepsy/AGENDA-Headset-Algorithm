import tensorflow as tf

def get_pre_train_step(model_dict, optimizer_dict, loss_fn):
    @tf.function
    def pre_train_step(batch):
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

        # Compute and apply gradients
        vars_feat = model_dict["feature_extractor"].trainable_variables
        vars_proj = model_dict["projector"].trainable_variables
        grads = tape.gradient(loss, vars_feat + vars_proj)

        grads_feat = grads[:len(vars_feat)]
        grads_proj = grads[len(vars_feat):]

        optimizer_dict["feature_extractor"].apply_gradients(zip(grads_feat, vars_feat))
        optimizer_dict["projector"].apply_gradients(zip(grads_proj, vars_proj))

        return loss
    return pre_train_step

@tf.function
def pre_val_step(model_dict, loss_fn, batch):
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

    return loss

def get_cls_train_step(model, optimizer, loss_fn):
    @tf.function
    def cls_train_step(batch):
        model_inputs = {
            "x": batch["data"],
            "attention_mask": batch["attention_mask"],
        }
        
        model_targets = {
            "targets": batch["internal_label"],
        }

        with tf.GradientTape() as tape:
            outputs = model(**model_inputs, training=True)
            loss = loss_fn(y_pred=outputs, y_true=model_targets)
            loss += tf.add_n(model.losses)
        
        grads = tape.gradient(loss, model.trainable_variables)
        grads_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
        optimizer.apply_gradients(grads_vars)

        return loss
    return cls_train_step

def get_cls_val_step(model, loss_fn):
    @tf.function
    def cls_val_step(batch):
        model_inputs = {
            "x": batch["data"],
            "attention_mask": batch["attention_mask"],
        }
        
        model_targets = {
            "targets": batch["internal_label"],
        }

        outputs = model(**model_inputs, training=False)
        loss = loss_fn(y_pred=outputs, y_true=model_targets)
        loss += tf.add_n(model.losses)
        return loss
    return cls_val_step
