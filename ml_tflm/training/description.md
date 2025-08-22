# Models TF

This folder contains relevant codes for training, and configuration files for the model. They can be divided into following categories:

## Training codes

- **train_exp.py**: this function defines the one-pass training code equiped with Hydra management
- **train_mil.py**: this function defines a training procedure that uses more classic MIL workflow (pooling logits instead of features)
- **train_opt.py**: this function defines the architecture optimization procedure with Optuna's default parameter search algorithm
- **train_QAT.py**: this function defines the QAT training procedure without Hydra parameter management (due to many incompatibilities, must be toggled independently)
- **train_val.py**: this function defines the validation training procedure that passes through all the target parameters with k-fold validation

## Trainer and Components

- **trainer.py**: contains the trainer class that manages train, validation, metrics tracking, reporting objects

- **loss.py**: contains definition of many loss functions
- **metrics.py**: contains the function for metric evaluation (on casted model outputs)

## Training Helpers

- **cast_prediction.py**: this script contains casters that interprets the output of model (often logits) into classification labels
- **train_utils.py**: this script contains utility functions that assists the training scripts

