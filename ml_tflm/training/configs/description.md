# CONGFIGS

This folder contains the configuration for the neural network architecture. 

## Component

This folder contains all the building blocks of classifier model, namely feature extractor, pooling layer, and classifier head.

### Feature Extractor

Feature extractor contains variants for the EEGNet, currently the two existing versions are:

- **EEGNet**: is almost the vannilla EEGNet with some modifications in filter configurations and activation
- **EEGNetMultiscale**: is the EEGNet with modification to the first temporal filtering stage (with multiple branches with different dilation in temporal filtering stage). This has been proved not very useful and slow to train. 

#### Parameters Specification:

- eegnet: contains the configurations for eegnet
    - F1: the number of temporal filters
    - D: the depth mutiplier for depthwise convolution
    - F2: the number of output filters
    - dropout_rate: the dropout layer rate used in the eegnet
    - bottleneck_dim: is an optional mlp layer after the flattening - useful when the output filter number is high, set to 0 to skip
    - kernel_length: the length of temporal kernel
    - temporal_type: the type of temporal filtering used, options are "vanilla" or "multiscale"
    - activation: the activation functions used for eegnet, options include "relu", "leaky_relu", "elu", etc. For more info check training.train_utils.get_activation(name)
    - l2_weight: the value of l2_weight used for the eegnet
    - norm_layer: the type of normalization layer used for the eegnet, options include "BATCH" (batch normalization), "LAYER" (layer normalization), and "NONE" (no normalization)

- model_input_lookup: contains the lookup keys for the trainer class, please do not change unless you plan to modify the trainer or dataset.

### Pooling Layer

Pooling layer contains many variants that we have experimented, not going to go through all the details here, please check models_tf.attention_pooling and models_tf.experimental_pooling if interested. 

#### Parameter Specification:

Pooling layer contains many different parameters based on the type of pooling layer used. Here only the most useful configurations are explained:

If using the **attention pooling skeleton**, following parameters will be useful (attention pooling skeleton is not necessarily attention based, it has been hacked to fit in many interesting variants).

- pool:
    - scorer: the type of scorer used for each sample, this includes "AVG" (for average pooling), "MLP" (for classic attention pooling), "Ablation" (for ablation pooling), etc. Check models_tf.attention_pooling if interested.
    - num_heads: the number of replica of pooling layers (each independently trainable), pooled vectors are concatenated before classifier

- loss: this section could be skipped if you do not want to use entroy loss
    - target: defines the loss function used, please do not change unless you're confident

    - label_to_heads: defines for each label, which of the heads are penaltized for entropy
    - lambda_entropy: defines the weighing of entropy loss

If using other **experimental pooling**, check out the models_tf.experimental_pooling. The following parameter will be useful:

- pool:
    - pool: defines the type of pool you are willing to use, like "GRU". To add custom pooling, write the implementation and add the name to POOLING_REGISTRY in models_tf.attention_pooling

### Classifier Head

Classifier head contains the classifier heads variants for different classification setups, also including the loss function and output caster (which binds).

- **Binary**: classifies only epileptic or non-epileptic with two logits at output
- **Flat**: classifiers the four classes in one-hot encoding
- **Hierarchical**: classifies the four classes in hierarchical order, each with two logits
- **LeanHierarchical**: classifiers the four classes in hierarchical order, each with one logits

No change to the parameters is recommended

### Dataset

Dataset contains the relevant configurations for using .h5 datasets. Considering that we cannot upload the converted dataset to github, it is recommended to customize new dataset configurations based on following parameter specification:

#### Parameter Specification:

- h5_path: defins the path of the .h5 file relative to the root of folder
- name: defines the name of dataset to look within the .h5 file

- ablation: defines the list of electrodes to remove from dataset

### Optimizer

Optimizer contains optimizer used in the training process, for the time being it's a simplest ADAM optimizer with only learning rate configurable.

### Training

Training contains a lot of training-related parameters, and some misc parameters (that sounds like they should show up in somewhere else).

#### Parameter Specification:

- val_frac: the fraction of training-validation set used for evaluation, also implies the number of folds used for validation
- test_frac: the fraction of total dataset used for testing, isolated from the training-validation set
- k_fold: whether or not to use the k-fold
- stratify: whether or not to use stratified division (by subject label)
- internal_label_cap_keys: the labels that need to be capped in number when loading
- internal_label_cap_values: the number of subjects by label that need to be capped
- class_balance: whether or not to use class balancing weights in loss

- mirror_flag: whether or not to flip the subject of left/right focal to form augmented focal data

- batch_sz: the batch size of loaded data
- chunk_size: the size of continuous chunk of segments sampled from whole recording

- epochs: the nubmer of epochs
- steps_per_epoch: the number of steps (batches) per epoch

- *attention_warmup_epoch*: Not useful, please just keep it there

- anneal_interval: controlls how often the loss function is annealed. Only valid for the flat classifier, not recommended to change
- anneal_coeff: controlls coefficient of annealing each time it's requested

- save_ckpt: whether or not we save ckpt
- ckpt_interval: how many epochs we wait until saving ckpt for once
- ckpt_save_dir: the directory to which the ckpt is saved

- load_ckpt: whether or not we load ckpt
- ckpt_load_dir: the directory from which the ckpt is loaded

- metric_save_dir: the directory to which the training metrics is saved