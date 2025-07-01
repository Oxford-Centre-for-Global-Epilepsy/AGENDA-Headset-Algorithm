import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from ml_tflm.models_tf.classifier_QAT import get_classifier_QAT
import tensorflow as tf

if __name__ == "__main__":
    eegnet_args = {
        "num_channels": 21,
        "num_samples": 256,
        "F1": 8,
        "D": 2,
        "F2": 16,
        "dropout_rate": 0.25,
        "kernel_length": 64,
        "activation": tf.nn.relu
    }

    pooling_args = {
        "hidden_dim": 64,
        "activation": tf.nn.tanh
    }

    qat_model = get_classifier_QAT(eegnet_args, pooling_args)
    
    print("Quantisation Successful!")