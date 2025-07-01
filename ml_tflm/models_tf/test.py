import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from ml_tflm.models_tf.classifier_QAT import get_eegnet_qat, EEGNetFlatClassifierQAT
import tensorflow as tf

if __name__ == "__main__":
    pass