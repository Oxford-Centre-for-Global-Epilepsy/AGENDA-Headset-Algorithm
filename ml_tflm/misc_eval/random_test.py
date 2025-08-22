import tensorflow as tf
import keras  # stand-alone, if installed

from ml_tflm.models_tf.classifier_QAT import HardSwish

print("is tf.keras layer:", isinstance(HardSwish(), tf.keras.layers.Layer))
print("is standalone keras layer:", isinstance(HardSwish(), keras.layers.Layer))
