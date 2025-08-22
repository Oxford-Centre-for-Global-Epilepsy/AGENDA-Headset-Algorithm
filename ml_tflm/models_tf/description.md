# Models TF

This folder contains relevant codes for defining Tensorflow models. They can be divided into following categories:

## Normal Models

- **EEGNet_stack.py** defines the EEGNet-based model that will be used in normal training workflow

- **feature_extractor.py** defines the feature extractor that is a component of the EEGNet-based model
- **attention_pooling.py** defines the attention pooling layer that is a component of the EEGNet-based model
- **experimental_pooling.py** defines experimental pooling layer that cannot be defined using the attnetion pooling skeleton, it is a component of the EEGNet-based model that could replace attention pooling
- **classification_head.py** defines the classification head that is a component of the EEGNet-based model

## QAT Models

- **Classifier_QAT.py** defines the EEGNet-based model with custom H-Swish layer and QAT relevant definition