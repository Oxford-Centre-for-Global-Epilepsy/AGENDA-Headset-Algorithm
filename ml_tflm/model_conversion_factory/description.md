# Model Conversion Factory

This folder contains relevant codes for converting tensorflow models into quantised (or not) micro models. They can be divided into following categories:

## Converter

- **convert.py**: contains codes for calling everything and execute conversion

## Conversion Helpers

- **Conversion_utils.py**: contains codes for doing conversion from tensorflow model to lite model and micro models
- **representative_dataset.py**: contains codes for preparing representative dataset that guides input quantisation

## Model Inspectors

- **lite_inference.py**: contains codes that loads the converted lite model and output inference results to be compared with Teensy inference outputs
- **lite_inspector.py**: contains codes that loads the converted lite model and print out the model input/output, layer and operator information
