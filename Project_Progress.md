# EEG Hierarchical Classification and Interpretability Framework

## Overview

This project is a PyTorch-based pipeline for classifying EEG recordings into a 3-level hierarchical label structure. The model is based on EEGNet and supports different temporal pooling strategies: mean pooling, attention pooling, and transformer-based pooling. It supports stratified data splits, k-fold cross-validation, feature projection, attention visualization, and saliency maps for interpretability.

## Dataset

* EEG recordings are stored in HDF5 format.
* Each subject has multiple epochs (E), channels (C), and time samples (T).
* Labels are structured hierarchically:

  * Level 1: Neurotypical vs Epileptic
  * Level 2: Focal vs Generalized
  * Level 3: Left vs Right
* Channel names are retrieved from HDF5 attributes and stored in the `EEGRecordingDataset` class.

## Model

### Architecture:

* Base: EEGNet
* Temporal Pooling Options:

  * MeanPooling
  * AttentionPooling
  * TransformerPooling
* Multi-head classifier for 3-level hierarchy
* Outputs include logits at each level, pooled features, and optional attention weights

## Evaluation

Implemented in `evaluate.py`, the evaluation pipeline performs:

* Confusion matrix computation (per level)
* Attention weight visualization (per level)
* Multichannel attention overlay on EEG traces
* Feature projection using UMAP or t-SNE (per level)
* Saliency map generation via vanilla gradients

## Interpretability

### Attention Visualizations

* Attention weights are plotted across time for each class at each hierarchy level
* Heatmap overlays are generated on multichannel EEG recordings

### Feature Projection

* Features are extracted using the model's pooled representations
* UMAP is used to reduce features to 2D
* Color-coded scatter plots show class separability at each level

### Saliency Maps

* Vanilla gradient saliency maps are computed per epoch
* Output is visualized as a heatmap with rows as channels and columns as time
* Saliency plots include predicted and true class labels (where available)

## Configuration

* Configurations are defined in YAML and merged with CLI overrides
* Options include model hyperparameters, dataset path, output directory, pooling type, seed, and k-fold index

## Tools and Libraries

* PyTorch, NumPy, Matplotlib, Seaborn, Scikit-learn
* UMAP (`umap-learn`) and optionally t-SNE from `sklearn.manifold`
* OmegaConf for configuration management

## Status

All components (model, evaluation, visualization) are working. Current model outputs attention overlays, confusion matrices, feature projections, and saliency heatmaps with labeled axes and diagnostic information. Debug messages confirm proper functioning of attention and saliency plotting.

## Next Steps

* Extend saliency to averaged plots across subjects
* Add support for guided backprop or Grad-CAM
* Enable interactive plotting with Plotly or Bokeh
* Track all outputs in a `reports/` or `outputs/` metadata structure
