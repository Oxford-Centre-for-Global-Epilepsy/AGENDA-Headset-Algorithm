# Misc Eval

This folder contains relevant codes for evaluating miscellaneous aspects of training workflow. They can be divided into following categories:

## Training Result Evaluators

## Dataset Evaluators

- **dataset_checker.py**: reviews the freuqency content of the processesed and saved .h5 dataset
- **edf_checker.py**: reviews the frequency content of the raw .edf dataset and experiments filter configurations

## Result Evaluators

- **kfold_review.py**: reports and visualizes the k-fold protocol training-validation results
- **optimization_review.py**: reviews the performance of different architecture results derived from Optuna optimization
- **result_cleanup.py**: screens the results and clean up the ones without target params (thus outdated)