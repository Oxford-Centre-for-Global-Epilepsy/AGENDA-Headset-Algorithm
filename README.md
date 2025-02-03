# AGENDA Headset Algorithm

🚀 Development of EEG-based classification models for epilepsy diagnosis as part of Workstream 2 of the AGENDA project. Initial algorithm work focuses on fine-tuning EEGNet for EEG-based classification using Snakemake and DVC for reproducible data preprocessing and model training.

---

## 📌 Project Overview
This project fine-tunes EEGNet, a compact convolutional neural network for EEG data classification. The goal is to classify EEG recordings as neurotypical, epileptic (generalized or focal), or abnormal.

### 🛠 Features
✅ EEGNet Architecture - Pre-trained and fine-tuned for EEG classification.
✅ DVC (Data Version Control) - Efficient data tracking and storage.
✅ Snakemake Workflow - Fully automated pipeline for data preprocessing and model training.
✅ Multi-Site Generalization - Supports EEG data from multiple recording sites.

---

## 📂 Project Structure
`
project-root/
├── data/                # DVC-managed EEG data
│   ├── raw/             # Original EEG files (e.g., .edf, .hdf5)
│   ├── processed/       # Preprocessed EEG data
│   └── external/        # External datasets
├── notebooks/           # Jupyter notebooks for data exploration
├── src/                 # Source code for the project
│   ├── data/            # Data preprocessing scripts
│   ├── models/          # EEGNet model definition & modifications
│   ├── training/        # Fine-tuning & training scripts
│   └── evaluation/      # Model evaluation scripts
├── scripts/             # Utility scripts for data processing
├── workflow/            # Snakemake pipeline
│   ├── Snakefile        # Main workflow file
│   ├── rules/           # Modular Snakemake rules
│   ├── envs/            # Conda environments for each rule
├── results/             # Outputs: trained models, logs, reports
├── tests/               # Unit tests
├── docs/                # Documentation
│   ├── installation.md
│   ├── data_preprocessing.md
│   ├── model_training.md
│   ├── evaluation.md
│   ├── dvc_setup.md
│   ├── contributing.md
│   ├── references.md
│   └── changelog.md
├── environment.yml      # Conda environment file
├── dvc.yaml             # DVC pipeline
├── params.yaml          # Hyperparameter configurations
├── .gitignore           # Ignore large files & temporary logs
└── README.md            # This file
`
---

## 🛠 Installation Guide

### 1️⃣ Clone the Repository
`
git clone https://github.com/yourusername/yourproject.git
cd yourproject
`

### 2️⃣ Set Up Conda Environment
`
conda env create -f environment.yml
conda activate eegnet_project
`

### 3️⃣ Configure DVC (Data Version Control)
`
dvc pull  # Pull dataset from remote storage
`
_(Ensure dvc is properly configured; see docs/dvc_setup.md)_

### 4️⃣ Install Additional Dependencies (if needed)
`
pip install -r requirements.txt
`
---
## 🚀 Usage

### 1️⃣ Preprocess EEG Data
`
snakemake --cores 4 --use-conda
`
_(Processes raw EEG data and prepares it for training.)_

### 2️⃣ Train & Fine-Tune EEGNet
`
python src/training/train.py --epochs 50 --batch_size 32
`
_(Fine-tunes EEGNet on preprocessed EEG data.)_ 

### 3️⃣ Evaluate Model Performance
`
python src/evaluation/evaluate.py --model results/eegnet_model.pth
`
_(Generates accuracy, loss, confusion matrix, and ROC curves.)_

### 4️⃣ Version Control for Data
`
dvc add data/processed/
git add data/processed.dvc
git commit -m "Updated processed EEG data"
dvc push
`
_(Ensures reproducibility by tracking data changes.)_

---

## 📊 Results & Model Performance
- __Baseline EEGNet Accuracy__: XX%
- __Fine-Tuned EEGNet Accuracy__: XX%
- __Generalized Performance Across Sites__: XX%
- __ROC-AUC Score__: XX%

---

## 📝 Documentation
📖 Full documentation available in docs/

- [Installation Guide](docs/installation.md)
- [Data Preprocessing](docs/data_preprocessing.md)
- [Model Training](docs/model_training.md)
- [Evaluation Metrics](docs/evaluation.md)
- [DVC Setup](docs/dvc_setup.md)

---

## 🤝 Contributing
Contributions are welcome! Please see docs/contributing.md for guidelines.

---

## 🔗 References
- [EEGNet Paper](https://arxiv.org/abs/1611.08024)
- [DVC Documentation](https://dvc.org/doc)
- [Snakemake Documentation](https://snakemake.readthedocs.io/en/stable/)
- [PyPrep Documentation](https://pyprep.readthedocs.io/en/stable/)

--- 

## 📅 Changelog

See docs/changelog.md for version updates.

---

## 📜 License
📝 MIT License - Free to modify and use for research and non-commercial applications.

--- 

## 🎯 Next Steps
- 📌 Optimize EEGNet hyperparameters for improved classification.
- 📌 Integrate attention mechanisms to enhance feature learning.
- 📌 Deploy the trained model for real-time EEG analysis.

---

🔹 __Author__: John E. Fleming
🔹 __Affiliation__: University of Oxford
🔹 __Contact__: john.fleming@ndcn.ox.ac.uk

---

🚀 Happy Coding & Good Luck! 🎉

---