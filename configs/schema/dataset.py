from dataclasses import dataclass

@dataclass
class DatasetSplitConfig:
    cv_dataset_folds_dir: str
    fold_index: int
    k_folds: int

@dataclass
class FinalTestConfig:
    final_eval: bool
    test_dataset_dir: str
    testset_file_name: str

@dataclass
class DatasetConfig:
    project_name: str = "AGENDA"
    site_name: str = "South Africa"
    dataset_name: str = "EEG"
    dataset_path: str = "${oc.env:DATA}/AGENDA-Headset-Algorithm/data/experiment_datasets/electrode_ablation/south_africa/monopolar/standard_10_20/processed_dataset.h5"
    cv_datasets: DatasetSplitConfig = DatasetSplitConfig(
        cv_dataset_folds_dir="${oc.env:DATA}/AGENDA-Headset-Algorithm/data/experiment_datasets/electrode_ablation/south_africa/monopolar/standard_10_20/dataset_splits",
        fold_index=0,
        k_folds=5
    )
    final_test_set: FinalTestConfig = FinalTestConfig(
        final_eval=False,
        test_dataset_dir="${oc.env:DATA}/AGENDA-Headset-Algorithm/data/experiment_datasets/electrode_ablation/south_africa/monopolar/standard_10_20/dataset_splits",
        testset_file_name="test_split.json"
    )
