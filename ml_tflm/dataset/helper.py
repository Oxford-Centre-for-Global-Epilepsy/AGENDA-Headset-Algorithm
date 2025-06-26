import h5py

with h5py.File("ml_tflm/dataset/sample_data/anyu_dataset_south_africa_monopolar_standard_10_20.h5", "r") as f:
    print("Top-level keys (dataset root names):", list(f.keys()))
