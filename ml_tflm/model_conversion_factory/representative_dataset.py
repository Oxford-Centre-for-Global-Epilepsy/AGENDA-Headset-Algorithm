import tensorflow as tf
import random
import numpy as np
from ml_tflm.dataset.eeg_dataset import EEGRecordingTFGenerator
from ml_tflm.training.train_utils import load_label_config

class EEGRepresentativeDataset:
    def __init__(self, tf_dataset, total_epochs=100, epochs_per_subject=3):
        """
        tf_dataset: tf.data.Dataset from EEGRecordingTFGenerator.as_dataset()
        total_epochs: total number of individual epochs to collect
        """
        self.epochs = []
        all_batches = list(tf_dataset)
        num_subjects = len(all_batches)

        max_available_epochs = num_subjects * epochs_per_subject
        if max_available_epochs >= total_epochs:
            # Limit the number of subjects used
            required_subjects = (total_epochs + epochs_per_subject - 1) // epochs_per_subject
            selected_batches = random.sample(all_batches, required_subjects)
            n_epochs_per_subject = epochs_per_subject
        else:
            # Use all available subjects and distribute needed epochs evenly
            selected_batches = all_batches
            n_epochs_per_subject = (total_epochs + num_subjects - 1) // num_subjects

        for batch in selected_batches:
            if len(self.epochs) >= total_epochs:
                break

            full_data = batch["data"].numpy()[0]  # [E, C, T]
            E = full_data.shape[0]
            n = min(E, n_epochs_per_subject, total_epochs - len(self.epochs))
            indices = random.sample(range(E), n)

            for idx in indices:
                epoch = full_data[idx]  # [C, T]
                epoch = np.expand_dims(epoch, axis=0)   # [1, C, T]
                epoch = np.expand_dims(epoch, axis=-1)  # [1, C, T, 1]
                self.epochs.append(epoch.astype(np.float32))

                if len(self.epochs) >= total_epochs:
                    break

        print(f"Collected {len(self.epochs)} epochs (requested {total_epochs})")

    def generator(self):
        for x in self.epochs:
            yield {"input_1": x}

    def __len__(self):
        return len(self.epochs)
    
def get_rep_dataset(h5_file_path, dataset_name, total_epochs=100, epochs_per_subject=3):
    label_config = load_label_config("ml_tflm/training/label_map.JSON")
    tf_dataset = EEGRecordingTFGenerator(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config
    ).as_dataset(batch_size=1, shuffle=True)
    return EEGRepresentativeDataset(tf_dataset, total_epochs, epochs_per_subject)
    
if __name__ == "__main__":
    # Construct the representative dataset
    rep_dataset = get_rep_dataset(h5_file_path="ml_tflm/dataset/sample_data/anyu_dataset_south_africa_monopolar_standard_10_20.h5",
                                  dataset_name="anyu_dataset_south_africa_monopolar_standard_10_20",
                                  total_epochs=100, epochs_per_subject=3)

    # Verify shape and dtype of generated samples
    for i, sample in enumerate(rep_dataset.generator()):
        print(f"Epoch {i+1}: shape={sample.shape}, dtype={sample.dtype}")
        if i >= 4:
            break