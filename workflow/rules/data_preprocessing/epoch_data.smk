import os

DATA_PATH = os.getenv("DATA")

data_temp = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/temp"
data_epochs_config_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/workflow/config/epoch_settings.yaml"

# =======================
# 🔄  Epoch the EEG data
# =======================
rule epoch_data:
    input:
        fif = data_temp + "/{montage_type}/{montage_name}/{site}/{data_label}/{sample}_headset_montage.fif",
        config=data_epochs_config_path
    output:
        fif = data_temp + "/{montage_type}/{montage_name}/{site}/{data_label}/{sample}_epoched.fif"
    params:
        script="scripts/data_preprocessing/epoch_data.py"
    conda:
        "../../envs/data_preprocessing.yaml"
    touch: True
    shell:
        """
        echo "🔄  Epoching {input} → {output}"
        mkdir -p $(dirname {output.fif})
        set -ex
        python {params.script} "{input.fif}" "{output.fif}" "{input.config}"
        """