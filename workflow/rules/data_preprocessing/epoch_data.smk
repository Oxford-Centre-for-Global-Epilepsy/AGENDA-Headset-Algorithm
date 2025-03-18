import os

DATA_PATH = os.getenv("DATA")

data_fif = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/fif"
data_epochs_config_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/workflow/config/epoch_settings.yaml"

# =======================
# 🔄  Epoch the EEG data
# =======================
rule epoch_data:
    input:
        fif = data_fif + "/{montage_type_montage_name[0]}/{montage_type_montage_name[1]}/{site}/{sample}_bipolar.fif",
        config=data_epochs_config_path
    output:
        fif = temp(data_fif + "/{montage_type_montage_name[0]}/{montage_type_montage_name[1]}/{site}/{sample}_epoched.fif")
    params:
        script="scripts/data_preprocessing/epoch_data.py"
    conda:
        "../../envs/data_preprocessing.yaml"
    shell:
        """
        echo "🔄  Epoching {input} → {output}"
        mkdir -p $(dirname {output.fif})
        python {params.script} "{input.fif}" "{output.fif}" "{input.config}" 
        """