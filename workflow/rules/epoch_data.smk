import os

DATA_PATH = os.getenv("DATA")

data_fif = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/fif"
data_epochs_config_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/workflow/config/epoch_settings.yaml"

# =======================
# 🔄  Epoch the EEG data
# =======================
rule epoch_data:
    input:
        fif = data_fif + "/{montage}/{sample}_bipolar.fif",
        config=data_epochs_config_path
    output:
        fif = data_fif + "/{montage}/{sample}_epoched.fif"
    params:
        script="scripts/epoch_data.py"
    conda:
        "../envs/convert_to_bipolar_montage.yaml"
    shell:
        """
        echo "🔄  Epoching {input} → {output}"
        mkdir -p $(dirname {output.fif})
        python {params.script} "{input.fif}" "{output.fif}" "{input.config}" 
        """