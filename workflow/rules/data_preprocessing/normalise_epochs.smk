import os

DATA_PATH = os.getenv("DATA")

data_temp = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/temp"

# =======================
# 🔄  Normalise the Epoched EEG data
# =======================
rule normalise_epoched_data:
    input:
        fif = data_temp + "/{montage_type}/{montage_name}/{site}/{data_label}/{sample}_epoched.fif"
    output:
        fif = temp(data_temp + "/{montage_type}/{montage_name}/{site}/{data_label}/{sample}_normalised.fif")
    params:
        script="scripts/data_preprocessing/normalise_epoched_data.py"
    conda:
        "../../envs/data_preprocessing.yaml"
    shell:
        """
        echo "Normalising {input} → {output}"
        mkdir -p $(dirname {output.fif})
        python {params.script} "{input.fif}" "{output.fif}" 
        """