import os

DATA_PATH = os.getenv("DATA")

data_temp = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/temp"
filters_config_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/workflow/config/filter_settings.yaml"

# =======================
# 🔄  Resample EEG data
# =======================
rule resample_data:
    input:
        fif = data_temp + "/{montage_type}/{montage_name}/{site}/{data_label}/{sample}_filtered.fif",
        config=filters_config_path
    output:
        fif = temp(data_temp + "/{montage_type}/{montage_name}/{site}/{data_label}/{sample}_resampled.fif")
    params:
        script="scripts/data_preprocessing/resample_data.py"
    conda:
        "../../envs/data_preprocessing.yaml"
    shell:
        """
        echo "Resampling {input} → {output}"
        mkdir -p $(dirname {output.fif})
        python {params.script} "{input.fif}" "{output.fif}" "{input.config}" 
        """