import os

DATA_PATH = os.getenv("DATA")

data_fif = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/fif"
filters_config_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/workflow/config/filter_settings.yaml"

# =======================
# 🔄  Resample EEG data
# =======================
rule resample_data:
    input:
        fif = data_fif + "/{sample}_filtered.fif",
        config=filters_config_path
    output:
        fif = data_fif + "/{sample}_resampled.fif"
    params:
        script="scripts/resample_data.py"
    conda:
        "../envs/convert_to_bipolar_montage.yaml"
    shell:
        """
        echo "🔄  Resampling {input} → {output}"
        mkdir -p $(dirname {output.fif})
        python {params.script} "{input.fif}" "{output.fif}" "{input.config}" 
        """