import os

DATA_PATH = os.getenv("DATA")

data_temp = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/temp"
config_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/workflow/config/spatial_montages.yaml"

# ==============================================================================
# ⚡ Convert to Headset Montage:   
#      - Montages to use for data processing are specified in config/config.yaml
#      - Montage configurations are specified in config/spatial_montages.yaml)
# ==============================================================================
rule convert_to_montage:
    input:
        fif = data_temp + "/{montage_type}/{montage_name}/{site}/{data_label}/{sample}_resampled.fif",
        config=config_path
    output:
        fif = temp(data_temp + "/{montage_type}/{montage_name}/{site}/{data_label}/{sample}_headset_montage.fif")
    params:
        script="scripts/data_preprocessing/convert_to_montage.py"
    conda:
        "../../envs/data_preprocessing.yaml"
    shell:
        """
        echo "⚡ Converting {input.fif} to {wildcards.montage_type} ({wildcards.montage_name}): {output.fif}"
        mkdir -p $(dirname {output.fif})
        python {params.script} "{input.fif}" "{output.fif}" "{input.config}" "{wildcards.montage_type}" "{wildcards.montage_name}"
        """