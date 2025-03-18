import os

DATA_PATH = os.getenv("DATA")

data_fif = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/fif"
config_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/workflow/config/spatial_montages.yaml"

# ==============================================================================
# ⚡ Convert to Headset Montage:   
#      - Montages to use for data processing are specified in config/config.yaml
#      - Montage configurations are specified in config/spatial_montages.yaml)
# ==============================================================================
rule convert_to_montage:
    input:
        fif= data_fif + "/{montage_type_montage_name[0]}/{montage_type_montage_name[1]}/{site}/{sample}_resampled.fif",
        config=config_path
    output:
        fif= temp(data_fif + "/{montage_type_montage_name[0]}/{montage_type_montage_name[1]}/{site}/{sample}_bipolar.fif")
    params:
        script="scripts/data_preprocessing/convert_to_montage.py"
    conda:
        "../../envs/data_preprocessing.yaml"
    shell:
        """
        echo "⚡ Converting {input.fif} to {wildcards.montage_type} ({wildcards.montage_name}): {output.fif}"
        mkdir -p $(dirname {output.fif})
        python {params.script} "{input.fif}" "{output.fif}" "{input.config}" "{wildcards.montage_type_montage_name[0]}" "{wildcards.montage_type_montage_name[1]}"
        """