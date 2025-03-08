import os

DATA_PATH = os.getenv("DATA")

data_fif = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/fif"
config_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/workflow/config/spatial_montages.yaml"

# =======================
# ⚡ Convert to Bipolar Montage
# =======================
rule convert_to_bipolar:
    input:
        fif="{data_fif}/{sample}_resampled.fif",
        config=config_path
    output:
        fif="{data_fif}/{sample}_bipolar_{montage}.fif"
    params:
        script="scripts/convert_to_bipolar.py"
    conda:
        "../envs/convert_to_bipolar_montage.yaml"
    shell:
        """
        echo "⚡ Converting {input.fif} to Bipolar Montage: {output.fif}"
        mkdir -p $(dirname {output.fif})
        python {params.script} "{input.fif}" "{output.fif}" "{input.config}" "{wildcards.montage}"
        """