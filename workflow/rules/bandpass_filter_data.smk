import os

# Define paths
DATA_PATH = os.getenv("DATA")

data_edf = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/edf"
data_fif = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/fif"
filters_config_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/workflow/config/filter_settings.yaml"

# =======================
# 📡 Bandpass Filtering Rule
# =======================
rule bandpass_filter_data:
    input:
        edf = data_edf + "/{sample}.edf",
        config=filters_config_path
    output:
        fif = data_fif + "/{montage}/{sample}_filtered.fif"
    params:
        script="scripts/bandpass_filter_data.py"
    conda:
        "../envs/convert_to_bipolar_montage.yaml"
    shell:
        """
        echo "📡 Applying bandpass filter to {input.edf}"
        mkdir -p $(dirname {output.fif})
        python {params.script} "{input.edf}" "{output.fif}" "{input.config}"
        """
