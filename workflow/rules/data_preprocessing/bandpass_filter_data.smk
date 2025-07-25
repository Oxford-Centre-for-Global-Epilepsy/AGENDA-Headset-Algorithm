import os

# Define paths
DATA_PATH = os.getenv("DATA")

data_edf = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/edf"
data_temp = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/temp"
filters_config_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/workflow/config/filter_settings.yaml"

# =======================
# 📡 Bandpass Filtering Rule
# =======================
rule bandpass_filter_data:
    input:
        edf = data_edf + "/{site}/{data_label}/{sample}.edf",
        config=filters_config_path
    output:
        fif = data_temp + "/{montage_type}/{montage_name}/{site}/{data_label}/{sample}_filtered.fif"
    params:
        script="scripts/data_preprocessing/bandpass_filter_data.py"
    conda:
        "../../envs/data_preprocessing.yaml"
    touch: True
    shell:
        """
        echo "📡 Applying bandpass filter to {input.edf}"
        
        echo "🔍 Running bandpass filter with:"
        echo "    Wildcards: {wildcards}"
        
        mkdir -p $(dirname {output.fif})
        set -ex
        python {params.script} "{input.edf}" "{output.fif}" "{input.config}"
        """
