import os

# Define paths
DATA_PATH = os.getenv("DATA")

data_temp = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/temp"
data_processed = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/processed"
hdf5_config_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/workflow/config/hdf5_settings.yaml"

# ==============================================
# 📡 Convert Normalised Data to HDF5 File Rule
# ==============================================

rule convert_to_hdf5:
    input:
        fif = data_temp + "/{montage_type}/{montage_name}/{site}/{data_label}/{sample}_normalised.fif",
        config = hdf5_config_path
    output:
        hdf5 = data_processed + "/{montage_type}/{montage_name}/{site}/{data_label}/{sample}.h5"
    conda:
        "../../envs/data_preprocessing.yaml"
    params:
        script = "scripts/data_preprocessing/convert_to_hdf5.py"
    touch: True
    shell:
        """
        echo "📦 Converting {input.fif} → {output.hdf5}"
        mkdir -p $(dirname {output.hdf5})
        set -ex
        python {params.script} "{input.fif}" "{output.hdf5}" "{input.config}" "{wildcards.montage_type}" "{wildcards.montage_name}" "{wildcards.site}" "{wildcards.data_label}"
        """
