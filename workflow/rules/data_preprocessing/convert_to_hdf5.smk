import os

# Define paths
DATA_PATH = os.getenv("DATA")

data_fif = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/fif"
data_hdf5 = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/hdf5"
data_preprocessed = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/preprocessed"
hdf5_config_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/workflow/config/hdf5_settings.yaml"
montage_config_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/workflow/config/spatial_montages.yaml"

# ==============================================
# 📡 Convert Normalised Data to HDF5 File Rule
# ==============================================

rule convert_to_hdf5:
    input:
        fif = data_fif + "/{montage_type_montage_name[0]}/{montage_type_montage_name[1]}/{site}/{sample}_normalised.fif",
        config = hdf5_config_path,
        montage_config = montage_config_path
    output:
        hdf5 = data_preprocessed + "/{montage_type_montage_name[0]}/{montage_type_montage_name[1]}/{site}/{sample}.h5"
    conda:
        "../../envs/data_preprocessing.yaml"
    params:
        script = "scripts/data_preprocessing/convert_to_hdf5.py"
    shell:
        """
        echo "📦 Converting {input.fif} → {output.hdf5}"
        mkdir -p $(dirname {output.hdf5})
        python {params.script} "{input.fif}" "{output.hdf5}" "{input.config}" "{input.montage_config}" "{wildcards.montage_type_montage_name[0]}" "{wildcards.montage_type_montage_name[1]}" "{wildcards.site}"
        """
