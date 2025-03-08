import os

# Define paths
DATA_PATH = os.getenv("DATA")

data_fif = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/fif"
data_hdf5 = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/hdf5"
hdf5_config_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/workflow/config/hdf5_settings.yaml"

# ==============================================
# 📡 Convert Normalised Data to HDF5 File Rule
# ==============================================

rule convert_to_hdf5:
    input:
        fif = data_fif + "/{sample}_normalised_{montage}.fif",
        config = hdf5_config_path
    output:
        hdf5 = data_hdf5 + "/{sample}_normalised_{montage}.h5"
    conda:
        "../envs/convert_to_hdf5.yaml"
    shell:
        """
        echo "📦 Converting {input.fif} → {output.hdf5}"
        mkdir -p $(dirname {output.hdf5})
        python scripts/convert_to_hdf5.py {input.fif} {output.hdf5} {input.config}
        """
