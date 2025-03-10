import os

# Define paths
DATA_PATH = os.getenv("DATA")

data_fif = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/fif"
data_hdf5 = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/hdf5"
data_tensors = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/tensors"
config_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/workflow/config/spatial_montages.yaml"

# ==============================================
# 📡 Generate Spatial Tensors Rule
# ==============================================
rule make_tensor:
    input:
        hdf5=data_hdf5 + "/{sample}_normalised_{montage}.h5",
        spatial_config=config_path
    output:
        eeg_tensor=data_tensors + "/{sample}_{montage}.h5"
    params:
        script="scripts/make_tensor.py"
    conda:
        "../envs/convert_to_hdf5.yaml"
    shell:
        """
        echo "⚡ Converting {input.hdf5} tensor: {output.eeg_tensor}"
        mkdir -p $(dirname {output.eeg_tensor})
        python {params.script} "{input.hdf5}" "{output.eeg_tensor}" "{input.spatial_config}"
        """