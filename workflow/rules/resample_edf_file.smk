import os

DATA_PATH = os.getenv("DATA")

data_fif = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/fif"

# =======================
# 🔄  Resample .EDF Files to Data/EDF
# =======================
rule resample_edf_files:
    input:
        data_edf + "/{sample}.edf"
    output:
        data_fif + "/{sample}_resampled.fif"
    params:
        sampling_frequency=125
    conda:
        "../envs/resample_edf_file.yaml"
    shell:
        """
        echo "🔄  Resampling {input} → {output}"
        mkdir -p $(dirname {output})
        python scripts/resample_edf_file.py "{input}" "{output}" {params.sampling_frequency} 
        """