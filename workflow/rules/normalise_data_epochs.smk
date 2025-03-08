import os

DATA_PATH = os.getenv("DATA")

data_fif = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/fif"

# =======================
# 🔄  Normalise the Epoched EEG data
# =======================
rule normalise_epoched_data:
    input:
        fif = data_fif + "/{sample}_epoched_{montage}.fif"
    output:
        fif = data_fif + "/{sample}_normalised_{montage}.fif"
    params:
        script="scripts/normalise_epoched_data.py"
    conda:
        "../envs/convert_to_bipolar_montage.yaml"
    shell:
        """
        echo "🔄  Normalising {input} → {output}"
        mkdir -p $(dirname {output.fif})
        python {params.script} "{input.fif}" "{output.fif}" 
        """