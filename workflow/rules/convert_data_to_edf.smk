import os

DATA_PATH = os.getenv("DATA")

data_raw = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/raw"

converter = f"{DATA_PATH}/AGENDA-Headset-Algorithm/tools/nk2edf"

# =======================
# 🔄 Convert .EEG → .EDF
# =======================
rule convert_eeg_to_edf:
    input:
        data_raw + "/{sample}.EEG"
    output:
        temp(data_raw + "/{sample}_1-1.edf")  # Temporary output
    params:
        converter=converter
    shell:
        """
        echo "🔄 Converting {input} → {output}"
        mkdir -p $(dirname {output})
        {params.converter} -no-annotations "{input}"
        """

# =======================
# 📝 Rename .EDF files
# =======================
rule rename_converted_edf:
    input:
        data_raw + "/{sample}_1-1.edf"
    output:
        data_raw + "/{sample}.edf"
    shell:
        """
        echo "📝 Renaming {input} → {output}"
        mv "{input}" "{output}"
        """
