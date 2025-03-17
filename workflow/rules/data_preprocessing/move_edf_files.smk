import os

DATA_PATH = os.getenv("DATA")

data_raw = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/raw"
data_edf = f"{DATA_PATH}/AGENDA-Headset-Algorithm/data/edf"

# =======================
# 🚚 Move .EDF Files to Data/EDF
# =======================
rule move_edf_files:
    input:
        data_raw + "/{sample}.edf"
    output:
        data_edf + "/{sample}.edf"
    shell:
        """
        echo "🚚 Moving {input} → {output}"
        mkdir -p $(dirname {output})
        mv "{input}" "{output}"
        """