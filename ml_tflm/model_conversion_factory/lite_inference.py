import tensorflow as tf
import numpy as np
from pathlib import Path

# === CONFIG ===
current_dir = Path(__file__).resolve().parent
model_path = current_dir / "model_park" / "model_eegnet_quant_test01.tflite"

parent_dir = current_dir.parent
csv_path = parent_dir / "dataset" / "teensy_data" / "N0000.csv"

# === Load model ===
interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()

# Get input/output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extract quantization params
in_scale, in_zero_point = input_details[0]['quantization']
out_scale, out_zero_point = output_details[0]['quantization']

# === Load CSV (skip header) ===
data = np.loadtxt(csv_path, delimiter=",", skiprows=1)  # shape: (32768, 16)

# Sanity check
if data.shape[1] != 16:
    raise ValueError(f"Expected 16 channels (columns), got {data.shape[1]}")
if data.shape[0] < 128:
    raise ValueError(f"Not enough samples: need at least 128, got {data.shape[0]}")

# === Batch Inference Check =====

BATCH_SIZE = 128          # 1 s @ 128 Hz
NUM_BATCHES = 256

outputs = []
outputs_dtype = np.float32

for batch_idx in range(NUM_BATCHES):
    start = batch_idx * BATCH_SIZE
    end = start + BATCH_SIZE
    if end > data.shape[0]:
        print(f"[WARN] Not enough samples for batch {batch_idx}, stopping.")
        break

    # (128, 16) -> (16, 128) -> (1, 16, 128, 1), all float32
    epoch = data[start:end, :].astype(np.float32, copy=False)  # (128, 16)
    epoch = epoch.T                                           # (16, 128)
    sample = np.expand_dims(epoch, axis=(0, -1))              # (1, 16, 128, 1)

    # === Quantize (force float32 math like MCU) ===
    val32 = (sample.astype(np.float32) / np.float32(in_scale)) + np.float32(in_zero_point)
    q = np.rint(val32).astype(np.int32)
    q = np.clip(q, 0, 255).astype(np.uint8)

    # Set input and run
    interpreter.set_tensor(input_details[0]['index'], q)
    interpreter.invoke()

    # === Get and print raw UINT8 output ===
    output_uint8 = interpreter.get_tensor(output_details[0]['index']).astype(np.uint8, copy=False)
    print("Raw UINT8 output:", ", ".join(str(v) for v in output_uint8.flatten()))

    # === Dequantize output (float32 math) ===
    output_quant = interpreter.get_tensor(output_details[0]['index']).astype(np.int32)
    output_float = (output_quant.astype(np.float32) - np.float32(out_zero_point)) * np.float32(out_scale)
    output_float = output_float.astype(outputs_dtype, copy=False)

    # Flatten if needed (e.g., (1, D) -> (D))
    outputs.append(np.squeeze(output_float, axis=0))

# Stack and average (float32)
outputs = np.stack(outputs, axis=0).astype(np.float32, copy=False)  # (num_batches, D[,...])
avg_output = np.mean(outputs, axis=0, dtype=np.float32)

# === Formatted print ===
print("")
print("=== Averaged Output ===")
print(", ".join(f"{v:.6f}" for v in avg_output))