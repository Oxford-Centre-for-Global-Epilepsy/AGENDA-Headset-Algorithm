import tensorflow as tf
from pathlib import Path
import numpy as np

# Get the directory where lite_inspector.py is
current_dir = Path(__file__).resolve().parent
model_path = current_dir / "model_park" / "model_classifier_test01.tflite"

def print_tflite_summary(model_path):
    model_path = Path(model_path).expanduser().resolve()
    print(f"=== Inspecting: {model_path} ===\n")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()

    # --- Basic IO info ---
    print("=== Model Inputs ===")
    for inp in input_details:
        print(f"Name: {inp['name']}, shape={inp['shape']}, dtype={inp['dtype']}")
        _print_quant_params(inp)

    print("\n=== Model Outputs ===")
    for out in output_details:
        print(f"Name: {out['name']}, shape={out['shape']}, dtype={out['dtype']}")
        _print_quant_params(out)

    # --- Tensor list ---
    print("\n=== All Tensors ===")
    for idx, t in enumerate(tensor_details):
        print(f"[{idx}] {t['name']}")
        print(f"    shape={t['shape']}, dtype={t['dtype']}")
        print(f"    index={t['index']}, allocation_type={t.get('allocation_type', 'n/a')}")
        _print_quant_params(t)

    # --- Operator mapping ---
    print("\n=== Operator Summary ===")
    try:
        from tensorflow.lite.python import schema_py_generated as schema_fb
        from flatbuffers import Builder

        # Load model buffer
        buf = open(model_path, "rb").read()
        model_obj = schema_fb.Model.GetRootAsModel(buf, 0)

        subgraph = model_obj.Subgraphs(0)
        op_codes = [model_obj.OperatorCodes(i).BuiltinCode() for i in range(model_obj.OperatorCodesLength())]

        for i in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(i)
            op_code = op_codes[op.OpcodeIndex()]
            inputs = [tensor_details[t].get("name", f"t{t}") for t in [op.Inputs(j) for j in range(op.InputsLength())]]
            outputs = [tensor_details[t].get("name", f"t{t}") for t in [op.Outputs(j) for j in range(op.OutputsLength())]]
            print(f"Op {i}: BuiltinCode={op_code}, Inputs={inputs}, Outputs={outputs}")
    except ImportError:
        print("FlatBuffers not available, skipping operator summary.")

def _print_quant_params(tensor_detail):
    # Per-tensor
    q = tensor_detail.get("quantization", None)
    if q and not np.allclose(q[0], 0):
        print(f"    Per-tensor quant: scale={q[0]}, zero_point={q[1]}")
    # Per-channel
    qp = tensor_detail.get("quantization_parameters", None)
    if qp:
        scales = qp.get("scales", [])
        zero_points = qp.get("zero_points", [])
        axis = qp.get("quantized_dimension", None)
        if len(scales) > 0:
            print(f"    Per-channel quant: axis={axis}, scales(len)={len(scales)}, zero_points(len)={len(zero_points)}")
            # Short preview if too many
            preview_scales = np.array(scales[:min(8, len(scales))])
            preview_zp = np.array(zero_points[:min(8, len(zero_points))])
            if len(scales) > 8:
                print(f"      scales preview={preview_scales} ...")
                print(f"      zero_points preview={preview_zp} ...")
            else:
                print(f"      scales={np.array(scales)}")
                print(f"      zero_points={np.array(zero_points)}")

if __name__ == "__main__":
    print_tflite_summary(model_path)