import h5py

def explore_h5_structure(file_path):
    def visit(name, node):
        indent = "  " * (name.count("/") - 1)
        if isinstance(node, h5py.Dataset):
            print(f"{indent}- Dataset: {name} | shape: {node.shape} | dtype: {node.dtype}")
            # Try print dataset content if it's small
            if node.shape == () or (node.shape[0] < 20 and node.ndim == 1):
                print(f"{indent}  → content preview: {node[()]}")
        elif isinstance(node, h5py.Group):
            print(f"{indent}+ Group: {name}")
            for key, val in node.attrs.items():
                print(f"{indent}  @ attr: {key} = {val}")

    with h5py.File(file_path, 'r') as f:
        print("Exploring structure...\n")
        f.visititems(visit)

        print("\nGuessing possible channel label datasets...\n")
        for name in f:
            item = f[name]
            if isinstance(item, h5py.Dataset):
                if "label" in name.lower() or "chan" in name.lower():
                    print(f"- {name}: {item[()]}")
            elif isinstance(item, h5py.Group):
                for subname in item:
                    d = item[subname]
                    if isinstance(d, h5py.Dataset):
                        if "label" in subname.lower() or "chan" in subname.lower():
                            print(f"- {name}/{subname}: {d[()]}")

if __name__ == "__main__":
    h5_file_path="ml_tflm/dataset/agenda_data_03/combined_south_africa_monopolar_standard_10_20.h5"
    dataset_name="anyu_dataset_south_africa_monopolar_standard_10_20"

    explore_h5_structure(h5_file_path)