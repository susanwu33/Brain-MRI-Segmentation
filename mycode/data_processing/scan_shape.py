import os
import nibabel as nib
import pandas as pd
import numpy as np
import h5py
from collections import Counter

def scan_all_shapes(main_csv, split_csv, t2_dir, seg_dir, split="train"):
    df = pd.read_csv(main_csv)
    split_df = pd.read_csv(split_csv)
    split_pids = split_df[split_df["split"].str.lower() == split]["pid"].astype(str)

    df_filtered = df[df["pid"].astype(str).isin(split_pids)]
    print(f"Scanning {len(df_filtered)} {split} volumes...")

    shapes = []
    for _, row in df_filtered.iterrows():
        pid = str(row["pid"])
        series_uid = row["Series Instance UID (MRI)"]

        image_path = os.path.join(t2_dir, f"{series_uid}.nii.gz")
        if not os.path.exists(image_path):
            continue

        try:
            image_data = nib.load(image_path).get_fdata()
            shapes.append(image_data.shape)
        except Exception as e:
            print(f"Error reading {image_path}: {e}")

    if not shapes:
        print("No shapes collected.")
        return

    shapes_np = np.array(shapes)
    min_shape = np.min(shapes_np, axis=0)
    print(f"Smallest volume shape (Z, Y, X): {min_shape}")
    print(f"Mean volume shape: {np.mean(shapes_np, axis=0)}")
    print(f"Max volume shape: {np.max(shapes_np, axis=0)}")
    
    distinct_shapes = sorted(set(shapes), key=lambda x: (x[0], x[1], x[2]))
    print(f"\nDistinct volume shapes in {split} set:")
    for shape in distinct_shapes:
        print(f"  {shape}")


def scan_hdf5_shapes(hdf5_dir, split="train"):
    file_paths = [
        os.path.join(hdf5_dir, fname)
        for fname in os.listdir(hdf5_dir)
        if fname.endswith(".h5")
    ]
    
    print(f"\nScanning {len(file_paths)} HDF5 volumes in {split} set...")

    shapes = []
    label_shapes = []
    for path in file_paths:
        try:
            with h5py.File(path, 'r') as f:
                if 'raw' not in f:
                    print(f"'raw' dataset missing in {path}")
                    continue
                shape = f['raw'].shape  # (Z, Y, X)
                shapes.append(shape)

                if 'label' not in f:
                    print(f"'label' dataset missing in {path}")
                    continue
                label_shape = f['label'].shape  # (Z, Y, X)
                label_shapes.append(label_shape)
                
        except Exception as e:
            print(f"Error reading {path}: {e}")

    if not shapes:
        print("No valid shapes found.")
        return

    shapes_np = np.array(shapes)
    print(f"Smallest shape: {np.min(shapes_np, axis=0)}")
    print(f"Mean shape: {np.mean(shapes_np, axis=0)}")
    print(f"Largest shape: {np.max(shapes_np, axis=0)}")

    label_shapes_np = np.array(label_shapes)
    print(f"Smallest label shape: {np.min(label_shapes_np, axis=0)}")
    print(f"Mean label shape: {np.mean(label_shapes_np, axis=0)}")
    print(f"Largest label shape: {np.max(label_shapes_np, axis=0)}")

    # Count distinct shapes
    shape_counts = Counter(shapes)
    print(f"\nDistinct shapes and counts in {split} set:")
    for shape, count in sorted(shape_counts.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        print(f"  {shape}: {count}")

    label_shape_counts = Counter(label_shapes)
    print(f"\nDistinct label shapes and counts in {split} set:")
    for shape, count in sorted(label_shape_counts.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        print(f"  {shape}: {count}")

if __name__ == "__main__":
    MAIN_CSV = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/demo/demo-tcia-biopsy-all-trainvaltest.csv"
    SPLIT_CSV = "/home/yw2692/yolo_dataset/Prostate-MRI-US-Biopsy/patient_split.csv"
    T2_DIR = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/nifti/t2"
    SEG_DIR = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/segmentation/MR"

    TRAIN_HDF5_DIR = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/susan/hdf5_dataset/train"
    VAL_HDF5_DIR = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/susan/hdf5_dataset/val"

    # scan_all_shapes(MAIN_CSV, SPLIT_CSV, T2_DIR, SEG_DIR, split="train")
    # scan_all_shapes(MAIN_CSV, SPLIT_CSV, T2_DIR, SEG_DIR, split="val")

    scan_hdf5_shapes(TRAIN_HDF5_DIR, split="train")
    scan_hdf5_shapes(VAL_HDF5_DIR, split="val")