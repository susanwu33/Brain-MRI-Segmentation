import os
import nibabel as nib
import numpy as np
import pandas as pd
import h5py

def convert_nifti_to_hdf5(image_path, label_path, output_path):
    # Load the NIfTI files
    image_nifti = nib.load(image_path)
    label_nifti = nib.load(label_path)
    
    # Get the data arrays
    image_data = image_nifti.get_fdata()
    label_data = label_nifti.get_fdata()
    
    # Ensure the data is in the correct shape and type
    image_data = image_data.astype(np.float32)
    label_data = label_data.astype(np.uint8)

    # Transpose from (H, W, Z) to (Z, H, W) = (Z, Y, X)
    image_data = np.transpose(image_data, (2, 0, 1))
    label_data = np.transpose(label_data, (2, 0, 1))
    
    # Create the HDF5 file
    with h5py.File(output_path, 'w') as h5f:
        h5f.create_dataset('raw', data=image_data, compression="gzip")
        h5f.create_dataset('label', data=label_data, compression="gzip")


def process_all_cases(main_csv, split_csv, t2_dir, seg_dir, output_dir, split="train"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(main_csv)
    split_df = pd.read_csv(split_csv)
    split_pids = split_df[split_df["split"].str.lower() == split]["pid"].astype(str)

    df_filtered = df[df["pid"].astype(str).isin(split_pids)]
    print(f"Processing {len(df_filtered)} {split} cases...")

    for _, row in df_filtered.iterrows():
        pid = str(row["pid"])
        series_uid = row["Series Instance UID (MRI)"]

        image_path = os.path.join(t2_dir, f"{series_uid}.nii.gz")
        pid_suffix = pid.lstrip("8")
        label_path = os.path.join(seg_dir, f"Prostate-MRI-US-Biopsy-{pid_suffix}-ProstateSurface-seriesUID-{series_uid}.nii.gz")

        h5_path = os.path.join(output_dir, f"{pid}.h5")
        convert_nifti_to_hdf5(image_path, label_path, h5_path)


if __name__ == "__main__":
    # Set these to your paths
    MAIN_CSV = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/demo/demo-tcia-biopsy-all-trainvaltest.csv"
    SPLIT_CSV = "/home/yw2692/yolo_dataset/Prostate-MRI-US-Biopsy/patient_split.csv"
    T2_DIR = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/nifti/t2"
    SEG_DIR = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/segmentation/MR"
    OUTPUT_ROOT = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/susan/hdf5_dataset"

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Process both splits
    process_all_cases(MAIN_CSV, SPLIT_CSV, T2_DIR, SEG_DIR, os.path.join(OUTPUT_ROOT, "train"), split="train")
    process_all_cases(MAIN_CSV, SPLIT_CSV, T2_DIR, SEG_DIR, os.path.join(OUTPUT_ROOT, "val"), split="val")
