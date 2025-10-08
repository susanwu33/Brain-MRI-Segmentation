import os
import h5py
import numpy as np
from PIL import Image
import argparse

def save_slices_from_h5(h5_path, output_subdir, pid, dataset_key="predictions", threshold=0.5):
    """
    Load a 3D array from h5_path[f[dataset_key]] and save each Z-slice
    (Y x X) as a binary 0/255 PNG in output_subdir.
    """
    with h5py.File(h5_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"Key '{dataset_key}' not found in {h5_path}. Available keys: {list(f.keys())}")
        vol = f[dataset_key][()]
    # handle shape (1,Z,Y,X) or (Z,Y,X)
    if vol.ndim == 4 and vol.shape[0] == 1:
        vol = vol[0]
    z_dim, y_dim, x_dim = vol.shape

    os.makedirs(output_subdir, exist_ok=True)
    for z in range(z_dim):
        slice_arr = vol[z, :, :]
        # binarize → 0 or 255
        mask = (slice_arr > threshold).astype(np.uint8) * 255
        out_path = os.path.join(output_subdir, f"{pid}_{z:03d}.png")
        Image.fromarray(mask).save(out_path)

def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D-UNet HDF5 predictions into per-slice PNG masks."
    )
    parser.add_argument(
        "--pred_dir", "-i", required=True,
        help="Folder containing *_predictions.h5 files"
    )
    parser.add_argument(
        "--out_dir", "-o", required=True,
        help="Root output folder: will create one subfolder per PID"
    )
    parser.add_argument(
        "--key", "-k", default="predictions",
        help="Dataset key in each HDF5 (default: 'predictions')"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.5,
        help="Binarization threshold (default: 0.5)"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for fname in sorted(os.listdir(args.pred_dir)):
        if not fname.lower().endswith(".h5"):
            continue
        pid = os.path.splitext(fname)[0].split("_")[0]
        h5_path = os.path.join(args.pred_dir, fname)
        subdir  = os.path.join(args.out_dir, pid)
        print(f"[{pid}] → slicing {fname} into {subdir}/")
        try:
            save_slices_from_h5(
                h5_path,
                subdir,
                pid,
                dataset_key=args.key,
                threshold=args.threshold
            )
        except Exception as e:
            print(f"  ERROR processing {fname}: {e}")

if __name__ == "__main__":
    main()