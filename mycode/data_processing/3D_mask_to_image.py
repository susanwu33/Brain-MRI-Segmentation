import os
import ast
import numpy as np
import nibabel as nib
import pandas as pd
import cv2
import glob

# -----------------------
# User settings
# -----------------------
CSV_PATH   = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/demo/demo-tcia-biopsy-all-trainvaltest.csv"
MASKS_ROOT = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/segmentation/MR"
OUT_ROOT   = "/home/yw2692/preprocess_3Dimages/Prostate-MRI-US-Biopsy"
os.makedirs(OUT_ROOT, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def load_nii(path):
    return nib.load(path).get_fdata()

def to_bin(arr, thr=0.5):
    return (arr > thr).astype(np.uint8)

def save_png(mask2d, out_path):
    # black/white: 0 -> 0, 1 -> 255
    m = (mask2d.astype(np.uint8) * 255)
    cv2.imwrite(out_path, m)

def write_stack_as_pngs(stack3d, out_dir, pid):
    H, W, D = stack3d.shape
    os.makedirs(out_dir, exist_ok=True)
    for z in range(D):
        out_path = os.path.join(out_dir, f"{pid}_{z:03d}.png")
        save_png(stack3d[:, :, z], out_path)

def pid_key(pid):
    """Always strip leading '8' from pid."""
    return str(pid).lstrip("8")

def tumor_mask_paths(pid, series_uid):
    pattern = os.path.join(
        MASKS_ROOT,
        f"Prostate-MRI-US-Biopsy-{pid_key(pid)}-Target*-seriesUID-{series_uid}.nii.gz"
    )
    return sorted(glob.glob(pattern))

def surface_mask_path(pid, series_uid):
    return os.path.join(
        MASKS_ROOT,
        f"Prostate-MRI-US-Biopsy-{pid_key(pid)}-ProstateSurface-seriesUID-{series_uid}.nii.gz"
    )

# -----------------------
# Core per-case processing
# -----------------------
def process_case(pid, series_uid, targets):
    pid_str = str(pid)

    # 1) Build tumor union (if any)
    tumor_union = None
    tpaths = tumor_mask_paths(pid, series_uid)
    for t in targets:
        # More robust than relying on targets: use actual files present
        # but keep targets loop to preserve your CSV contract
        path = os.path.join(
            MASKS_ROOT,
            f"Prostate-MRI-US-Biopsy-{pid_key(pid)}-Target{t}-seriesUID-{series_uid}.nii.gz"
        )
        if not os.path.exists(path):
            # If the exact Target{t} is missing, try any Target* paths (covers small CSV mismatches)
            continue
        m = to_bin(load_nii(path))
        if tumor_union is None:
            tumor_union = np.zeros_like(m, dtype=np.uint8)
        if m.shape != tumor_union.shape:
            print(f"[WARN] PID {pid} | shape mismatch in {path} → skipping this mask")
            continue
        tumor_union |= m

    # If loop above didn’t find any from 'targets', try all Target* files discovered by glob
    # if tumor_union is None and len(tpaths) > 0:
    #     for p in tpaths:
    #         m = to_bin(load_nii(p))
    #         if tumor_union is None:
    #             tumor_union = np.zeros_like(m, dtype=np.uint8)
    #         if m.shape != tumor_union.shape:
    #             print(f"[WARN] PID {pid} | shape mismatch in {p} → skipping this mask")
    #             continue
    #         tumor_union |= m

    # 2) Load prostate surface (if available)
    prostate = None
    s_path = surface_mask_path(pid, series_uid)
    if os.path.exists(s_path):
        prostate = to_bin(load_nii(s_path))

    # 3) Decide reference shape
    ref = None
    if prostate is not None:
        ref = prostate
    elif tumor_union is not None:
        ref = tumor_union
    else:
        print(f"[INFO] PID {pid} | no tumor masks and no ProstateSurface → skipped.")
        return

    # 4) Ensure both volumes exist (blank where missing)
    if tumor_union is None:
        tumor_union = np.zeros_like(ref, dtype=np.uint8)
    if prostate is None:
        prostate = np.zeros_like(ref, dtype=np.uint8)

    # 5) Write per-slice PNGs
    tumor_dir    = os.path.join(OUT_ROOT, pid_str, "tumor_masks")
    prostate_dir = os.path.join(OUT_ROOT, pid_str, "prostate_masks")
    write_stack_as_pngs(tumor_union, tumor_dir, pid_str)
    write_stack_as_pngs(prostate, prostate_dir, pid_str)

    D = ref.shape[2]
    print(f"[OK] PID {pid} | slices: {D} | wrote tumor → {tumor_dir} | prostate → {prostate_dir}")

# -----------------------
# Main
# -----------------------
def main():
    df = pd.read_csv(CSV_PATH)
    for _, row in df.iterrows():
        pid = row["pid"]
        series_uid = row["Series Instance UID (MRI)"]
        # Parse targets safely (e.g., "[1, 2]")
        try:
            targets = ast.literal_eval(row["targets"])
            if not isinstance(targets, list):
                targets = []
        except Exception:
            targets = []
        process_case(pid, series_uid, targets)

if __name__ == "__main__":
    main()