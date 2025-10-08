import os
import h5py
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === CONFIGURATION ===
PRED_DIR = "/home/yw2692/3D-UNet/test_output_3"
GT_NIFTI_DIR = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/segmentation/MR"
IMG_DIR = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/nifti/t2"
SERIES_CSV = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/demo/demo-tcia-biopsy-all-trainvaltest.csv"
SPLIT_CSV = "/home/yw2692/yolo_dataset/Prostate-MRI-US-Biopsy/patient_split.csv"
SLICE_VIS_DIR = "/home/yw2692/3D-UNet/test_output_3_vis"
os.makedirs(SLICE_VIS_DIR, exist_ok=True)

# === UTILS ===
def compute_iou(pred, gt):
    pred = pred > 0.5
    gt = gt > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union > 0 else 1.0

def compute_dice(pred, gt):
    pred = pred > 0.5
    gt = gt > 0
    intersection = np.sum(pred * gt)
    return 2 * intersection / (pred.sum() + gt.sum() + 1e-6)

def save_nifti(pred_array, ref_path, save_path):
    ref = nib.load(ref_path)
    nifti_img = nib.Nifti1Image(pred_array.astype(np.uint8), affine=ref.affine, header=ref.header)
    nib.save(nifti_img, save_path)

def visualize_slice(pred, gt, slice_idx, pid, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pred[:, :, slice_idx], cmap='plasma')
    axes[0].set_title("Prediction")
    axes[1].imshow(gt[:, :, slice_idx], cmap='gray')
    axes[1].set_title("Ground Truth")
    for ax in axes:
        ax.axis('off')
    plt.suptitle(f"PID {pid} - Slice {slice_idx}")
    plt.savefig(os.path.join(save_dir, f"{pid}_slice_{slice_idx}.png"), bbox_inches='tight')
    plt.close()

def visualize_slice_overlay(pred, gt, orig, slice_idx, pid, save_dir):
    """
    Overlays prediction and ground truth masks on the original image slice.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    slice_img = orig[:, :, slice_idx]
    
    # Normalize image to [0, 1] for display
    img_norm = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-6)

    # Predicted overlay
    axes[0].imshow(img_norm, cmap='gray')
    axes[0].imshow(pred[:, :, slice_idx], cmap='spring', alpha=0.5)
    axes[0].set_title("Prediction Overlay")
    axes[0].axis("off")

    # Ground truth overlay
    axes[1].imshow(img_norm, cmap='gray')
    axes[1].imshow(gt[:, :, slice_idx], cmap='Blues', alpha=0.5)
    axes[1].set_title("Ground Truth Overlay")
    axes[1].axis("off")

    plt.suptitle(f"PID {pid} - Slice {slice_idx}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{pid}_slice_{slice_idx}.png"))
    plt.close()

# === LOAD MAPPING FROM PID TO SERIES_UID ===
df_info = pd.read_csv(SERIES_CSV)
df_split = pd.read_csv(SPLIT_CSV)

# Keep only test split
test_pids = df_split[df_split['split'].str.lower() == 'val']['pid'].astype(str).tolist()

# Mapping from PID to series UID for test PIDs only
df_info_test = df_info[df_info['pid'].astype(str).isin(test_pids)]
pid_to_uid = dict(zip(df_info_test['pid'].astype(str), df_info_test['Series Instance UID (MRI)']))

# === MAIN LOOP ===
pid_scores = []
all_dice = []
all_iou = []

for fname in os.listdir(PRED_DIR):
    if not fname.endswith(".h5"):
        continue

    pid = fname.split("_")[0]
    series_uid = pid_to_uid.get(pid)
    if not series_uid:
        print(f"[!] UID not found for PID {pid}")
        continue

    gt_path = os.path.join(GT_NIFTI_DIR, f"Prostate-MRI-US-Biopsy-{pid.lstrip('8')}-ProstateSurface-seriesUID-{series_uid}.nii.gz")
    t2_path = os.path.join(IMG_DIR, f"{series_uid}.nii.gz")

    if not os.path.exists(gt_path) or not os.path.exists(t2_path):
        print(f"[!] Missing GT or T2 for PID {pid}")
        continue

    # === Load data ===
    with h5py.File(os.path.join(PRED_DIR, fname), 'r') as f:
        # print(f"Keys in {fname}:", list(f.keys()))
        pred = f['predictions'][()]  # (Z, Y, X)

    # Fix pred shape to match GT (Y, X, Z)
    if pred.ndim == 4 and pred.shape[0] == 1:
        pred = pred[0]  # (Z, Y, X)
        pred = np.transpose(pred, (1, 2, 0))  # → (Y, X, Z)

    gt = nib.load(gt_path).get_fdata()
    t2 = nib.load(t2_path).get_fdata()

    print("pred shape:", pred.shape)
    print("gt shape:", gt.shape)
    print("t2 shape:", t2.shape)

    # Ensure shape match
    min_shape = np.minimum.reduce([pred.shape, gt.shape, t2.shape])
    pred = pred[:min_shape[0], :min_shape[1], :min_shape[2]]
    gt = gt[:min_shape[0], :min_shape[1], :min_shape[2]]
    t2 = t2[:min_shape[0], :min_shape[1], :min_shape[2]]

    # convert to 0 and 1
    pred_bin = (pred > 0.5).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    # === Output directory for this PID ===
    pid_out_dir = os.path.join(SLICE_VIS_DIR, pid)
    os.makedirs(pid_out_dir, exist_ok=True)

    # === Save NIfTI prediction ===
    save_nifti(pred_bin, gt_path, os.path.join(pid_out_dir, f"{pid}_pred.nii.gz"))

    # === Slice-by-slice overlay visualization ===
    for slice_idx in range(pred.shape[2]):
        visualize_slice_overlay(pred_bin, gt_bin, t2, slice_idx, pid, pid_out_dir)

    # === Dice & IoU Scores ===
    dice = compute_dice(pred, gt)
    iou = compute_iou(pred, gt)
    pid_scores.append({'pid': pid, 'dice': dice, 'iou': iou})
    all_dice.append(dice)
    all_iou.append(iou)

    print(f"PID {pid}: Dice={dice:.4f}, IoU={iou:.4f}")

# Create DataFrame of results
score_df = pd.DataFrame(pid_scores)

# Append the mean row
mean_row = {
    'pid': 'mean',
    'dice': score_df['dice'].mean(),
    'iou': score_df['iou'].mean()
}
score_df = pd.concat([score_df, pd.DataFrame([mean_row])], ignore_index=True)

# Save updated CSV
score_df.to_csv("val_scores.csv", index=False)

# Print each PID score
print("\n--- Per-PID Dice/IoU ---")
print(score_df)

# Overall mean
mean_dice = score_df["dice"].mean()
mean_iou = score_df["iou"].mean()

print(f"\n Overall Mean Dice: {mean_dice:.4f}")
print(f" Overall Mean IoU : {mean_iou:.4f}")