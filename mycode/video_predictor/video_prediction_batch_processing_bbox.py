import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

import nibabel as nib
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import gc
import logging

from sam2.build_sam import build_sam2_video_predictor

# Enable autocast for CUDA with bfloat16 precision
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# Select Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------- Logging Configuration ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sam2_inference_bbox.log", mode="w"),
        logging.StreamHandler()
    ]
)

# ================== Utility Functions =======================
np.random.seed(3)

def show_mask(mask, ax, obj_id=None, color=(1, 0, 1, 0.6)):  # Purple for prediction overlay
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_gt_mask(gt_mask, ax, color=(1, 0, 0, 0.4)):  # Red for ground truth overlay
    h, w = gt_mask.shape[-2:]
    gt_mask_image = gt_mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(gt_mask_image)

def get_bounding_box_from_mask(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x_min, y_min, w, h = cv2.boundingRect(contours[0])
    return np.array([x_min, y_min, x_min + w, y_min + h], dtype=np.float32)

def save_nifti_from_slices(pred_mask_slices, reference_nifti_path, output_nifti_path):
    ref_nifti = nib.load(reference_nifti_path)
    original_shape = ref_nifti.shape 
    mask_3d = np.zeros(original_shape, dtype=np.uint8)
    assert len(pred_mask_slices) == original_shape[2], "Mismatch in number of slices!"
    for i in range(len(pred_mask_slices)):
        mask_3d[:, :, i] = pred_mask_slices[i]
    nifti_img = nib.Nifti1Image(mask_3d, affine=ref_nifti.affine, header=ref_nifti.header)
    nib.save(nifti_img, output_nifti_path)
    print(f"Saved full-size 3D NIfTI mask: {output_nifti_path}")

# ================== Refined Metric Functions =====================

def calculate_iou(pred_masks, gt_masks):
    """
    Computes per-slice and mean Intersection over Union (IoU). If both masks are empty,
    returns IoU = 1.0.
    """
    iou_scores = []
    num_slices = gt_masks.shape[2]
    for i in range(num_slices):
        gt_mask = (gt_masks[:, :, i] > 0).astype(np.uint8)
        pred_mask = (pred_masks[i] > 0).astype(np.uint8)
        intersection = np.count_nonzero(np.logical_and(gt_mask, pred_mask))
        union = np.count_nonzero(np.logical_or(gt_mask, pred_mask))
        iou = 1.0 if union == 0 else intersection / union
        iou_scores.append(iou)
    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU: {mean_iou:.4f}")
    return mean_iou, iou_scores

def calculate_dice(pred_masks, gt_masks):
    """
    Computes per-slice and mean Dice coefficient. If both masks are empty,
    returns Dice = 1.0.
    """
    dice_scores = []
    num_slices = gt_masks.shape[2]
    for i in range(num_slices):
        gt_mask = (gt_masks[:, :, i] > 0).astype(np.uint8)
        pred_mask = (pred_masks[i] > 0).astype(np.uint8)
        intersection = np.count_nonzero(gt_mask * pred_mask)
        total = np.count_nonzero(gt_mask) + np.count_nonzero(pred_mask)
        dice = 1.0 if total == 0 else (2 * intersection) / total
        dice_scores.append(dice)
    mean_dice = np.mean(dice_scores)
    print(f"Mean Dice: {mean_dice:.4f}")
    return mean_dice, dice_scores

# ========== New Vague Mask Function: Smaller Ellipse ==========
def create_vague_mask_small(gt_mask, scale_factor=0.8, blur_kernel=(15,15)):
    """
    Generates a vague mask by drawing a filled ellipse that is smaller than the ground truth segmentation.
    It computes the bounding box from the largest contour, scales the ellipse dimensions by scale_factor (<1),
    applies a Gaussian blur, and thresholds the result.
    
    Parameters:
      - gt_mask: Input binary ground truth mask.
      - scale_factor: Factor (< 1) to shrink the ellipse (e.g. 0.8).
      - blur_kernel: Kernel size for Gaussian blur.
      
    Returns:
      - A binary vague mask (0 or 255).
    """
    mask_bin = (gt_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_bin
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    center = (x + w // 2, y + h // 2)
    axes = (int((w // 2) * scale_factor), int((h // 2) * scale_factor))
    vague_mask = np.ascontiguousarray(np.zeros_like(mask_bin))
    cv2.ellipse(vague_mask, center, axes, 0, 0, 360, 255, -1)
    vague_mask = cv2.GaussianBlur(vague_mask, blur_kernel, sigmaX=0)
    _, binary_mask = cv2.threshold(vague_mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

# ===========================================================

# ================== Main Processing ==================

# Process subjects from the main CSV
main_csv_path = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/demo/demo-tcia-biopsy-all-trainvaltest.csv"
df = pd.read_csv(main_csv_path)

# Load patient split CSV and filter for validation subjects.
split_csv_path = "/home/yw2692/yolo_dataset/Prostate-MRI-US-Biopsy/patient_split.csv"
split_df = pd.read_csv(split_csv_path)
val_df = split_df[split_df['split'].str.lower() == 'val']
val_pids = set(val_df['pid'].astype(str))

# Only process subjects in the validation set.
df_sample = df[df['pid'].astype(str).isin(val_pids)]
logging.info(f"Total subjects in main CSV: {len(df)}")
logging.info(f"Processing {len(df_sample)} validation subjects.")

# Set workspace directory for saving results.
workspace_dir = "/home/yw2692/workspace/Brain-MRI-Segmentation/sam2_vis_result_bbox_prompt"
os.makedirs(workspace_dir, exist_ok=True)

# Initialize SAM2 Predictor.
predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", "../../sam2/checkpoints/sam2.1_hiera_large.pt")

# Store overall metrics for logging.
video_iou_scores = []
video_dice_scores = []
forced_empty_iou_scores = []
forced_empty_dice_scores = []

for idx, row in df_sample.iterrows():
    pid = str(row["pid"])
    series_uid = row["Series Instance UID (MRI)"]

    # Construct T2 MRI file path.
    nii_file_path = f"/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/nifti/t2/{series_uid}.nii.gz"
    if not os.path.exists(nii_file_path):
        logging.info(f"Missing T2 file for PID {pid}")
        continue

    # Construct segmentation file path.
    pid_without_8 = pid.lstrip("8")
    gt_nifti_path = f"/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/segmentation/MR/Prostate-MRI-US-Biopsy-{pid_without_8}-ProstateSurface-seriesUID-{series_uid}.nii.gz"
    if not os.path.exists(gt_nifti_path):
        logging.info(f"Missing GT segmentation for PID {pid}")
        continue

    # Load image and GT segmentation volumes.
    image_data = nib.load(nii_file_path).get_fdata()
    gt_masks = nib.load(gt_nifti_path).get_fdata()
    print(f"Processing PID: {pid}, Image Shape: {image_data.shape}")

    # Set up directories.
    case_output_dir = os.path.join(workspace_dir, pid)
    output_vis_dir = os.path.join(case_output_dir, "visualizations")
    gt_output_dir = os.path.join(case_output_dir, "ground truth segmentation")
    pred_output_dir = os.path.join(case_output_dir, "predicted segmentation")
    slice_output_dir = os.path.join(case_output_dir, "image slices")
    for dir_path in [case_output_dir, output_vis_dir, gt_output_dir, pred_output_dir, slice_output_dir]:
        os.makedirs(dir_path, exist_ok=True)
    output_nifti_path = os.path.join(pred_output_dir, f"pred_seg_{pid}.nii.gz")

    # Preprocess slices: Normalize each slice and save image & GT mask.
    def normalize_image(slice_2d):
        mini, maxi = np.min(slice_2d), np.max(slice_2d)
        return ((slice_2d - mini) / (maxi - mini) * 255).astype(np.uint8)

    frame_names = []
    bounding_boxes = {}  # Store bbox for slices with non-empty GT.
    for i in range(image_data.shape[2]): 
        slice_2d = normalize_image(image_data[:, :, i])
        slice_filename = os.path.join(slice_output_dir, f'{i:03d}.jpg')
        cv2.imwrite(slice_filename, slice_2d)
        frame_names.append(slice_2d)

        gt_mask = (gt_masks[:, :, i] > 0).astype(np.uint8)
        gt_mask_filename = os.path.join(gt_output_dir, f"gt_mask_{i:03d}.png")
        cv2.imwrite(gt_mask_filename, gt_mask * 255)

        bbox = get_bounding_box_from_mask(gt_mask)
        if bbox is not None:
            bounding_boxes[i] = bbox

    print(f"Total slices with bounding box (object): {len(bounding_boxes)}")

    # Initialize SAM2 state.
    inference_state = predictor.init_state(video_path=slice_output_dir)

    # Run segmentation: For slices with a bounding box prompt, add the box.
    video_segments = {}
    for i in bounding_boxes.keys():
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state, frame_idx=i, obj_id=1, box=bounding_boxes[i])
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
        if out_frame_idx not in video_segments:
            video_segments[out_frame_idx] = {out_obj_ids[0]: (out_mask_logits[0] > 0.2).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}

    # Save visualizations and collect predicted masks from video propagation.
    pred_mask_slices = []
    for out_frame_idx in range(len(frame_names)):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Frame {out_frame_idx}")

        frame_image = frame_names[out_frame_idx]
        ax[0].imshow(frame_image)
        ax[0].set_title("Prediction (Video Propagation)")

        merged_pred_mask = np.zeros_like(gt_masks[:, :, out_frame_idx], dtype=np.uint8)
        if out_frame_idx in video_segments:
            for _, out_mask in video_segments[out_frame_idx].items():
                merged_pred_mask = np.logical_or(merged_pred_mask, out_mask).astype(np.uint8)
                show_mask(out_mask, ax[0])
        else:
            # If no segmentation was propagated, use an all-empty mask.
            merged_pred_mask = np.zeros((gt_masks.shape[0], gt_masks.shape[1]), dtype=np.uint8)

        ax[1].imshow(frame_image)
        ax[1].set_title("Ground Truth")
        show_gt_mask(gt_masks[:, :, out_frame_idx], ax[1])
        vis_filename = os.path.join(output_vis_dir, f"frame_{out_frame_idx:03d}.png")
        plt.savefig(vis_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        pred_mask_slices.append(merged_pred_mask)

    # Save predicted 3D segmentation as NIfTI.
    save_nifti_from_slices(pred_mask_slices, gt_nifti_path, output_nifti_path)

    # === Metric Calculations ===
    # Metric Set 1: Use the output from video propagation (pred_mask_slices).
    miou_video, iou_video = calculate_iou(pred_mask_slices, gt_masks)
    mdice_video, dice_video = calculate_dice(pred_mask_slices, gt_masks)
    # Log metrics.
    logging.info(f"Subject {pid} (Video Propagation): Mean IoU = {miou_video:.4f}, Mean Dice = {mdice_video:.4f}")

    # Metric Set 2: Forced empty for slices without a bounding box.
    # For each slice, if the slice index is in bounding_boxes, use the predicted mask; otherwise, force it to be empty.
    pred_mask_slices_empty = []
    for i in range(len(frame_names)):
        if i in bounding_boxes:
            pred_mask_slices_empty.append(pred_mask_slices[i])
        else:
            pred_mask_slices_empty.append(np.zeros((gt_masks.shape[0], gt_masks.shape[1]), dtype=np.uint8))
    miou_forced, iou_forced = calculate_iou(pred_mask_slices_empty, gt_masks)
    mdice_forced, dice_forced = calculate_dice(pred_mask_slices_empty, gt_masks)
    logging.info(f"Subject {pid} (Forced Empty for No-BBox): Mean IoU = {miou_forced:.4f}, Mean Dice = {mdice_forced:.4f}")

    # Save metrics for overall logging.
    video_iou_scores.append(miou_video)
    video_dice_scores.append(mdice_video)
    forced_empty_iou_scores.append(miou_forced)
    forced_empty_dice_scores.append(mdice_forced)

    # Cleanup for current subject.
    predictor.reset_state(inference_state)
    del inference_state, video_segments, bounding_boxes, pred_mask_slices, frame_names, image_data, gt_masks
    gc.collect()
    torch.cuda.empty_cache()

# Convert lists to NumPy arrays
video_ious   = np.array(video_iou_scores)
video_dices  = np.array(video_dice_scores)
forced_ious  = np.array(forced_empty_iou_scores)
forced_dices = np.array(forced_empty_dice_scores)

# Video Propagation metrics
mean_iou_video = video_ious.mean()
var_iou_video  = video_ious.var()
std_iou_video  = video_ious.std()

mean_dice_video = video_dices.mean()
var_dice_video  = video_dices.var()
std_dice_video  = video_dices.std()

# Forced‐Empty metrics
mean_iou_forced = forced_ious.mean()
var_iou_forced  = forced_ious.var()
std_iou_forced  = forced_ious.std()

mean_dice_forced = forced_dices.mean()
var_dice_forced  = forced_dices.var()
std_dice_forced  = forced_dices.std()

# Log everything
logging.info(
    f"\nVideo Propagation IoU  — mean: {mean_iou_video:.4f}, var: {var_iou_video:.4f}, std: {std_iou_video:.4f}"
)
logging.info(
    f"Video Propagation Dice — mean: {mean_dice_video:.4f}, var: {var_dice_video:.4f}, std: {std_dice_video:.4f}"
)
logging.info(
    f"\nForced Empty IoU  — mean: {mean_iou_forced:.4f}, var: {var_iou_forced:.4f}, std: {std_iou_forced:.4f}"
)
logging.info(
    f"Forced Empty Dice — mean: {mean_dice_forced:.4f}, var: {var_dice_forced:.4f}, std: {std_dice_forced:.4f}"
)