import os
import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
import torch
import gc
import logging

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sam2_inference.log", mode="w"),
        logging.StreamHandler()
    ]
)

# Utility Functions (same as before)
def show_mask(mask, ax, color=(1, 0, 1, 0.6)):  # Purple for predicted segmentation
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_gt_mask(gt_mask, ax, color=(1, 0, 0, 0.4)):  # Red for ground truth
    h, w = gt_mask.shape[-2:]
    gt_mask_image = gt_mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(gt_mask_image)

def save_nifti_from_slices(pred_mask_slices, reference_nifti_path, output_nifti_path):
    ref_nifti = nib.load(reference_nifti_path)
    original_shape = ref_nifti.shape 
    mask_3d = np.zeros(original_shape, dtype=np.uint8)
    assert len(pred_mask_slices) == original_shape[2], "Mismatch in number of slices!"
    for i in range(len(pred_mask_slices)):
        mask_3d[:, :, i] = pred_mask_slices[i]  
    nifti_img = nib.Nifti1Image(mask_3d, affine=ref_nifti.affine, header=ref_nifti.header)
    nib.save(nifti_img, output_nifti_path)
    print(f"Saved 3D segmentation NIfTI: {output_nifti_path}")

def calculate_iou(pred_masks, gt_masks):
    assert len(pred_masks) == gt_masks.shape[2], "Mismatch in number of slices!"
    iou_scores = []
    for i, pred_mask in enumerate(pred_masks):
        gt_mask = (gt_masks[:, :, i] > 0).astype(np.uint8)
        pred_mask = (pred_mask > 0).astype(np.uint8)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / union if union > 0 else 1.0
        iou_scores.append(iou)
    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU: {mean_iou:.4f}")
    return mean_iou, iou_scores

def compute_dice(pred_masks, gt_masks):
    assert len(pred_masks) == gt_masks.shape[2], "Mismatch in number of slices!"
    dice_scores = []
    for i, pred_mask in enumerate(pred_masks):
        gt_mask = (gt_masks[:, :, i] > 0).astype(np.uint8)
        pred_mask = (pred_mask > 0).astype(np.uint8)
        intersection = np.sum(pred_mask * gt_mask)
        sum_masks = np.sum(pred_mask) + np.sum(gt_mask)
        dice = (2 * intersection) / (sum_masks + 1e-6) if sum_masks > 0 else 1.0
        dice_scores.append(dice)
    mean_dice = np.mean(dice_scores)
    print(f"Mean Dice: {mean_dice:.4f}")
    return mean_dice, dice_scores

# ------------------------------
# Initialize SAM2 predictor (for segmentation)
sam2_predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml",
                                             "../../sam2/checkpoints/sam2.1_hiera_large.pt")

# ------------------------------
# Settings and directories (should match those used in Phase 1)
output_dir = "/home/yw2692/yolo-sam2_output"
preprocess_dir = "/home/yw2692/preprocess_3Dimages/Prostate-MRI-US-Biopsy"
# In Phase 1, we saved slice images and bbox files per subject in workspace_dir/subject_id/
os.makedirs(output_dir, exist_ok=True)

# Process subjects from the main CSV
main_csv_path = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/demo/demo-tcia-biopsy-all-trainvaltest.csv"
df = pd.read_csv(main_csv_path)

# Load patient split CSV and filter for validation subjects.
split_csv_path = "/home/yw2692/yolo_dataset/Prostate-MRI-US-Biopsy/patient_split.csv"
split_df = pd.read_csv(split_csv_path)
val_df = split_df[split_df['split'].str.lower() == 'val']
val_pids = set(val_df['pid'].astype(str))

# Only process subjects in the validation set.
subjects = df[df['pid'].astype(str).isin(val_pids)]
print(f"Original {len(df)} subjects in total.")
print(f"Processing {len(subjects)} validation subjects.")

# Dictionary to store IoU scores per subject
subject_iou = {}
subject_dice = {}
subject_iou_bbox = {}
subject_dice_bbox = {}

with torch.no_grad():

    for idx, row in subjects.iterrows():
        pid = str(row["pid"])
        series_uid = row["Series Instance UID (MRI)"]
        nii_file_path = f"/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/nifti/t2/{series_uid}.nii.gz"
        if not os.path.exists(nii_file_path):
            print(f"Missing T2 file for PID {pid}")
            continue
        pid_without_8 = pid.lstrip("8")
        gt_nifti_path = f"/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/segmentation/MR/Prostate-MRI-US-Biopsy-{pid_without_8}-ProstateSurface-seriesUID-{series_uid}.nii.gz"
        if not os.path.exists(gt_nifti_path):
            print(f"Missing GT segmentation for PID {pid}")
            continue

        print(f"\nProcessing subject {pid} for SAM2 inference")
        # Set up subject directories (must be the same as in Phase 1)
        output_subject_dir = os.path.join(output_dir, pid)
        preprocess_subject_dir = os.path.join(preprocess_dir, pid)
        slice_output_dir = os.path.join(preprocess_subject_dir, "image_slices")
        bbox_output_dir = os.path.join(preprocess_subject_dir, "yolo_bboxes")
        pred_output_dir = os.path.join(output_subject_dir, "predicted_segmentation")
        vis_output_dir = os.path.join(output_subject_dir, "visualizations")
        os.makedirs(output_subject_dir, exist_ok=True)
        os.makedirs(pred_output_dir, exist_ok=True)
        os.makedirs(vis_output_dir, exist_ok=True)

        # Load volumes
        image_data = nib.load(nii_file_path).get_fdata()
        gt_masks = nib.load(gt_nifti_path).get_fdata()
        num_slices = image_data.shape[2]
        print(f"Volume has {num_slices} slices")
        print(f"Image shape: {image_data.shape}")
        print(f"GT mask shape: {gt_masks.shape}")

        # List to store predicted masks for 3D reconstruction
        pred_mask_slices = []
        # for assessing performance only on YOLO bbox slices
        pred_mask_slices_bbox = []
        gt_masks_bbox = []
        # Dictionary for SAM2 outputs
        video_segments = {}
        # Dictionary for boxes
        bboxes = {}
        
        # # Initialize SAM2 predictor (for segmentation)
        # sam2_predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml",
        #                                             "../../sam2/checkpoints/sam2.1_hiera_large.pt")
        # For SAM2, initialize inference state using the slice images directory
        inference_state = sam2_predictor.init_state(video_path=slice_output_dir)
        
        # Loop over slices: only process slices for which a YOLO bbox file exists
        for i in range(num_slices):
            bbox_txt = os.path.join(bbox_output_dir, f"pred_bbox_{pid}_{i:03d}.txt")
            if os.path.exists(bbox_txt):
                # Read the saved bbox (comma-separated)
                with open(bbox_txt, "r") as f:
                    bbox = np.array([float(x) for x in f.read().strip().split(",")])
                bboxes[i] = bbox
                # Use SAM2 with the bounding box prompt
                _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                    inference_state, frame_idx=i, obj_id=1, box=bbox
                )
                # Threshold to obtain a binary mask
                # pred_mask = (out_mask_logits[0] > 0.2).cpu().numpy().astype(np.uint8)
                # video_segments[i] = {out_obj_ids[0]: pred_mask}
            else:
                # For slices without a YOLO bbox, assume no object => zero mask.
                video_segments[i] = {0: np.zeros((image_data.shape[0], image_data.shape[1]), dtype=np.uint8)}
        
        # Propagate segmentation if needed (here we simply use the SAM2 outputs as is)
        for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state, start_frame_idx=0):
            # if out_frame_idx in video_segments:
            if out_frame_idx in bboxes:
                # Update video_segments with propagated results (only update the ones with bbox)
                video_segments[out_frame_idx] = {out_obj_ids[0]: (out_mask_logits[0] > 0.2).cpu().numpy()}
        
        # Create visualization and collect predicted masks
        frame_names = []  # We need to reload slice images from slice_output_dir
        for i in range(num_slices):
            slice_filename = os.path.join(slice_output_dir, f"{pid}_{i:03d}.jpg")
            slice_img = cv2.imread(slice_filename)
            if slice_img is None:
                print(f"Missing slice image for slice {i:03d}")
                continue
            # Convert BGR to RGB for matplotlib
            slice_img = cv2.cvtColor(slice_img, cv2.COLOR_BGR2RGB)
            frame_names.append(slice_img)
        
        # List for 3D reconstruction: only use masks for slices where a bbox was present
        for i in range(num_slices):
            # Use the propagated SAM2 mask (if exists) for slice i
            if i in bboxes:
                # There may be one object; combine if more than one
                merged_mask = np.zeros_like(gt_masks[:, :, i], dtype=np.uint8)
                for _, mask in video_segments[i].items():
                    merged_mask = np.logical_or(merged_mask, mask).astype(np.uint8)
                pred_mask_slices.append(merged_mask)
                pred_mask_slices_bbox.append(merged_mask)
                gt_masks_bbox.append(gt_masks[:, :, i])
                # Visualization: overlay predicted mask and GT on the slice image
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(frame_names[i])
                ax[0].set_title("Prediction Overlay")
                show_mask(merged_mask, ax[0])
                ax[1].imshow(frame_names[i])
                ax[1].set_title("Ground Truth Overlay")
                show_gt_mask(gt_masks[:, :, i], ax[1])
                vis_filename = os.path.join(vis_output_dir, f"slice_{i:03d}.png")
                plt.savefig(vis_filename, bbox_inches='tight', pad_inches=0)
                plt.close()
            else:
                # If no bbox was saved, assume zero mask
                pred_mask_slices.append(np.zeros((gt_masks.shape[0], gt_masks.shape[1]), dtype=np.uint8))
        
        # Save the predicted 3D segmentation volume
        pred_nifti_path = os.path.join(pred_output_dir, f"pred_seg_{pid}.nii.gz")
        save_nifti_from_slices(pred_mask_slices, gt_nifti_path, pred_nifti_path)
        
        # Compute IoU and Dice
        miou_score, _ = calculate_iou(pred_mask_slices, gt_masks)
        mean_dice, _ = compute_dice(pred_mask_slices, gt_masks)
        # Compute IoU and Dice for bbox slices only
        gt_masks_bbox_np = np.stack(gt_masks_bbox, axis=-1)
        miou_score_bbox, _ = calculate_iou(pred_mask_slices_bbox, gt_masks_bbox_np)
        mean_dice_bbox, _ = compute_dice(pred_mask_slices_bbox, gt_masks_bbox_np)
        logging.info(f"Subject {pid}: Mean IoU = {miou_score:.4f}, Mean Dice = {mean_dice:.4f}")
        logging.info(f"Subject {pid}: Mean IoU (bbox only) = {miou_score_bbox:.4f}, Mean Dice (bbox only) = {mean_dice_bbox:.4f}")
        subject_iou[pid] = miou_score
        subject_dice[pid] = mean_dice
        subject_iou_bbox[pid] = miou_score_bbox
        subject_dice_bbox[pid] = mean_dice_bbox

        sam2_predictor.reset_state(inference_state)
        # inference_state = inference_state.cpu()
        # video_segments = video_segments.cpu()
        # bboxes = bboxes.cpu()
        # pred_mask_slices = pred_mask_slices.cpu()
        del inference_state, video_segments, bboxes, pred_mask_slices, frame_names, image_data, gt_masks
        
        gc.collect()
        torch.cuda.empty_cache()


# Convert to arrays:
ious = np.array(list(subject_iou.values()))
dices = np.array(list(subject_dice.values()))
ious_bbox = np.array(list(subject_iou_bbox.values()))
dices_bbox = np.array(list(subject_dice_bbox.values()))

# Compute mean, variance, std:
mean_iou   = ious.mean()
var_iou    = ious.var()
std_iou    = ious.std()

mean_dice  = dices.mean()
var_dice   = dices.var()
std_dice   = dices.std()

mean_iou_bbox   = ious_bbox.mean()
var_iou_bbox    = ious_bbox.var()
std_iou_bbox    = ious_bbox.std()

mean_dice_bbox  = dices_bbox.mean()
var_dice_bbox   = dices_bbox.var()
std_dice_bbox  = dices_bbox.std()

# Log them:
logging.info(
    f"\nOverall IoU across {len(ious)} subjects: "
    f"mean={mean_iou:.4f}, var={var_iou:.4f}, std={std_iou:.4f}"
)
logging.info(
    f"Overall Dice across {len(dices)} subjects: "
    f"mean={mean_dice:.4f}, var={var_dice:.4f}, std={std_dice:.4f}"
)
logging.info(
    f"Overall IoU (bbox only) across {len(ious_bbox)} subjects: "
    f"mean={mean_iou_bbox:.4f}, var={var_iou_bbox:.4f}, std={std_iou_bbox:.4f}"
)
logging.info(
    f"Overall Dice (bbox only) across {len(dices_bbox)} subjects: "
    f"mean={mean_dice_bbox:.4f}, var={var_dice_bbox:.4f}, std={std_dice_bbox:.4f}"
)
