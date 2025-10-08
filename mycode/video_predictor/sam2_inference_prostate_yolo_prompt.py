import torch
import os
import nibabel as nib
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO  # for YOLO predictions
from sam2.build_sam import build_sam2_video_predictor
import shutil

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# Select Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed
np.random.seed(3)

# Utility Functions
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
    print(f"Saved full-size 3D NIfTI mask: {output_nifti_path}")

def calculate_iou(pred_masks, gt_masks):
    assert len(pred_masks) == gt_masks.shape[2], "Mismatch in number of slices!"
    iou_scores = []
    for i, pred_mask in enumerate(pred_masks):
        gt_mask = (gt_masks[:, :, i] > 0).astype(np.uint8)
        pred_mask = (pred_mask > 0).astype(np.uint8)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU Score: {mean_iou:.4f}")
    return mean_iou, iou_scores

def compute_dice(pred_masks, gt_masks):
    """
    Compute the Dice coefficient for each slice and return the mean and all slice scores.
    Dice = 2*|pred ∩ gt| / (|pred| + |gt|)
    """
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
    print(f"Mean Dice Score: {mean_dice:.4f}")
    return mean_dice, dice_scores


# ------------------------------
# Initialize YOLO model (for bounding box prediction)
yolo_model = YOLO('/home/yw2692/workspace/Brain-MRI-Segmentation/mycode/YOLO/detect_prostate/yolo11s_finetune2/weights/best.pt')  # Update checkpoint as needed
yolo_conf_threshold = 0.25

# Initialize SAM2 predictor (for segmentation)
sam2_predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml",
                                             "../../sam2/checkpoints/sam2.1_hiera_large.pt")

# ------------------------------
# Process all subjects from CSV
csv_path = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/demo/demo-tcia-biopsy-all-trainvaltest.csv"
df = pd.read_csv(csv_path)

# We'll process all subjects; you can filter if needed
subjects = df

# Workspace directory to store results per subject
workspace_dir = "/home/yw2692/workspace/Brain-MRI-Segmentation/yolo-sam2"
preprocess_dir = "/home/yw2692/preprocess_3Dimages/Prostate-MRI-US-Biopsy"
os.makedirs(workspace_dir, exist_ok=True)
os.makedirs(preprocess_dir, exist_ok=True)

# Dictionary to store IoU scores per subject
subject_iou = {}
subject_dice = {}

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

    print(f"\nProcessing subject {pid}")
    # Create subject-specific output directories
    case_output_dir = os.path.join(workspace_dir, pid)
    preprocess_case_dir = os.path.join(preprocess_dir, pid)
    output_vis_dir = os.path.join(case_output_dir, "visualizations")
    pred_output_dir = os.path.join(case_output_dir, "predicted_segmentation")
    gt_output_dir = os.path.join(preprocess_case_dir, "ground_truth")
    slice_output_dir = os.path.join(preprocess_case_dir, "image_slices")
    output_nifti_path = os.path.join(pred_output_dir, f"pred_seg_{pid}.nii.gz")
    for d in [case_output_dir, output_vis_dir, gt_output_dir, pred_output_dir, slice_output_dir]:
        os.makedirs(d, exist_ok=True)

    # Load volumes
    image_data = nib.load(nii_file_path).get_fdata()
    gt_masks = nib.load(gt_nifti_path).get_fdata()
    num_slices = image_data.shape[2]
    print(f"Volume has {num_slices} slices")

    # Lists to store slices and predicted masks for 3D reconstruction
    frame_names = []
    pred_mask_slices = []
    # We'll save predicted bboxes into a separate directory (within case_output_dir)
    bbox_output_dir = os.path.join(preprocess_case_dir, "yolo_bboxes")
    os.makedirs(bbox_output_dir, exist_ok=True)

    # Dictionary to store SAM2 segmentation for slices (only slices with bbox prompt)
    video_segments = {}

    # dictionary to save yolo output bbox
    bboxes = {}

    # Process each slice: only process slices that yield a YOLO detection
    for i in range(num_slices):
        # Normalize slice image
        slice_img = ((image_data[:, :, i] - np.min(image_data[:, :, i])) / 
                     (np.max(image_data[:, :, i]) - np.min(image_data[:, :, i])) * 255).astype(np.uint8)
        frame_names.append(slice_img)
        slice_filename = os.path.join(slice_output_dir, f"{pid}_{i:03d}.jpg")
        cv2.imwrite(slice_filename, slice_img)

        # Save GT mask for visualization (optional)
        gt_mask = (gt_masks[:, :, i] > 0).astype(np.uint8)
        gt_mask_filename = os.path.join(gt_output_dir, f"gt_mask_{i:03d}.png")
        cv2.imwrite(gt_mask_filename, gt_mask * 255)

        # Run YOLO prediction on the slice image
        yolo_results = yolo_model.predict(source=slice_filename, imgsz=640, conf=yolo_conf_threshold, save=False)
        
        # If YOLO detects at least one bounding box, use the first detection as the prompt
        if yolo_results and len(yolo_results[0].boxes) > 0:
            bbox_tensor = yolo_results[0].boxes.xyxy[0]  # [x_min, y_min, x_max, y_max]
            bbox = bbox_tensor.cpu().numpy()
            # save
            bboxes[i] = bbox
            # Save predicted bounding box to a txt file
            bbox_txt = os.path.join(bbox_output_dir, f"pred_bbox_{pid}_{i:03d}.txt")
            with open(bbox_txt, "w") as f:
                f.write(",".join([f"{coord:.2f}" for coord in bbox]))
            print(f"Slice {i:03d}: Detected bbox {bbox}")

            # Use SAM2 with the bounding box prompt
            # _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
            #     inference_state, frame_idx=i, obj_id=1, box=bbox
            # )
            # # Assume single object; threshold the logits to obtain binary mask
            # pred_mask = (out_mask_logits[0] > 0.2).cpu().numpy().astype(np.uint8)
            # # Store SAM2 segmentation in a dictionary for later visualization
            # video_segments[i] = {out_obj_ids[0]: pred_mask}
            # # Save the predicted mask for 3D reconstruction
            # pred_mask_slices.append(pred_mask)
        else:
            # No YOLO detection: skip SAM2 processing for this slice and assume background.
            # We do not save a predicted bbox or predicted mask for this slice.
            print(f"Slice {i:03d}: No bbox detected.")
            # For consistency in 3D volume, use a zero mask.
            # pred_mask_slices.append(np.zeros(slice_img.shape, dtype=np.uint8))

    # For SAM2, initialize inference state using the directory containing slice images
    inference_state = sam2_predictor.init_state(video_path=slice_output_dir)

    # add bbox to sam2 separately for detection
    for i in range(num_slices):
        if i in bboxes:
            # Use SAM2 with the bounding box prompt
            _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                inference_state, frame_idx=i, obj_id=1, box=bboxes[i]
            )
            # Assume single object; threshold the logits to obtain binary mask
            # pred_mask = (out_mask_logits[0] > 0.2).cpu().numpy().astype(np.uint8)
            # # Store SAM2 segmentation in a dictionary for later visualization
            # video_segments[i] = {out_obj_ids[0]: pred_mask}
            # Save the predicted mask for 3D reconstruction
            # pred_mask_slices.append(pred_mask)
        # else:
            # No YOLO detection: skip SAM2 processing for this slice and assume background.
            # We do not save a predicted bbox or predicted mask for this slice.
            # print(f"Slice {i:03d}: No bbox detected, skipping SAM2.")
            # For consistency in 3D volume, use a zero mask.
            # pred_mask_slices.append(np.zeros(slice_img.shape, dtype=np.uint8))

    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state, start_frame_idx=0):
        if out_frame_idx not in video_segments:
            if out_frame_idx in bboxes:
                # no video_segment saved if no bbox
                video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.2).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}


    # Create visualizations only for slices with YOLO detections (i.e. where a bbox was predicted)
    for i in range(num_slices):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Slice {i:03d} (Subject {pid})")

        # Original slice image
        ax[0].imshow(frame_names[i])
        ax[0].set_title("Prediction Overlay")

        merged_pred_mask = np.zeros_like(gt_masks[:, :, out_frame_idx], dtype=np.uint8)

        if out_frame_idx in video_segments:
            for _, out_mask in video_segments[out_frame_idx].items():
                merged_pred_mask = np.logical_or(merged_pred_mask, out_mask).astype(np.uint8)
                show_mask(out_mask, ax[0])

        pred_mask_filename = os.path.join(pred_output_dir, f"pred_mask_{out_frame_idx:03d}.png")
        cv2.imwrite(pred_mask_filename, merged_pred_mask * 255)

        # Overlay predicted segmentation mask
        # for _, pred_mask in video_segments[i].items():
        #     show_mask(pred_mask, ax[0])
        # Overlay ground truth segmentation mask
        ax[1].imshow(frame_names[i])
        ax[1].set_title("Ground Truth Overlay")
        show_gt_mask(gt_masks[:, :, i], ax[1])

        vis_filename = os.path.join(output_vis_dir, f"slice_{i:03d}.png")
        plt.savefig(vis_filename, bbox_inches='tight', pad_inches=0)
        plt.close()

        pred_mask_slices.append(merged_pred_mask)

    # Save the predicted 3D segmentation volume (constructed from all slices, using zero masks for non-prompted slices)
    save_nifti_from_slices(pred_mask_slices, gt_nifti_path, os.path.join(pred_output_dir, f"pred_seg_{pid}.nii.gz"))
    
    # Compute IoU over all slices (subject-wise)
    miou_score, all_ious = calculate_iou(pred_mask_slices, gt_masks)
    print(f"Subject {pid}: Mean IoU = {miou_score:.4f}")
    subject_iou[pid] = miou_score

    mean_dice, dice_scores = compute_dice(pred_mask_slices, gt_masks)
    print(f"Subject {pid}: Mean Dice = {mean_dice:.4f}")
    subject_dice[pid] = mean_dice

overall_iou = np.mean(list(subject_iou.values()))
overall_dice = np.mean(list(subject_dice.values()))
print(f"\nOverall Mean IoU Across Subjects: {overall_iou:.4f}")
print(f"Overall Mean Dice Across Subjects: {overall_dice:.4f}")

