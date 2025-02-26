import torch
import os
import nibabel as nib
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# Select Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Utility Functions
np.random.seed(3)

def show_mask(mask, ax, obj_id=None, color=(1, 0, 1, 0.6)):  # Purple
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_gt_mask(gt_mask, ax, color=(1, 0, 0, 0.4)):  # Red
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

def calculate_iou(pred_masks, gt_masks):
    assert len(pred_masks) == gt_masks.shape[2], f"Mismatch in number of slices! Predicted mask {len(pred_masks)} and ground truth {gt_masks.shape[2]}"

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

# Load Dataset CSV
csv_path = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/demo/demo-tcia-biopsy-all-trainvaltest.csv"
df = pd.read_csv(csv_path)

# Select the first 10 cases with cancer
df_sample = df[df["cancer"] == True].head(10)

# Path to save all results
workspace_dir = "/home/yw2692/workspace/Brain-MRI-Segmentation/sam2_vis_result_bbox_prompt"
os.makedirs(workspace_dir, exist_ok=True)

# Store IoU scores
iou_scores = []

# Process Each Case
for idx, row in df_sample.iterrows():
    pid = str(row["pid"])
    series_uid = row["Series Instance UID (MRI)"]

    # Construct T2 MRI file path
    nii_file_path = f"/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/nifti/t2/{series_uid}.nii.gz"
    
    if not os.path.exists(nii_file_path):
        print(f"Missing T2 file for PID {pid}")
        continue

    # Construct segmentation file path
    pid_without_8 = pid.lstrip("8")  # Remove leading 8
    gt_nifti_path = f"/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/segmentation/MR/Prostate-MRI-US-Biopsy-{pid_without_8}-ProstateSurface-seriesUID-{series_uid}.nii.gz"

    if not os.path.exists(gt_nifti_path):
        print(f"Missing GT segmentation for PID {pid}")
        continue

    # Load data
    image_data = nib.load(nii_file_path).get_fdata()
    gt_masks = nib.load(gt_nifti_path).get_fdata()

    print(f"\nProcessing PID: {pid}, Shape: {image_data.shape}")

    # Set Up Directories
    case_output_dir = os.path.join(workspace_dir, pid)
    output_vis_dir = os.path.join(case_output_dir, "visualizations")
    gt_output_dir = os.path.join(case_output_dir, "ground truth segmentation")
    pred_output_dir = os.path.join(case_output_dir, "predicted segmentation")
    slice_output_dir = os.path.join(case_output_dir, "image slices")
    output_nifti_path = os.path.join(pred_output_dir, f"pred_seg_{pid}.nii.gz")

    for dir_path in [case_output_dir, output_vis_dir, gt_output_dir, pred_output_dir, slice_output_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Preprocess Slices
    def normalize_image(slice_2d):
        return ((slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d)) * 255).astype(np.uint8)

    frame_names = []
    bounding_boxes = {}
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

    print(f"Total Bounding Boxes Found: {len(bounding_boxes)}")

    # Initialize Predictor
    predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", "../sam2/checkpoints/sam2.1_hiera_large.pt")
    inference_state = predictor.init_state(video_path=slice_output_dir)

    # Run Segmentation
    video_segments = {}

    for frame_idx, input_box in bounding_boxes.items():
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state, frame_idx=frame_idx, obj_id=1, box=input_box)

    #for direction in [True, False]:
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
        if out_frame_idx not in video_segments:
            video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}

    # Save Visualizations
    pred_mask_slices = []

    for out_frame_idx in range(len(frame_names)):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Frame {out_frame_idx}")

        frame_image = frame_names[out_frame_idx]
        ax[0].imshow(frame_image)
        ax[0].set_title("Prediction")

        merged_pred_mask = np.zeros_like(gt_masks[:, :, out_frame_idx], dtype=np.uint8)
        if out_frame_idx in video_segments:
            for _, out_mask in video_segments[out_frame_idx].items():
                merged_pred_mask = np.logical_or(merged_pred_mask, out_mask).astype(np.uint8)
                show_mask(out_mask, ax[0])

        ax[1].imshow(frame_image)
        ax[1].set_title("Ground Truth")
        show_gt_mask(gt_masks[:, :, out_frame_idx], ax[1])

        plt.savefig(os.path.join(output_vis_dir, f"frame_{out_frame_idx:03d}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

        pred_mask_slices.append(merged_pred_mask)

    # Save as NIfTI & Compute IoU
    save_nifti_from_slices(pred_mask_slices, gt_nifti_path, output_nifti_path)
    miou_score, all_ious = calculate_iou(pred_mask_slices, gt_masks)
    iou_scores.append(miou_score)

    # deal with the iou score calculation for slices
    with_prompt_iou = {}
    miou_prompt = 0
    for k in bounding_boxes.keys():
        with_prompt_iou[k] = all_ious[k]
        miou_prompt += all_ious[k]
    miou_prompt /= len(with_prompt_iou)

    print('IoU scores from slices with prompt:')
    print(with_prompt_iou)
    print(f"\nAverage IoU from slices with prompt: {miou_prompt:.4f}")

# Compute & Print Average IoU
avg_iou = np.mean(iou_scores)
print(f"\nAverage IoU Across 10 Cases: {avg_iou:.4f}")

