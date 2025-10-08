import torch
import os
import re
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# ========= Select Device ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========= Utility Functions ==========
np.random.seed(3)

def show_mask(mask, ax, obj_id=None, color=(1, 0, 1, 0.6)):
    """Overlay predicted mask."""
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_gt_mask(gt_mask, ax, color=(1, 0, 0, 0.4)):
    """Overlay ground truth mask."""
    h, w = gt_mask.shape[-2:]
    gt_mask_image = gt_mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(gt_mask_image)

def get_bounding_box_from_mask(mask):
    """Compute bounding box from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # No object found
    x_min, y_min, w, h = cv2.boundingRect(contours[0])
    return np.array([x_min, y_min, x_min + w, y_min + h], dtype=np.float32)

def save_nifti_from_slices(pred_mask_slices, reference_nifti_path, output_nifti_path):
    """
    Save predicted 2D mask slices as a full-size 3D NIfTI file, matching the original resolution.

    Args:
        pred_mask_slices (list of np.ndarray): List of 2D predicted masks.
        reference_nifti_path (str): Path to reference NIfTI file (for metadata).
        output_nifti_path (str): Path to save the output NIfTI file.
    """
    ref_nifti = nib.load(reference_nifti_path)
    original_shape = ref_nifti.shape 

    mask_3d = np.zeros(original_shape, dtype=np.uint8)

    assert len(pred_mask_slices) == original_shape[2], "Mismatch in number of slices!"
    #print(f"Predicted mask shape: {pred_mask_slices}")

    for i in range(len(pred_mask_slices)):
        mask_3d[:, :, i] = pred_mask_slices[i]  

    nifti_img = nib.Nifti1Image(mask_3d, affine=ref_nifti.affine, header=ref_nifti.header)

    nib.save(nifti_img, output_nifti_path)
    print(f"Saved full-size 3D NIfTI mask: {output_nifti_path}")

def calculate_iou(pred_masks, gt_masks):
    """Compute mean IoU between predicted and ground truth masks."""
    assert len(pred_masks) == len(gt_masks), "Mismatch in number of slices!"
    iou_scores = []
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU Score: {mean_iou:.4f}")
    return mean_iou

# ========= Load Data ==========
nii_file_path = '/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/nifti/t2/1.3.6.1.4.1.14519.5.2.1.150726869713582626558410584636710663048.nii.gz'
image_data = nib.load(nii_file_path).get_fdata()
gt_nifti_path = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/segmentation/MR/Prostate-MRI-US-Biopsy-0593-ProstateSurface-seriesUID-1.3.6.1.4.1.14519.5.2.1.150726869713582626558410584636710663048.nii.gz"
gt_masks = nib.load(gt_nifti_path).get_fdata()

long_filename = os.path.basename(gt_nifti_path) 
match = re.search(r"Prostate-MRI-US-Biopsy-(\d+)-", long_filename)
nii_filename = match.group(1) if match else "unknown"
print(f"Extracted NIfTI Identifier: {nii_filename}")

print(f"Image shape: {image_data.shape}, Mask shape: {gt_masks.shape}")

# ========= Set Up Directories ==========
#nii_filename = "1088"
output_dir = os.path.join(os.getcwd(), nii_filename)
output_vis_dir = os.path.join(os.getcwd(), f'vis_{nii_filename}')
output_gt_mask_dir = os.path.join(os.getcwd(), f'gt_seg_{nii_filename}')
output_nifti_path = os.path.join(output_vis_dir, f"pred_seg_{nii_filename}.nii.gz")

for dir_path in [output_dir, output_vis_dir, output_gt_mask_dir]:
    os.makedirs(dir_path, exist_ok=True)

# ========= Preprocess Slices ==========
def normalize_image(slice_2d):
    return ((slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d)) * 255).astype(np.uint8)

frame_names = []
bounding_boxes = {}
for i in range(image_data.shape[2]): 
    slice_2d = normalize_image(image_data[:, :, i])
    slice_filename = os.path.join(output_dir, f'{i:03d}.jpg')
    cv2.imwrite(slice_filename, slice_2d)
    frame_names.append(slice_filename)

    gt_mask = (gt_masks[:, :, i] > 0).astype(np.uint8)
    gt_mask_filename = os.path.join(output_gt_mask_dir, f"gt_mask_{i:03d}.png")
    cv2.imwrite(gt_mask_filename, gt_mask * 255)

    bbox = get_bounding_box_from_mask(gt_mask)
    if bbox is not None:
        bounding_boxes[i] = bbox

print(f"All image slices saved in {output_dir}, GT masks in {output_gt_mask_dir}")

# ========= Initialize Predictor ==========
predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", "../sam2/checkpoints/sam2.1_hiera_large.pt")
inference_state = predictor.init_state(video_path=output_dir)

# ========= Run Segmentation ==========
video_segments = {}

for frame_idx, input_box in bounding_boxes.items():
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state, frame_idx=frame_idx, obj_id=1, box=input_box)

for direction in [True, False]:  # Backward first, then forward
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=direction):
        if out_frame_idx not in video_segments:
            video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}

# ========= Save Visualizations & Evaluate ==========
pred_mask_slices = []

for out_frame_idx in range(len(frame_names)):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Frame {out_frame_idx}")

    frame_image = Image.open(frame_names[out_frame_idx])
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

# Save as 3D NIfTI & Compute IoU
save_nifti_from_slices(pred_mask_slices, gt_nifti_path, output_nifti_path)
calculate_iou(pred_mask_slices, [(gt_masks[:, :, i] > 0).astype(np.uint8) for i in range(len(frame_names))])
