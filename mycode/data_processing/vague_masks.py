import os
import cv2
import nibabel as nib
import numpy as np
import pandas as pd

# --- Vague Mask Functions ---

def perturb_mask_vague_expand(mask, variation=0.2, blur_kernel=(7, 7)):
    """
    Perturbs a binary mask by finding its largest contour and drawing a randomly
    perturbed ellipse. This creates a rough "vague" version of the mask.
    """
    mask_bin = (mask > 0).astype(np.uint8) * 255
    
    # Find contours in the mask.
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask_bin  # return original if no contour is found.
    
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    # get the center
    cx, cy = x + w / 2, y + h / 2
    # random perturb
    new_w = w * (1 + np.random.uniform(-variation, variation))
    new_h = h * (1 + np.random.uniform(-variation, variation))
    # random rotation
    angle = np.random.uniform(0, 360)
    
    # create a blank mask for drawing the new mask
    perturbed_mask = np.ascontiguousarray(np.zeros_like(mask_bin))
    center = (int(cx), int(cy))
    axes = (int(new_w / 2), int(new_h / 2))
    
    # draw a filled ellipse
    cv2.ellipse(perturbed_mask, center, axes, angle, 0, 360, 255, -1)
    
    # blend the ellipse with a dilation of the original mask to keep some structure
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(mask_bin, kernel, iterations=1)
    blended = cv2.addWeighted(perturbed_mask, 0.5, dilated, 0.5, 0)
    
    #  Gaussian blur
    blurred = cv2.GaussianBlur(blended, blur_kernel, sigmaX=0)
    # Threshold back to a binary mask.
    _, binary_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return binary_mask


def create_vague_mask_large(gt_mask, padding=0.3, blur_kernel=(15,15)):
    """
    Creates a vague mask by drawing a filled ellipse around the area covered 
    by the ground truth mask. The ellipse is defined on the largest contour's 
    bounding box (with added padding) and then blurred to produce a smooth, vague boundary.
    
    Parameters:
      - gt_mask: Input binary mask (0 and >0 values are treated as foreground).
      - padding: Fraction of width/height to add as padding around the bounding box.
      - blur_kernel: Kernel size for Gaussian blurring.
    
    Returns:
      - binary_mask: A binary vague mask (0 or 255) produced by the ellipse overlay and smoothing.
    """
    # Create a binary version of the mask.
    mask_bin = (gt_mask > 0).astype(np.uint8) * 255
    
    # Find contours in the mask.
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask_bin  # Return original if no contour is found.
    
    # Use the largest contour.
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding to the bounding box.
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x_new = max(x - pad_x, 0)
    y_new = max(y - pad_y, 0)
    w_new = w + 2 * pad_x
    h_new = h + 2 * pad_y
    
    # Define ellipse parameters: center and axes lengths.
    center = (x_new + w_new // 2, y_new + h_new // 2)
    axes = (w_new // 2, h_new // 2)
    
    # Create a blank mask and draw a filled ellipse.
    vague_mask = np.ascontiguousarray(np.zeros_like(mask_bin))
    cv2.ellipse(vague_mask, center, axes, 0, 0, 360, 255, -1)
    
    # Optional: Apply Gaussian blur to smooth the edges, then threshold to obtain a binary mask.
    vague_mask = cv2.GaussianBlur(vague_mask, blur_kernel, sigmaX=0)
    _, binary_mask = cv2.threshold(vague_mask, 127, 255, cv2.THRESH_BINARY)
    
    return binary_mask


def create_vague_mask_small(gt_mask, scale_factor=0.8, blur_kernel=(15,15)):
    """
    Creates a vague mask by drawing a filled ellipse that is smaller than the ground truth
    region. The ellipse is defined on the largest contour's bounding box (without padding)
    but its size is scaled by 'scale_factor' (< 1) so that it covers less area than
    the ground truth. A Gaussian blur is applied to smooth the boundaries.
    
    Parameters:
      - gt_mask: Input binary mask (nonzero values are treated as foreground).
      - scale_factor: Multiplier (< 1) to shrink the ellipse dimensions relative to the bounding box.
      - blur_kernel: Kernel size for Gaussian blurring.
    
    Returns:
      - binary_mask: A binary vague mask (0 or 255) produced by the scaled ellipse and subsequent smoothing.
    """
    # Create a binary mask from the ground truth.
    mask_bin = (gt_mask > 0).astype(np.uint8) * 255
    
    # Find contours.
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask_bin  # If no contours, return original.
    
    # Use the largest contour.
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Compute the center of the bounding box.
    center = (x + w // 2, y + h // 2)
    # Compute the axes lengths scaled by scale_factor.
    axes = (int((w / 2) * scale_factor), int((h / 2) * scale_factor))
    
    # Create a contiguous blank mask.
    vague_mask = np.ascontiguousarray(np.zeros_like(mask_bin))
    # Draw a filled ellipse using the scaled parameters.
    cv2.ellipse(vague_mask, center, axes, 0, 0, 360, 255, -1)
    
    # Optionally apply Gaussian blur to smooth the edges.
    vague_mask = cv2.GaussianBlur(vague_mask, blur_kernel, sigmaX=0)
    # Threshold to restore a binary mask.
    _, binary_mask = cv2.threshold(vague_mask, 127, 255, cv2.THRESH_BINARY)
    
    return binary_mask


def apply_random_affine_transform(mask, max_rotation=10, max_translation=10, max_scale_variation=0.2):
    """
    Applies a small random affine transformation (rotation, scaling, translation) to a binary mask.
    
    Parameters:
      - mask: Binary input mask (values 0 or 255).
      - max_rotation: Maximum rotation angle in degrees.
      - max_translation: Maximum shift in pixels for x and y directions.
      - max_scale_variation: Maximum percentage to scale up/down (e.g., 0.1 = ±10%)
    
    Returns:
      - Transformed binary mask (0 or 255).
    """
    h, w = mask.shape

    # Random rotation angle, scale, and translation values
    angle = np.random.uniform(-max_rotation, max_rotation)
    scale = 1 + np.random.uniform(-max_scale_variation, max_scale_variation)
    tx = np.random.uniform(-max_translation, max_translation)
    ty = np.random.uniform(-max_translation, max_translation)

    # Compute the affine matrix
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[:, 2] += [tx, ty]  # Add translation

    # Apply the affine transform
    transformed = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

    # Binarize again (optional but safe)
    _, binary_transformed = cv2.threshold(transformed, 127, 255, cv2.THRESH_BINARY)
    return binary_transformed


def randomly_remove_white_pixels(mask, drop_rate=0.15):
    """
    Randomly removes a fraction of white pixels from the mask to further degrade its edges.
    """
    modified_mask = mask.copy()
    white_pixel_indices = np.argwhere(modified_mask == 255)
    num_pixels_to_remove = int(len(white_pixel_indices) * drop_rate)
    
    if num_pixels_to_remove > 0:
        indices_to_remove = np.random.choice(len(white_pixel_indices), num_pixels_to_remove, replace=False)
        pixels_to_remove = white_pixel_indices[indices_to_remove]
        for pixel in pixels_to_remove:
            modified_mask[pixel[0], pixel[1]] = 0
    return modified_mask

def create_vague_mask(gt_mask, variation=0.2, blur_kernel=(7,7), drop_rate=0.15):
    """
    Combines perturbation and random removal to generate a vague mask from a ground truth mask.
    """
    rough_mask = perturb_mask_vague_expand(gt_mask, variation=variation, blur_kernel=blur_kernel)
    # vague_mask = randomly_remove_white_pixels(rough_mask, drop_rate=drop_rate)
    # return vague_mask
    return rough_mask


# --- Evalutation Function ---
def compute_iou(gt_mask, vague_mask):
    """
    Compute Intersection over Union between binary masks.
    """
    gt = (gt_mask > 0).astype(np.uint8)
    vague = (vague_mask > 0).astype(np.uint8)
    intersection = np.logical_and(gt, vague).sum()
    union = np.logical_or(gt, vague).sum()
    return intersection / union if union > 0 else 1.0

def compute_dice(gt_mask, vague_mask):
    """
    Compute Dice coefficient between binary masks.
    """
    gt = (gt_mask > 0).astype(np.uint8)
    vague = (vague_mask > 0).astype(np.uint8)
    intersection = (gt * vague).sum()
    total = gt.sum() + vague.sum()
    return (2 * intersection) / (total + 1e-6) if total > 0 else 1.0

def compute_pixel_accuracy(gt_mask, vague_mask):
    """
    Pixel-wise accuracy between GT and vague mask.
    """
    gt = (gt_mask > 0).astype(np.uint8)
    vague = (vague_mask > 0).astype(np.uint8)
    correct = (gt == vague).sum()
    total = gt.size
    return correct / total

# --- Main Script to Create and Save Vague Masks ---

def main():
    # Set paths to the CSV and output folder for vague masks
    csv_path = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/demo/demo-tcia-biopsy-all-trainvaltest.csv"
    output_base_dir = "/home/yw2692/SAMRefiner_dataset/Prostate-MRI-US-Biopsy/affine_masks"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Load dataset CSV
    df = pd.read_csv(csv_path)

    all_iou_scores = []
    all_dice_scores = []
    all_acc_scores = []
    
    # Loop over each case in the dataset
    for idx, row in df.iterrows():
        pid = str(row["pid"])
        series_uid = row["Series Instance UID (MRI)"]
        
        # Construct the GT segmentation file path (removing leading '8' if necessary)
        pid_without_8 = pid.lstrip("8")
        gt_nifti_path = f"/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/segmentation/MR/Prostate-MRI-US-Biopsy-{pid_without_8}-ProstateSurface-seriesUID-{series_uid}.nii.gz"
        if not os.path.exists(gt_nifti_path):
            print(f"Missing GT segmentation for PID {pid}")
            continue
        
        # Load the ground truth segmentation as a 3D volume
        gt_masks = nib.load(gt_nifti_path).get_fdata()
        print(f"Processing PID: {pid}, mask shape: {gt_masks.shape}")
        
        # Create an output directory for this case
        case_dir = os.path.join(output_base_dir, pid)
        case_output_dir = os.path.join(case_dir, 'vague_masks')
        case_gt_dir = os.path.join(case_dir, 'gt_masks')
        os.makedirs(case_dir, exist_ok=True)
        os.makedirs(case_output_dir, exist_ok=True)
        os.makedirs(case_gt_dir, exist_ok=True)
        
        # Process each slice in the 3D mask volume
        num_slices = gt_masks.shape[2]
        for slice_idx in range(num_slices):
            # Convert slice to binary mask (0 or 255)
            gt_mask = (gt_masks[:, :, slice_idx] > 0).astype(np.uint8) * 255

            # Save the original GT mask regardless of content
            gt_output_filename = os.path.join(case_gt_dir, f"{pid}_{slice_idx:03d}.png")
            cv2.imwrite(gt_output_filename, gt_mask)
            print(f"Saved original GT mask for PID {pid}, slice {slice_idx}")
            
            # Only process slices with non-empty masks
            if np.any(gt_mask > 0):
                # choose transformation type
                # vague_mask = create_vague_mask_large(gt_mask, padding=0.3, blur_kernel=(15,15))  # Larger ellipse
                # vague_mask = create_vague_mask_small(gt_mask, scale_factor=0.8, blur_kernel=(15,15))  # Smaller ellipse
                # vague_mask = create_vague_mask(gt_mask)  # Perturbed ellipse
                vague_mask = apply_random_affine_transform(gt_mask)  # Noisy affine
                # Save the vague mask as a PNG file
                output_filename = os.path.join(case_output_dir, f"{pid}_{slice_idx:03d}.png")
                cv2.imwrite(output_filename, vague_mask)
                print(f"Saved vague mask for PID {pid}, slice {slice_idx}")

                # Compute similarity metrics between vague mask and ground truth
                iou_score = compute_iou(gt_mask, vague_mask)
                dice_score = compute_dice(gt_mask, vague_mask)
                acc_score = compute_pixel_accuracy(gt_mask, vague_mask)

                all_iou_scores.append(iou_score)
                all_dice_scores.append(dice_score)
                all_acc_scores.append(acc_score)

                print(f"Metrics for PID {pid}, Slice {slice_idx}: IoU = {iou_score:.4f}, Dice = {dice_score:.4f}, Acc = {acc_score:.4f}")
    

    print("Vague mask transformation complete.")

    print("\n=== Overall Vague Mask Similarity ===")
    print(f"Mean IoU: {np.mean(all_iou_scores):.4f}")
    print(f"Mean Dice: {np.mean(all_dice_scores):.4f}")
    print(f"Mean Accuracy: {np.mean(all_acc_scores):.4f}")

if __name__ == "__main__":
    main()
