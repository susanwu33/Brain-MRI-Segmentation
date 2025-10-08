import os
import argparse
import numpy as np
import pandas as pd
import gc
from PIL import Image
import torch
import sys
import logging
import matplotlib.pyplot as plt

# Append SAMRefiner repository to the Python path (adjust as needed)
sys.path.append("/home/yw2692/workspace/Brain-MRI-Segmentation/external/SAMRefiner")
from segment_anything import sam_model_registry
from sam_refiner import sam_refiner

# -----------------------------
# Metric computation functions per slice
def compute_iou_slice(gt_mask, pred_mask):
    """
    Compute IoU between binary GT and predicted masks (expected as 0/255 images).
    """
    gt_bin = (gt_mask > 0).astype(np.uint8)
    pred_bin = (pred_mask > 0).astype(np.uint8)
    intersection = (gt_bin & pred_bin).sum()
    union = (gt_bin | pred_bin).sum()
    return intersection / union if union > 0 else 1.0

def compute_dice_slice(gt_mask, pred_mask):
    """
    Compute Dice coefficient between binary GT and predicted masks (0/255 images).
    """
    gt_bin = (gt_mask > 0).astype(np.uint8)
    pred_bin = (pred_mask > 0).astype(np.uint8)
    intersection = (gt_bin * pred_bin).sum()
    total = gt_bin.sum() + pred_bin.sum()
    return (2 * intersection) / (total + 1e-6) if total > 0 else 1.0


# -----------------------------
# Visualization (Comparison of GT, Vague, Refined Masks overlayed on Image Slice)
def save_overlay_comparison(img_path, vague_mask, refined_mask, gt_mask, output_path):
    """
    Creates and saves a figure with three panels comparing the overlays of:
      - Vague mask
      - Refined mask
      - Ground Truth mask
    on top of the original image slice.

    Parameters:
      - img_path: Path to the image slice (e.g. JPG or PNG).
      - vague_mask: 2D numpy array containing the initial (vague) mask (0/255).
      - refined_mask: 2D numpy array containing the refined mask (0/255).
      - gt_mask: 2D numpy array containing the ground truth mask (0/255).
      - output_path: File path to save the figure.
    """
    # Load original image as RGB.
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    
    # For overlay, normalize masks from 0-255 to [0,1].
    vague_overlay = vague_mask.astype(np.float32) / 255.0
    refined_overlay = refined_mask.astype(np.float32) / 255.0
    gt_overlay = gt_mask.astype(np.float32) / 255.0

    # Create a figure with three panels.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # First panel: Original + vague mask overlay (using a purple cmap).
    axes[0].imshow(img_np)
    axes[0].imshow(vague_overlay, cmap='Purples', alpha=0.4)
    axes[0].set_title('Vague Mask Overlay')
    axes[0].axis('off')
    
    # Second panel: Original + refined mask overlay (using a green cmap).
    axes[1].imshow(img_np)
    axes[1].imshow(refined_overlay, cmap='Greens', alpha=0.4)
    axes[1].set_title('Refined Mask Overlay')
    axes[1].axis('off')
    
    # Third panel: Original + ground truth overlay (using a red cmap).
    axes[2].imshow(img_np)
    axes[2].imshow(gt_overlay, cmap='Reds', alpha=0.4)
    axes[2].set_title('Ground Truth Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# -----------------------------
# Process an individual slice using SAMRefiner by directly providing an image path.
def process_slice_from_path(img_path, init_mask, sam, output_path, use_point=True, use_box=True, use_mask=True):
    """
    Process a slice using SAMRefiner.
    
    Parameters:
      - img_path: Path to the image slice file (e.g., JPG or PNG).
      - init_mask: 2D numpy array of the initial vague mask (values 0/255).
      - sam: an initialized SAM model.
      - output_path: Path to save the refined mask.
      
    Returns:
      - refined_mask_bin: The refined binary mask as a 2D numpy array (0 or 255).
    """
    # Normalize the initial mask to [0,1] if in 0-255 scale
    init_mask_norm = init_mask / 255.0 if np.max(init_mask) == 255 else init_mask
    
    # Run SAMRefiner directly using the provided image path and the normalized initial mask.
    # The API expects a list of initial masks.
    refined_masks = sam_refiner(img_path, [init_mask_norm], sam, use_point=use_point, use_box=use_box, use_mask=use_mask)[0]
    refined_mask = refined_masks[0]
    
    # Threshold the refined mask (if it is a probability map) and convert back to a 0-255 binary mask.
    refined_mask_bin = ((refined_mask > 0.5).astype(np.uint8)) * 255
    
    # Save the refined mask to the output path.
    Image.fromarray(refined_mask_bin).save(output_path)
    
    return refined_mask_bin

# -----------------------------
# Main loop for processing slice images for a subject
def main():
    parser = argparse.ArgumentParser(
        description="Run SAMRefiner on slice images (using image paths directly) for validation subjects and compute IoU/Dice metrics."
    )
    parser.add_argument("--checkpoint", type=str, default="/home/yw2692/workspace/Brain-MRI-Segmentation/external/SAMRefiner/checkpoints/sam_vit_h.pth",
                        help="Path to the SAM checkpoint (e.g. sam_vit_h.pth)")
    parser.add_argument("--image_slice_dir", type=str, default="/home/yw2692/preprocess_3Dimages/Prostate-MRI-US-Biopsy",
                        help="Root directory containing subject folders with an 'image_slices' subfolder")
    parser.add_argument("--vague_mask_dir", type=str, default="/home/yw2692/SAMRefiner_dataset/Prostate-MRI-US-Biopsy/small_masks",
                        help="Root directory containing subject folders with a 'vague_masks' subfolder")
    parser.add_argument("--gt_slice_dir", type=str, default="/home/yw2692/SAMRefiner_dataset/Prostate-MRI-US-Biopsy/small_masks",
                        help="Root directory containing subject folders with a 'gt_masks' subfolder")
    parser.add_argument("--patient_split_csv", type=str, default="/home/yw2692/yolo_dataset/Prostate-MRI-US-Biopsy/patient_split.csv",
                        help="CSV file recording patient splits (with columns including 'pid' and 'split')")
    parser.add_argument("--output_dir", type=str, default="/home/yw2692/SAMRefiner_output/Prostate-MRI-US-Biopsy/large_masks",
                        help="Output directory to save refined masks")
    parser.add_argument("--use_point", action="store_true", help="Use point prompts for SAMRefiner")
    parser.add_argument("--use_box", action="store_true", help="Use box prompts for SAMRefiner")
    parser.add_argument("--use_mask", action="store_true", help="Use mask prompts for SAMRefiner")
    parser.add_argument("--log_file", type=str, default="refinement_metrics.log",
                        help="Log file to save metrics")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference ('cuda' or 'cpu')")
    parser.add_argument("--model_type", type=str, default="default",
                        help="Type of SAM model to use: default/vit_h/vit_l/vit_b.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------
    # Logging
    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # -------------------------------------

    # print what prompt combinations are being used
    print(f"Using the following prompt combinations:")
    if args.use_point:
        print("  - Point prompts")
    if args.use_box:
        print("  - Box prompts")
    if args.use_mask:
        print("  - Mask prompts")
    if not (args.use_point or args.use_box or args.use_mask):
        print("  - No prompts (using default SAMRefiner behavior)")
    
    # ------------------------------
    
    # Load patient splits and select only validation subjects.
    split_df = pd.read_csv(args.patient_split_csv)
    val_df = split_df[split_df['split'].str.lower() == 'val']
    if val_df.empty:
        print("No validation subjects found in the CSV.")
        return

    # Load SAM model
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    print(f"Loaded SAM model '{args.model_type}' on {args.device}.")
    
    overall_iou_initial = []
    overall_dice_initial = []
    overall_iou_refined = []
    overall_dice_refined = []
    
    # Iterate through validation subjects
    for index, row in val_df.iterrows():
        pid = str(row['pid'])
        print(f"\nProcessing subject pid: {pid}")
        
        subject_img_dir = os.path.join(args.image_slice_dir, pid, "image_slices")
        # subject_vague_dir = os.path.join(args.vague_mask_dir, pid, "vague_masks")
        subject_vague_dir = os.path.join(args.vague_mask_dir, pid)
        subject_gt_dir = os.path.join(args.gt_slice_dir, pid, "gt_masks")
        subject_output_dir = os.path.join(args.output_dir, pid, "refined_masks")
        subject_vis_dir = os.path.join(args.output_dir, pid, "visualization")
        os.makedirs(subject_output_dir, exist_ok=True)
        os.makedirs(subject_vis_dir, exist_ok=True)
        
        # Verify required directories exist.
        if not os.path.isdir(subject_img_dir):
            print(f"Missing image_slices directory for pid {pid}, skipping subject.")
            continue
        if not os.path.isdir(subject_vague_dir):
            print(f"Missing vague_masks directory for pid {pid}, skipping subject.")
            continue
        if not os.path.isdir(subject_gt_dir):
            print(f"Missing gt_masks directory for pid {pid}, skipping subject.")
            continue
        
        # Assume slice filenames are like: <pid>_000.jpg, <pid>_001.jpg, etc.
        # extract the vague masks files (only process these)
        img_files = sorted([f for f in os.listdir(subject_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if not img_files:
            print(f"No image slices found for pid {pid}, skipping subject.")
            continue
        
        subject_iou_init = []
        subject_dice_init = []
        subject_iou_refined = []
        subject_dice_refined = []
        
        for img_file in img_files:
            base_name, ext = os.path.splitext(img_file)
            try:
                slice_idx = int(base_name.split('_')[-1])
            except ValueError:
                print(f"Could not parse slice index from filename {img_file}, skipping.")
                continue

            # Construct file paths.
            img_path = os.path.join(subject_img_dir, img_file)

            gt_mask_path = os.path.join(subject_gt_dir, f"{pid}_{slice_idx:03d}.png")
            if not os.path.exists(img_path):
                print(f"Missing image slice for pid {pid}, slice {slice_idx:03d}, skipping slice.")
                continue
            if not os.path.exists(gt_mask_path):
                print(f"Missing GT mask for pid {pid}, slice {slice_idx:03d}, skipping slice.")
                continue
            gt_mask = np.array(Image.open(gt_mask_path).convert("L"))

            # Check if a vague mask exists.
            vague_mask_path = os.path.join(subject_vague_dir, f"{pid}_{slice_idx:03d}.png")
            out_slice_path = os.path.join(subject_output_dir, f"{pid}_{slice_idx:03d}.png")
            comparison_output_path = os.path.join(subject_vis_dir, f"{pid}_{slice_idx:03d}.png")
            if os.path.exists(vague_mask_path):
                init_mask = np.array(Image.open(vague_mask_path).convert("L"))
                # Process with SAMRefiner.
                refined_mask = process_slice_from_path(img_path, init_mask, sam, out_slice_path, use_point=args.use_point, use_box=args.use_box, use_mask=args.use_mask)
                # save visualization
                save_overlay_comparison(img_path, init_mask, refined_mask, gt_mask, comparison_output_path)
            else:
                print(f"Missing vague mask for pid {pid}, slice {slice_idx:03d}. Using all-black masks.")
                init_mask = np.zeros_like(gt_mask)
                refined_mask = np.zeros_like(gt_mask)
                # Save the all-black refined mask.
                # Image.fromarray(refined_mask).save(out_slice_path)
            
            # Compute metrics.
            init_iou = compute_iou_slice(gt_mask, init_mask)
            init_dice = compute_dice_slice(gt_mask, init_mask)
            subject_iou_init.append(init_iou)
            subject_dice_init.append(init_dice)
            
            refined_iou = compute_iou_slice(gt_mask, refined_mask)
            refined_dice = compute_dice_slice(gt_mask, refined_mask)
            subject_iou_refined.append(refined_iou)
            subject_dice_refined.append(refined_dice)
        
        if subject_iou_init:
            avg_iou_init = np.mean(subject_iou_init)
            avg_dice_init = np.mean(subject_dice_init)
            avg_iou_refined = np.mean(subject_iou_refined)
            avg_dice_refined = np.mean(subject_dice_refined)
            logging.info(f"Subject {pid}:\n"
                f"  Initial masks - Mean IoU: {avg_iou_init:.4f}, Mean Dice: {avg_dice_init:.4f}\n"
                f"  Refined masks - Mean IoU: {avg_iou_refined:.4f}, Mean Dice: {avg_dice_refined:.4f}")
            overall_iou_initial.append(avg_iou_init)
            overall_dice_initial.append(avg_dice_init)
            overall_iou_refined.append(avg_iou_refined)
            overall_dice_refined.append(avg_dice_refined)
        else:
            print(f"No slices processed for subject {pid}.")
    
        gc.collect()
        torch.cuda.empty_cache()
    
    if overall_iou_initial:
        # Convert to arrays
        arr_iou_init    = np.array(overall_iou_initial)
        arr_dice_init   = np.array(overall_dice_initial)
        arr_iou_refined = np.array(overall_iou_refined)
        arr_dice_refined= np.array(overall_dice_refined)

        # Compute means (you already have these)
        mean_iou_init    = arr_iou_init.mean()
        mean_dice_init   = arr_dice_init.mean()
        mean_iou_refined = arr_iou_refined.mean()
        mean_dice_refined= arr_dice_refined.mean()

        # Compute variances and stds
        var_iou_init    = arr_iou_init.var()
        std_iou_init    = arr_iou_init.std()
        var_dice_init   = arr_dice_init.var()
        std_dice_init   = arr_dice_init.std()

        var_iou_refined    = arr_iou_refined.var()
        std_iou_refined    = arr_iou_refined.std()
        var_dice_refined   = arr_dice_refined.var()
        std_dice_refined   = arr_dice_refined.std()

        # Log everything
        logging.info("Overall metrics (Validation subjects):")
        logging.info(
            f"  Initial masks  - IoU: mean {mean_iou_init:.4f}, var {var_iou_init:.4f}, std {std_iou_init:.4f}; "
            f"Dice: mean {mean_dice_init:.4f}, var {var_dice_init:.4f}, std {std_dice_init:.4f}"
        )
        logging.info(
            f"  Refined masks  - IoU: mean {mean_iou_refined:.4f}, var {var_iou_refined:.4f}, std {std_iou_refined:.4f}; "
            f"Dice: mean {mean_dice_refined:.4f}, var {var_dice_refined:.4f}, std {std_dice_refined:.4f}"
        )
    else:
        print("No slices processed for any validation subject.")

if __name__ == "__main__":
    main()
