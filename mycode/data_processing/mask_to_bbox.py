import os
import cv2
import nibabel as nib
import numpy as np
import pandas as pd

# --- Utility Functions ---

def get_bounding_box_from_mask(mask):
    """
    Compute the bounding box from a binary mask using contours.
    Returns an array [x_min, y_min, x_max, y_max] or None if no contours are found.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x_min, y_min, w, h = cv2.boundingRect(contours[0])
    return np.array([x_min, y_min, x_min + w, y_min + h], dtype=np.float32)

def normalize_image(slice_2d):
    """
    Normalize a 2D image slice to 0-255 and convert to uint8.
    """
    normalized = ((slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d)) * 255)
    return normalized.astype(np.uint8)

# --- Setup Directories ---
# Define the output directories for images and labels.
# yolo_images_dir = "/home/yw2692/yolo_dataset/Prostate-MRI-US-Biopsy/images"
yolo_labels_dir = "/home/yw2692/yolo_dataset/Prostate-MRI-US-Biopsy/labels"
# os.makedirs(yolo_images_dir, exist_ok=True)
os.makedirs(yolo_labels_dir, exist_ok=True)

# --- Load Dataset CSV ---
csv_path = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/demo/demo-tcia-biopsy-all-trainvaltest.csv"
df = pd.read_csv(csv_path)

# --- Process Each Case and Slice ---
for idx, row in df.iterrows():
    pid = str(row["pid"])
    series_uid = row["Series Instance UID (MRI)"]

    # Define paths to the T2 MRI and GT segmentation
    # nii_file_path = f"/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/nifti/t2/{series_uid}.nii.gz"
    # if not os.path.exists(nii_file_path):
    #    print(f"Missing T2 file for PID {pid}")
    #    continue

    pid_without_8 = pid.lstrip("8")  # Remove leading '8' if necessary
    gt_nifti_path = f"/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/segmentation/MR/Prostate-MRI-US-Biopsy-{pid_without_8}-ProstateSurface-seriesUID-{series_uid}.nii.gz"
    if not os.path.exists(gt_nifti_path):
        print(f"Missing GT segmentation for PID {pid}")
        continue

    # Load image and ground truth mask data
    # image_data = nib.load(nii_file_path).get_fdata()
    gt_masks = nib.load(gt_nifti_path).get_fdata()

    print(f"Processing PID: {pid}, Image shape: {gt_masks.shape}")

    # pid_dir = os.path.join(yolo_labels_dir, f"{pid}")
    # os.makedirs(pid_dir, exist_ok=True)

    # Process each slice in the 3D volume
    for slice_idx in range(gt_masks.shape[2]):
        # Normalize and save the image slice
        # slice_img = normalize_image(image_data[:, :, slice_idx])
        # image_filename = os.path.join(yolo_images_dir, f"{pid}_{slice_idx:03d}.jpg")
        # cv2.imwrite(image_filename, slice_img)

        # Process the corresponding GT mask slice and extract the bounding box
        gt_mask = (gt_masks[:, :, slice_idx] > 0).astype(np.uint8)
        bbox = get_bounding_box_from_mask(gt_mask)

        # Prepare the label file in YOLO format
        label_filename = os.path.join(yolo_labels_dir, f"{pid}_{slice_idx:03d}.txt")
        height, width = gt_mask.shape  # image dimensions

        if bbox is not None:
            # Convert bbox from [x_min, y_min, x_max, y_max] to YOLO format
            x_min, y_min, x_max, y_max = bbox
            box_width = x_max - x_min
            box_height = y_max - y_min
            x_center = x_min + box_width / 2.0
            y_center = y_min + box_height / 2.0

            # Normalize coordinates to be between 0 and 1
            x_center /= width
            y_center /= height
            box_width /= width
            box_height /= height

            # For a single class (e.g., tumor), we use class index 0
            label_line = f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"

            with open(label_filename, "w") as f:
                f.write(label_line)
        else:
            # If no object is found, you can either leave the file empty (indicating background) or skip saving it.
            # Here, we create an empty label file.
            open(label_filename, "w").close()

        print(f"Processed slice {slice_idx} for PID {pid}")

print("Dataset creation complete.")
