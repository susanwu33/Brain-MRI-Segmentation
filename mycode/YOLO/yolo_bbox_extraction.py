import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO  # for YOLO predictions
import nibabel as nib
import shutil

# Set random seed for reproducibility
np.random.seed(3)

# Initialize YOLO model (update the checkpoint path as needed)
yolo_model = YOLO('/home/yw2692/workspace/Brain-MRI-Segmentation/mycode/YOLO/detect_prostate/yolo11s_finetune2/weights/best.pt')
yolo_conf_threshold = 0.25

# Paths
csv_path = "/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/demo/demo-tcia-biopsy-all-trainvaltest.csv"
df = pd.read_csv(csv_path)
# Process all subjects (or filter as needed)
subjects = df

# Workspace to save preprocessed images and YOLO results
workspace_dir = "/home/yw2692/preprocess_3Dimages/Prostate-MRI-US-Biopsy"
os.makedirs(workspace_dir, exist_ok=True)

for idx, row in subjects.iterrows():
    pid = str(row["pid"])
    series_uid = row["Series Instance UID (MRI)"]
    nii_file_path = f"/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/nifti/t2/{series_uid}.nii.gz"
    if not os.path.exists(nii_file_path):
        print(f"Missing T2 file for PID {pid}")
        continue

    pid_without_8 = pid.lstrip("8")
    # Ground truth segmentation is not needed for YOLO extraction but may be saved for later visualization.
    gt_nifti_path = f"/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/segmentation/MR/Prostate-MRI-US-Biopsy-{pid_without_8}-ProstateSurface-seriesUID-{series_uid}.nii.gz"
    if not os.path.exists(gt_nifti_path):
        print(f"Missing GT segmentation for PID {pid}")
        continue

    print(f"\nProcessing subject {pid} for YOLO bbox extraction")
    # Create subject-specific output directories
    subject_dir = os.path.join(workspace_dir, pid)
    slice_output_dir = os.path.join(subject_dir, "image_slices")
    bbox_output_dir = os.path.join(subject_dir, "yolo_bboxes")
    os.makedirs(slice_output_dir, exist_ok=True)
    os.makedirs(bbox_output_dir, exist_ok=True)

    # Load T2 volume
    image_data = nib.load(nii_file_path).get_fdata()
    num_slices = image_data.shape[2]
    print(f"Volume has {num_slices} slices")

    # Process each slice: run YOLO and store bounding boxes
    for i in range(num_slices):
        # Normalize slice image
        slice_img = ((image_data[:, :, i] - np.min(image_data[:, :, i])) / 
                     (np.max(image_data[:, :, i]) - np.min(image_data[:, :, i])) * 255).astype(np.uint8)
        slice_filename = os.path.join(slice_output_dir, f"{pid}_{i:03d}.jpg")
        cv2.imwrite(slice_filename, slice_img)

        # Run YOLO prediction on the slice image
        yolo_results = yolo_model.predict(source=slice_filename, imgsz=640, conf=yolo_conf_threshold, save=False)
        
        if yolo_results and len(yolo_results[0].boxes) > 0:
            # Use the first bounding box detection (adjust selection as needed)
            bbox_tensor = yolo_results[0].boxes.xyxy[0]  # [x_min, y_min, x_max, y_max]
            bbox = bbox_tensor.cpu().numpy()
            # Save the bounding box into a txt file (comma-separated)
            bbox_txt_path = os.path.join(bbox_output_dir, f"pred_bbox_{pid}_{i:03d}.txt")
            with open(bbox_txt_path, "w") as f:
                f.write(",".join([f"{coord:.2f}" for coord in bbox]))
            print(f"Subject {pid} Slice {i:03d}: Detected bbox {bbox}")
        else:
            print(f"Subject {pid} Slice {i:03d}: No bbox detected.")
