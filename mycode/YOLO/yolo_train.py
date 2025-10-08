import os
import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import glob
import shutil
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO 
import argparse

# ----- Utility Functions -----
def normalize_image(slice_2d):
    """
    Normalize a 2D image slice to the range 0-255 and convert to uint8.
    """
    normalized = ((slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d)) * 255)
    return normalized.astype(np.uint8)

# ----- Phase 1: Data Processing (Generate Images) -----
def generate_images(csv_path, t2_nifti_base, generated_images_dir, generated_labels_dir):
    """
    Generate image slices from T2 NIfTI files based on the existence
    of corresponding label files. If images already exist, skip processing.
    """
    existing_imgs = glob.glob(os.path.join(generated_images_dir, "*.jpg"))
    if existing_imgs:
        print("Image slices already generated. Skipping data processing stage.")
        return

    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        pid = str(row["pid"])
        series_uid = row["Series Instance UID (MRI)"]
        nii_file_path = os.path.join(t2_nifti_base, f"{series_uid}.nii.gz")
        if not os.path.exists(nii_file_path):
            print(f"Missing T2 file for PID {pid}")
            continue

        # Load the T2 MRI volume
        image_data = nib.load(nii_file_path).get_fdata()
        num_slices = image_data.shape[2]
        print(f"Processing PID: {pid} (Volume slices: {num_slices})")

        # For each slice, if a corresponding label file exists, generate and save the image slice.
        for slice_idx in range(num_slices):
            label_file = os.path.join(generated_labels_dir, f"{pid}_{slice_idx:03d}.txt")
            if os.path.exists(label_file):
                slice_img = normalize_image(image_data[:, :, slice_idx])
                base_filename = f"{pid}_{slice_idx:03d}"
                image_filename = os.path.join(generated_images_dir, base_filename + ".jpg")
                cv2.imwrite(image_filename, slice_img)
                print(f"Processed PID {pid} slice {slice_idx}")
    print("Image generation complete.")

# ----- Phase 2: Dataset Splitting by Patient & Saving Split Info -----
def split_dataset_by_patient(generated_images_dir, generated_labels_dir, base_dataset_dir, csv_output_path):
    """
    Groups generated images by patient (using the PID in the filename), splits
    the patients into training and validation sets, copies the corresponding images
    and label files into dedicated directories, and saves a CSV file recording the split.
    """
    # Group images by PID (filenames assumed like "PID_###.jpg")
    images_by_patient = {}
    for img_path in glob.glob(os.path.join(generated_images_dir, "*.jpg")):
        base_name = os.path.basename(img_path)
        pid = base_name.split('_')[0]  # PID is the part before the first underscore
        images_by_patient.setdefault(pid, []).append(img_path)
    
    # List all unique patient IDs
    all_pids = list(images_by_patient.keys())
    print(f"Total patients: {len(all_pids)}")
    
    # Split patient IDs (e.g., 80/20 split)
    train_pids, val_pids = train_test_split(all_pids, test_size=0.2, random_state=42)
    print(f"Training patients: {len(train_pids)}; Validation patients: {len(val_pids)}")
    
    # Save the patient split info to CSV
    rows = []
    for pid in train_pids:
        rows.append({'pid': pid, 'split': 'train'})
    for pid in val_pids:
        rows.append({'pid': pid, 'split': 'val'})
    split_df = pd.DataFrame(rows)
    split_df.to_csv(csv_output_path, index=False)
    print(f"Patient split CSV saved to {csv_output_path}")

    # Build lists of image paths for training and validation
    train_images = []
    for pid in train_pids:
        train_images.extend(images_by_patient[pid])
    val_images = []
    for pid in val_pids:
        val_images.extend(images_by_patient[pid])
    print(f"Total training images: {len(train_images)}; Total validation images: {len(val_images)}")
    
    # Create directories for the split dataset
    train_images_dir = os.path.join(base_dataset_dir, "train", "images")
    train_labels_dir = os.path.join(base_dataset_dir, "train", "labels")
    val_images_dir = os.path.join(base_dataset_dir, "val", "images")
    val_labels_dir = os.path.join(base_dataset_dir, "val", "labels")
    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(d, exist_ok=True)

    def copy_dataset(image_list, dest_img_dir, dest_lbl_dir):
        for img_path in image_list:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            shutil.copy(img_path, os.path.join(dest_img_dir, base_name + ".jpg"))
            label_path = os.path.join(generated_labels_dir, base_name + ".txt")
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(dest_lbl_dir, base_name + ".txt"))
            else:
                open(os.path.join(dest_lbl_dir, base_name + ".txt"), "w").close()

    copy_dataset(train_images, train_images_dir, train_labels_dir)
    copy_dataset(val_images, val_images_dir, val_labels_dir)
    print(f"Copied training images: {len(train_images)}; Copied validation images: {len(val_images)}")
    
    return train_images_dir, train_labels_dir, val_images_dir, val_labels_dir

# ----- Phase 3: Create YAML Configuration -----
def create_yaml_config(base_dataset_dir, dataset_yaml_path):
    """
    Creates a YOLO-style YAML configuration file with relative paths.
    """
    dataset_yaml = {
        "path": base_dataset_dir,
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["prostate"]
    }
    with open(dataset_yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    print(f"YAML config saved to: {dataset_yaml_path}")
    return dataset_yaml_path

# ----- Phase 4: Training -----
def train_yolo(yaml_file, model_name, epochs, imgsz, project, name, device):
    """
    Trains a YOLO model using the provided YAML configuration.
    """
    model = YOLO(model_name)
    print("Starting training...")
    results = model.train(
        data=yaml_file,
        epochs=epochs,
        imgsz=imgsz,
        save_period=10,
        project=project,
        name=name,
        device=device
    )
    print("Training complete.")
    return model

# ----- Phase 5: Validation & Visualization -----
def validate_yolo(model, val_images_dir, imgsz, conf, project, name):
    """
    Runs inference on the validation set and saves visualization images.
    The visualizations are typically saved to a default output directory.
    """
    print("Running validation and generating visualizations...")
    results = model.predict(
        source=val_images_dir,
        imgsz=imgsz,
        conf=conf,
        save=True,
        project=project,
        name=name
    )
    print("Validation visualization saved.")
    return results

# ----- Main Function -----
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Training Pipeline for Prostate MRI")
    parser.add_argument("--phase", type=str, default="train",
                        choices=["train", "validate"],
                        help="Which phase to run: 'train' for training & validation, 'validate' for validation only.")
    parser.add_argument("--base_dataset_dir", type=str, default="/home/yw2692/yolo_dataset/Prostate-MRI-US-Biopsy",
                        help="Base dataset directory path.")
    parser.add_argument("--csv_path", type=str, default="/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/demo/demo-tcia-biopsy-all-trainvaltest.csv",
                        help="Path to the CSV file with whole dataset info.")
    parser.add_argument("--t2_nifti_base", type=str, default="/share/sablab/nfs04/data/Prostate-MRI-US-Biopsy/nifti/t2",
                        help="T2 NIfTI base directory.")
    parser.add_argument("--pretrain_ckpt", type=str, default='yolo11s.pt',
                        help="Pretrained checkpoint to be loaded into the model.")
    parser.add_argument("--epoch", type=int, default=50,
                        help="Training epochs.")
    parser.add_argument("--device",
                        type=lambda x: [int(i) for i in x.split(',')] if ',' in x else int(x),
                        default=0,
                        help="Device ID(s) for training: e.g., '0' for a single GPU, or '0,1' for multiple GPUs.")
    parser.add_argument("--project_name", type=str, default='detect_prostate',
                        help="Project name (directory name to store the checkpoint sub-directories).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--split_data", dest="split_data", action="store_true", help="Split data (default behavior).")
    group.add_argument("--no-split_data", dest="split_data", action="store_false", help="Do not split data (assume data is already split).")
    parser.set_defaults(split_data=True)
    args = parser.parse_args()

    print("Using device:", args.device)

    # ----- Base Paths Setup -----
    base_dataset_dir = args.base_dataset_dir
    generated_images_dir = os.path.join(base_dataset_dir, "images")
    generated_labels_dir = os.path.join(base_dataset_dir, "labels")
    os.makedirs(generated_images_dir, exist_ok=True)
    os.makedirs(generated_labels_dir, exist_ok=True)

    # ----- Variable Setup -----
    model_name = os.path.basename(args.pretrain_ckpt).split('.')[0]
    exp_name = f'{model_name}_finetune'

    # ----- Phase 1: Data Processing -----
    generate_images(args.csv_path, args.t2_nifti_base, generated_images_dir, generated_labels_dir)

    # ----- Phase 2 & 3: Dataset Splitting, YAML Configuration & Save Patient Split CSV -----
    split_csv_path = os.path.join(base_dataset_dir, "patient_split.csv")
    if args.split_data:
        train_images_dir, train_labels_dir, val_images_dir, val_labels_dir = split_dataset_by_patient(
            generated_images_dir, generated_labels_dir, base_dataset_dir, split_csv_path
        )
        yaml_file = os.path.join(base_dataset_dir, "dataset.yaml")
        create_yaml_config(args.base_dataset_dir, yaml_file)
    else:
        yaml_file = os.path.join(base_dataset_dir, "dataset.yaml")
        val_images_dir = os.path.join(base_dataset_dir, "val", "images")

    # ----- Phase 4: Training -----
    if args.phase == 'train':
        model = train_yolo(
            yaml_file=yaml_file,
            model_name=args.pretrain_ckpt,
            epochs=50,
            imgsz=640,
            project=args.project_name,
            name=exp_name,
            device=args.device
        )
    else:
        model = YOLO(args.pretrain_ckpt)

    # ----- Phase 5: Validation & Visualization -----
    validate_yolo(
        model=model,
        val_images_dir=val_images_dir,
        imgsz=640,
        conf=0.25,
        project=args.project_name,
        name=f'{model_name}_prediction'
    )
