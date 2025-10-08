'''
This script to extract YOLO bounding boxes from 3D MRI images.
We used the pre-trained YOLO model (without fine-tuning) to extract the bounding boxes.
'''

import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO  # for YOLO predictions
import nibabel as nib
import shutil
import logging



# --------------------- Logging Configuration ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("yolo_inference.log", mode="w"),
        logging.StreamHandler()
    ]
)


# ------------------ Configuration ------------------
images_dir = "/home/yw2692/yolo_dataset/Prostate-MRI-US-Biopsy/val/images"
labels_dir = "/home/yw2692/yolo_dataset/Prostate-MRI-US-Biopsy/val/labels"
predictions_output_dir = "/home/yw2692/YOLO_process_test/pred_bbox"
visualizations_dir = "/home/yw2692/YOLO_process_test/visualizations"
output_results_csv = "/home/yw2692/YOLO_process_test/evaluation_results.csv"

os.makedirs(predictions_output_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)

yolo_conf_threshold = 0.25
iou_threshold = 0.5  # Threshold for matching a detection (you can adjust this)

# ------------------ Initialize YOLO Model ------------------
yolo_model = YOLO('detect_prostate/yolo11s_finetune2/weights/best.pt')

# ------------------ Utility Functions ------------------

def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) for two boxes.
    Boxes format: [x_min, y_min, x_max, y_max]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    interArea = inter_width * inter_height
    
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    
    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0.0
    iou = interArea / union
    return iou

def compute_dice(boxA, boxB):
    """
    Compute the Dice coefficient for two boxes.
    Dice = (2 * intersection_area) / (area_boxA + area_boxB)
    Boxes format: [x_min, y_min, x_max, y_max]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    interArea = inter_width * inter_height

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    
    if (areaA + areaB) == 0:
        return 0.0
    dice = (2 * interArea) / float(areaA + areaB)
    return dice

def convert_yolo_to_xyxy(yolo_bbox, img_width, img_height):
    """
    Convert a YOLO bbox [class, x_center, y_center, width, height] (normalized)
    into absolute coordinates [x_min, y_min, x_max, y_max].
    """
    # Expected format: [class, x_center, y_center, width, height]
    _, x_center, y_center, box_width, box_height = yolo_bbox
    x_center *= img_width
    y_center *= img_height
    box_width *= img_width
    box_height *= img_height

    x_min = x_center - box_width / 2.0
    y_min = y_center - box_height / 2.0
    x_max = x_center + box_width / 2.0
    y_max = y_center + box_height / 2.0
    return [x_min, y_min, x_max, y_max]

# ------------------ Main Evaluation Loop ------------------
results_list = []

# Global counters
global_TP = 0
global_FP = 0
global_FN = 0
global_TN = 0

matched_ious = []   # to collect IoUs for matched predictions
matched_dices = []  # to collect Dice scores for matched predictions

image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
total_images = len(image_files)

for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Cannot load {img_file}")
        continue
    height, width = image.shape[:2]
    
    # Make a copy to draw annotations on.
    annotated_img = image.copy()
    
    basename = os.path.splitext(img_file)[0]
    
    # --- Load Ground Truth ---
    gt_box = None
    gt_label_path = os.path.join(labels_dir, f"{basename}.txt")
    if os.path.exists(gt_label_path):
        with open(gt_label_path, 'r') as f:
            line = f.readline().strip()
            if line:
                parts = line.split()
                if len(parts) == 5:
                    # Convert YOLO-format label (class, x_center, y_center, width, height) to absolute coordinates
                    yolo_bbox = [float(p) for p in parts]
                    gt_box = convert_yolo_to_xyxy(yolo_bbox, width, height)
                    # Draw the ground truth box in green
                    cv2.rectangle(annotated_img, 
                                  (int(gt_box[0]), int(gt_box[1])), 
                                  (int(gt_box[2]), int(gt_box[3])), 
                                  (0,255,0), 2)
                    cv2.putText(annotated_img, "GT", (int(gt_box[0]), int(gt_box[1])-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    else:
        print(f"No GT label for {basename}")

    # --- Run YOLO Prediction ---
    yolo_results = yolo_model.predict(source=img_path, imgsz=640, conf=yolo_conf_threshold, save=False)
    
    pred_boxes = []  # List to store all predicted boxes for the image
    if yolo_results and len(yolo_results[0].boxes) > 0:
        # Iterate over every detection
        # for bbox_tensor in yolo_results[0].boxes.xyxy:
        #     # Convert tensor to list of floats [x_min, y_min, x_max, y_max]
        #     box = bbox_tensor.cpu().numpy().tolist()
        #     pred_boxes.append(box)
        #     # Draw predicted bounding boxes in red
        #     cv2.rectangle(annotated_img, 
        #                   (int(box[0]), int(box[1])), 
        #                   (int(box[2]), int(box[3])), 
        #                   (0,0,255), 2)
        #     cv2.putText(annotated_img, "Pred", (int(box[0]), int(box[1])-5),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # Use the first bounding box detection (adjust selection as needed)
        bbox_tensor = yolo_results[0].boxes.xyxy[0]  # [x_min, y_min, x_max, y_max]
        bbox = bbox_tensor.cpu().numpy()
        pred_boxes.append(bbox)
        # Draw predicted bounding boxes in red
        cv2.rectangle(annotated_img,
                      (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])),
                      (0,0,255), 2)
        cv2.putText(annotated_img, "Pred", (int(bbox[0]), int(bbox[1])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    
    # --- Save All Predictions to File ---
    pred_file_path = os.path.join(predictions_output_dir, f"pred_bbox_{basename}.txt")
    with open(pred_file_path, "w") as f:
        for box in pred_boxes:
            line = ",".join([f"{coord:.2f}" for coord in box])
            f.write(line + "\n")
    
    # Save the annotated visualization image
    visualization_path = os.path.join(visualizations_dir, f"{basename}_vis.jpg")
    cv2.imwrite(visualization_path, annotated_img)
    
    # --- Evaluation per Image ---
    image_TP = 0
    image_FP = 0
    image_FN = 0
    image_TN = 0
    best_iou = 0.0
    best_dice = 0.0
    status = ""

    if gt_box is not None:
        if len(pred_boxes) > 0:
            # Compute IoU and Dice for each predicted box with the ground truth
            ious = [compute_iou(pred_box, gt_box) for pred_box in pred_boxes]
            dices = [compute_dice(pred_box, gt_box) for pred_box in pred_boxes]
            best_idx = np.argmax(ious)
            best_iou = ious[best_idx]
            best_dice = dices[best_idx]
            
            # If the best prediction meets the IoU threshold, consider it a true positive.
            if best_iou >= iou_threshold:
                image_TP = 1
                image_FP = len(pred_boxes) - 1  # additional boxes are extra detections
                image_FN = 0
                status = "TP"
                matched_ious.append(best_iou)
                matched_dices.append(best_dice)
            else:
                image_TP = 0
                image_FP = len(pred_boxes)
                image_FN = 1
                status = "FN"
        else:
            image_TP = 0
            image_FP = 0
            image_FN = 1
            status = "FN"
    else:
        if len(pred_boxes) > 0:
            image_TP = 0
            image_FP = len(pred_boxes)
            image_FN = 0
            status = "FP"
        else:
            status = "TN"
            image_TN = 1
    
    global_TP += image_TP
    global_FP += image_FP
    global_FN += image_FN
    global_TN += image_TN

    results_list.append({
        "image": basename,
        "gt_box": gt_box if gt_box is not None else "None",
        "num_predictions": len(pred_boxes),
        "best_iou": best_iou,
        "best_dice": best_dice,
        "status": status,
        "TP": image_TP,
        "FP": image_FP,
        "FN": image_FN,
        "TN": image_TN
    })
    
    logging.info(f"{basename}: {status}, Best IoU: {best_iou:.3f}, Best Dice: {best_dice:.3f}, "
          f"Predictions: {len(pred_boxes)}, GT exists: {'Yes' if gt_box is not None else 'No'}")

# ------------------ Save Evaluation Results ------------------
results_df = pd.DataFrame(results_list)
results_df.to_csv(output_results_csv, index=False)
print(f"\nSaved per-image evaluation results to: {output_results_csv}")

# ------------------ Compute Global Metrics ------------------
# Accuracy: (TP + TN) / total images
accuracy = (global_TP + global_TN) / total_images if total_images > 0 else 0
precision = global_TP / (global_TP + global_FP) if (global_TP + global_FP) > 0 else 0
recall = global_TP / (global_TP + global_FN) if (global_TP + global_FN) > 0 else 0
f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
avg_iou = np.mean(matched_ious) if len(matched_ious) > 0 else 0
avg_dice = np.mean(matched_dices) if len(matched_dices) > 0 else 0

total_gt_slices = global_TP + global_FN  # Slices that actually contain objects (i.e., GT exists)
detected_object_slices = global_TP       # Slices with GT that were correctly detected (TP)

logging.info("\n--- Global Evaluation Metrics ---")
logging.info(f"Total Images: {total_images}")
logging.info(f"Total Slices with Objects (GT available): {total_gt_slices}")
logging.info(f"Slices with Objects that YOLO Detected: {detected_object_slices}")
logging.info(f"Detection Rate on Object Slices: {(detected_object_slices / total_gt_slices):.3f}")
logging.info(f"True Positives (TP): {global_TP}")
logging.info(f"False Positives (FP): {global_FP}")
logging.info(f"False Negatives (FN): {global_FN}")
logging.info(f"True Negatives (TN): {global_TN}")
logging.info(f"Accuracy: {accuracy:.3f}")
logging.info(f"Precision: {precision:.3f}")
logging.info(f"Recall: {recall:.3f}")
logging.info(f"F1 Score: {f1_score:.3f}")
logging.info(f"Average Best IoU (for matched slices): {avg_iou:.3f}")
logging.info(f"Average Best Dice (for matched slices): {avg_dice:.3f}")