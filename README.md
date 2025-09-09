# 🧠 Brain MRI Segmentation

A project exploring **YOLO-based prostate detection** and **SAM/SAMRefiner-based segmentation refinement** on 3D prostate MRI scans.

---

## ✅ Project Progress

- [ ] Extend the segmentation to tumor
---

## 📊 Mean IoU Scores for SAM Prompting Methods

This table presents the **Mean IoU Scores** for different SAM prompting methods across 10 samples.  
Each method includes both the **overall mean IoU** and the **mean IoU from slices with prompts only**.

| **Sample ID** | **Masks (Overall)** | **Masks (Prompted)** | **Perturbed (Overall)** | **Perturbed (Prompted)** | **BBox (Overall)** | **BBox (Prompted)** | **Points (Overall)** | **Points (Prompted)** |
|--------------:|--------------------:|---------------------:|-------------------------:|--------------------------:|-------------------:|--------------------:|---------------------:|----------------------:|
| 81088 | 0.4667 | 1.0000 | 0.4040 | 0.8657 | 0.3851 | 0.8251 | 0.1884 | 0.4037 |
| 80593 | 0.6333 | 1.0000 | 0.5650 | 0.8922 | 0.5410 | 0.8542 | 0.3188 | 0.5033 |
| 80139 | 0.4000 | 1.0000 | 0.3434 | 0.8585 | 0.3264 | 0.8160 | 0.1195 | 0.2988 |
| 80421 | 0.6333 | 1.0000 | 0.5409 | 0.8540 | 0.5368 | 0.8476 | 0.3635 | 0.5740 |
| 81099 | 0.4833 | 1.0000 | 0.4103 | 0.8488 | 0.4086 | 0.8454 | 0.2584 | 0.5347 |
| 80900 | 0.4333 | 1.0000 | 0.3813 | 0.8798 | 0.3697 | 0.8531 | 0.1831 | 0.4225 |
| 80179 | 0.5500 | 1.0000 | 0.4734 | 0.8607 | 0.4447 | 0.8086 | 0.2578 | 0.4688 |
| 80701 | 0.3833 | 1.0000 | 0.3250 | 0.8478 | 0.3110 | 0.8112 | 0.1909 | 0.4981 |
| 80076 | 0.7125 | 1.0000 | 0.6389 | 0.8966 | 0.5901 | 0.8281 | 0.5080 | 0.7130 |
| 80889 | 0.5500 | 1.0000 | 0.4717 | 0.8576 | 0.4745 | 0.8627 | 0.2810 | 0.5109 |

**Legend:**
- **Overall** → IoU score across all slices  
- **Slices with Prompts** → IoU score on slices where prompts were applied  

**Average IoU Across 10 Cases:**
- **Masks (Overall):** 0.5246  
- **Perturbed Masks (Overall):** 0.4554  
- **Bounding Box (Overall):** 0.4388  
- **Points (Overall):** 0.2670  

---

## 🎯 YOLO Detection Performance on Prostate-MRI Slices

Our fine-tuned YOLO model was evaluated on a dataset of **7,280 images** extracted from 3D MRI scans (validation set).  
The task was to detect the prostate region as a single object class.  

We computed standard object detection metrics—including **Accuracy, Precision, Recall, F1 Score**—as well as bounding-box quality metrics:  
- **Average Best Intersection over Union (IoU)**  
- **Average Best Dice Coefficient**

⚠️ Matching IoU threshold = `0.5`.  
Both IoU and Dice are computed **only on matched slices (true positives)**.

---

### 📑 Evaluation Results

| **Metric**                        | **Value** |
|-----------------------------------|-----------|
| Total Images                      | 7280      |
| Total Slices with Objects (GT)    | 3736      |
| Slices with Objects Detected      | 3248      |
| Detection Rate on Object Slices   | 0.869     |
|                                   |           |
| True Positives (TP)               | 3248      |
| False Positives (FP)              | 114       |
| False Negatives (FN)              | 488       |
| True Negatives (TN)               | 3507      |
| Accuracy                          | 0.928     |
| Precision                         | 0.966     |
| Recall                            | 0.869     |
| F1 Score                          | 0.915     |
| Average Best IoU (Matched Slices) | 0.862     |
| Average Best Dice (Matched Slices)| 0.923     |

---

### 🔍 Interpretation

- High **precision** and strong bounding box overlap (**IoU, Dice**)  
- **Recall** shows some slices are missed → potential to improve slice detection  
- Future improvements:  
  - More complex models  
  - Data augmentation  

---

## ⚖️ Performance Comparison

Performance is evaluated with **Mean IoU / Dice ± Standard Deviation (SD).**

| **Method**                                        | **Mean IoU ± SD** | **Mean Dice ± SD** |
|---------------------------------------------------|:-----------------:|:------------------:|
| SAM2 (bbox prompt, video propagation)             | 0.7215 ± 0.1287   | 0.7618 ± 0.1299    |
| SAM2 (bbox prompt, forced empty)                  | 0.9061 ± 0.0205   | 0.9465 ± 0.0122    |
| SAM2 (point prompt, video propagation)            | 0.5281 ± 0.1378   | 0.5937 ± 0.1408    |
| SAM2 (point prompt, forced empty)                 | 0.7575 ± 0.0509   | 0.8231 ± 0.0461    |
| SAM2 (vague-mask prompt, video propagation)       | 0.6319 ± 0.1054   | 0.7066 ± 0.1071    |
| SAM2 (vague-mask prompt, forced empty)            | 0.7977 ± 0.0394   | 0.8724 ± 0.0255    |
| Finetuned YOLO v11s w/ SAM2                       | 0.8460 ± 0.0333   | 0.8838 ± 0.0291    |
| Finetuned YOLO v11s w/ SAM2 (only slices w/ bbox) | 0.7877 ± 0.0441   | 0.8700 ± 0.0363    |
| SAMRefiner (smaller-mask initial prompt)          | 0.8049 ± 0.0384   | 0.8781 ± 0.0246    |
| SAMRefiner (smaller-mask refined mask)            | 0.8744 ± 0.0296   | 0.9242 ± 0.0193    |
| SAMRefiner (bigger-mask initial prompt)           | 0.6918 ± 0.0600   | 0.7795 ± 0.0431    |
| SAMRefiner (bigger-mask refined prompt)           | 0.7001 ± 0.0582   | 0.7857 ± 0.0423    |
| SAMRefiner (affine-mask initial prompt)           | 0.8084 ± 0.0344   | 0.8740 ± 0.0244    |
| SAMRefiner (affine-mask refined prompt)           | 0.8405 ± 0.0293   | 0.8968 ± 0.0209    |

- **Total slices with objects:** 3736  
- **Slices with object successfully detected:** 3248  

> ⚠️ Note: Including empty masks as perfect matches (IoU = 1.0) may inflate averages.  

---

## 🔁 Reproduce SAMRefiner Results

Here we use **smaller ellipse-shape masks** as input, and test different **prompt combinations** extracted from coarse masks.

| **Prompt Combination**           | **Mean IoU ± SD** | **Mean Dice ± SD** |
|----------------------------------|:-----------------:|:------------------:|
| Initial Coarse Input             | 0.8049 ± 0.0384   | 0.8781 ± 0.0246    |
| Points + Box + Mask              | 0.8744 ± 0.0296   | 0.9242 ± 0.0193    |
| Box + Mask                       | 0.8776 ± 0.0306   | 0.9248 ± 0.0206    |
| Point + Box                      | 0.8295 ± 0.0390   | 0.8889 ± 0.0283    |
| Point + Mask                     | 0.8357 ± 0.0369   | 0.8870 ± 0.0287    |
| Point Only                       | 0.7154 ± 0.0552   | 0.7675 ± 0.0533    |
| Box Only                         | 0.8231 ± 0.0410   | 0.8813 ± 0.0307    |
| Mask Only                        | 0.5194 ± 0.0838   | 0.5391 ± 0.0796    |

---