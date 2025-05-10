# Brain-MRI-Segmentation


- [x] Use YOLO model to generate prompt
- [x] Try to run SAMRefiner


# Mean IoU Scores for SAM Prompting Methods

This table presents the **Mean IoU Scores** for different SAM prompting methods across 10 samples. Each method includes both the **overall mean IoU** and the **mean IoU from slices with prompts only**.

<table>
    <tr>
        <th rowspan="2">Sample ID</th>
        <th colspan="2">Masks (GT)</th>
        <th colspan="2">Perturbed Masks</th>
        <th colspan="2">Bounding Box</th>
        <th colspan="2">Points</th>
    </tr>
    <tr>
        <th>Overall</th>
        <th>Slices with Prompts</th>
        <th>Overall</th>
        <th>Slices with Prompts</th>
        <th>Overall</th>
        <th>Slices with Prompts</th>
        <th>Overall</th>
        <th>Slices with Prompts</th>
    </tr>
    <tr><td>81088</td><td>0.4667</td><td>1.0000</td><td>0.4040</td><td>0.8657</td><td>0.3851</td><td>0.8251</td><td>0.1884</td><td>0.4037</td></tr>
    <tr><td>80593</td><td>0.6333</td><td>1.0000</td><td>0.5650</td><td>0.8922</td><td>0.5410</td><td>0.8542</td><td>0.3188</td><td>0.5033</td></tr>
    <tr><td>80139</td><td>0.4000</td><td>1.0000</td><td>0.3434</td><td>0.8585</td><td>0.3264</td><td>0.8160</td><td>0.1195</td><td>0.2988</td></tr>
    <tr><td>80421</td><td>0.6333</td><td>1.0000</td><td>0.5409</td><td>0.8540</td><td>0.5368</td><td>0.8476</td><td>0.3635</td><td>0.5740</td></tr>
    <tr><td>81099</td><td>0.4833</td><td>1.0000</td><td>0.4103</td><td>0.8488</td><td>0.4086</td><td>0.8454</td><td>0.2584</td><td>0.5347</td></tr>
    <tr><td>80900</td><td>0.4333</td><td>1.0000</td><td>0.3813</td><td>0.8798</td><td>0.3697</td><td>0.8531</td><td>0.1831</td><td>0.4225</td></tr>
    <tr><td>80179</td><td>0.5500</td><td>1.0000</td><td>0.4734</td><td>0.8607</td><td>0.4447</td><td>0.8086</td><td>0.2578</td><td>0.4688</td></tr>
    <tr><td>80701</td><td>0.3833</td><td>1.0000</td><td>0.3250</td><td>0.8478</td><td>0.3110</td><td>0.8112</td><td>0.1909</td><td>0.4981</td></tr>
    <tr><td>80076</td><td>0.7125</td><td>1.0000</td><td>0.6389</td><td>0.8966</td><td>0.5901</td><td>0.8281</td><td>0.5080</td><td>0.7130</td></tr>
    <tr><td>80889</td><td>0.5500</td><td>1.0000</td><td>0.4717</td><td>0.8576</td><td>0.4745</td><td>0.8627</td><td>0.2810</td><td>0.5109</td></tr>
</table>


- **Overall**: IoU score considering all slices.
- **Slices with Prompts**: IoU score considering only slices where prompts were applied.
- **Average IoU Across 10 Cases**:
  - **Masks (Overall)**: 0.5246
  - **Perturbed Masks (Overall)**: 0.4554
  - **Bounding Box (Overall)**: 0.4388
  - **Points (Overall)**: 0.2670


# YOLO Detection Performance on Prostate-MRI Slices

Our fine-tuned YOLO model was evaluated on a dataset of **7,280 images** extracted from 3D MRI scans (validation set). The task was to detect the prostate region as a single object class. We computed standard object detection metrics—including Accuracy, Precision, Recall, and F1 Score—as well as metrics that measure the quality of the predicted bounding boxes using the **Average Best Intersection over Union (IoU)** and **Average Best Dice Coefficient**.

Note: We use IoU score to evaluate whether a prediction is matching the ground truth and we set the matching IoU threshold to be 0.5. The Average Best IoU and Average Best Dice both only take into account the matched sliced (counted as TP).

## Evaluation Results

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


### Interpretation

These results demonstrate a strong detection performance for the YOLO model. The high precision and bounding box overlap metrics (IoU and Dice) highlight that, for the detected slices, the predicted bounding boxes closely match the ground truth. However, the recall indicates that some slices with the target object are missed. This suggests there is room for further improvement in slice detection.
 
Based on these metrics, while the bounding box quality is excellent on the matched slices, improvement might be possible in detecting more slices (increasing recall) without compromising precision. Using a more complex model and data augmentation can be potential approachs to improve the performance.



# Performance Comparison

**Performance is evaluated with mean IoU/Dice with standard deviation (SD)**

| Method                                                     | Mean IoU ± SD   | Mean Dice ± SD  |
|:-----------------------------------------------------------|:---------------:|:---------------:|
| **SAM2 (bbox prompt, video propagation)**                  | 0.7215 ± 0.1287 | 0.7618 ± 0.1299 |
| **SAM2 (bbox prompt, forced empty)**                       | 0.9061 ± 0.0205 | 0.9465 ± 0.0122 |
| **SAM2 (point prompt, video propagation)**                 | 0.5281 ± 0.1378 | 0.5937 ± 0.1408 |
| **SAM2 (point prompt, forced empty)**                      | 0.7575 ± 0.0509 | 0.8231 ± 0.0461 |
| **SAM2 (vague‑mask prompt, video propagation)**            | 0.6319 ± 0.1054 | 0.7066 ± 0.1071 |
| **SAM2 (vague‑mask prompt, forced empty)**                 | 0.7977 ± 0.0394 | 0.8724 ± 0.0255 |
| **Finetuned YOLO v11s w/ SAM2**                            | 0.8460 ± 0.0333 | 0.8838 ± 0.0291 |
| **Finetuned YOLO v11s w/ SAM2 only slices with bbos**      | 0.7877 ± 0.0441 | 0.8700 ± 0.0363 |
| **SAMRefiner (smaller‑mask initial prompt)**               | 0.8049 ± 0.0384 | 0.8781 ± 0.0246 |
| **SAMRefiner (smaller‑mask refined mask)**                 | 0.8744 ± 0.0296 | 0.9242 ± 0.0193 |
| **SAMRefiner (bigger-mask initial prompt)**                | 0.6918 ± 0.0600 | 0.7795 ± 0.0431 |
| **SAMRefiner (bigger-mask refined prompt)**                | 0.7001 ± 0.0582 | 0.7857 ± 0.0423 |
| **SAMRefiner (affine-mask initial prompt)**                | 0.8084 ± 0.0344 | 0.8740 ± 0.0244 |
| **SAMRefiner (affine-mask refined prompt)**                | 0.8405 ± 0.0293 | 0.8968 ± 0.0209 |

- **Total slices with objects: 3736 & Slices with object successfully detected: 3248**  
- Should we include all the metrics performance for only averaging over the slices with prompt?
This may be a little bit lower probably because we count all empty masks as perfect match (1.0)



### Reproduce SAMRefiner Results

**Here we use smaller ecplise-shape masks as the input, and try different prompt combinations (extracted from the coarse masks)**


| Prompt Combinations                                        | Mean IoU ± SD   | Mean Dice ± SD  |
|:-----------------------------------------------------------|:---------------:|:---------------:|
| **Initial Coarse Input**                                   | 0.8049 ± 0.0384 | 0.8781 ± 0.0246 |
| **Points + Box + Mask**                                    | 0.8744 ± 0.0296 | 0.9242 ± 0.0193 |
| **Box + Mask**                                             | 0.8776 ± 0.0306 | 0.9248 ± 0.0206 |
| **Point + Box**                                            | 0.8295 ± 0.0390 | 0.8889 ± 0.0283 |
| **Point + Mask**                                           | 0.8357 ± 0.0369 | 0.8870 ± 0.0287 |
| **Point**                                                  | 0.7154 ± 0.0552 | 0.7675 ± 0.0533 |
| **Box**                                                    | 0.8231 ± 0.0410 | 0.8813 ± 0.0307 |
| **Mask**                                                   | 0.5194 ± 0.0838 | 0.5391 ± 0.0796 |