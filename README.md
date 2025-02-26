# Brain-MRI-Segmentation

### To-Do List - Feb 26

- [ ] Check the code of video propogation
- [ ] Analyze slice-wise performance (use image processing directly)
- [x] Create a table to visualize results from different prompts


# Mean IoU Scores for SAM Prompting Methods

This table presents the **Mean IoU Scores** for different SAM prompting methods across 10 samples. Each method includes both the **overall mean IoU** and the **mean IoU from slices with prompts only**.

<table>
    <tr>
        <th rowspan="2">Sample ID</th>
        <th colspan="2">Masks</th>
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
    </tr>
    <tr><td>81088</td><td>0.4667</td><td>1.0000</td><td>0.3851</td><td>0.8251</td><td>0.1884</td><td>0.4037</td></tr>
    <tr><td>80593</td><td>0.6333</td><td>1.0000</td><td>0.5410</td><td>0.8542</td><td>0.3188</td><td>0.5033</td></tr>
    <tr><td>80139</td><td>0.4000</td><td>1.0000</td><td>0.3264</td><td>0.8160</td><td>0.1195</td><td>0.2988</td></tr>
    <tr><td>80421</td><td>0.6333</td><td>1.0000</td><td>0.5368</td><td>0.8476</td><td>0.3635</td><td>0.5740</td></tr>
    <tr><td>81099</td><td>0.4833</td><td>1.0000</td><td>0.4086</td><td>0.8454</td><td>0.2584</td><td>0.5347</td></tr>
    <tr><td>80900</td><td>0.4333</td><td>1.0000</td><td>0.3697</td><td>0.8531</td><td>0.1831</td><td>0.4225</td></tr>
    <tr><td>80179</td><td>0.5500</td><td>1.0000</td><td>0.4447</td><td>0.8086</td><td>0.2578</td><td>0.4688</td></tr>
    <tr><td>80701</td><td>0.3833</td><td>1.0000</td><td>0.3110</td><td>0.8112</td><td>0.1909</td><td>0.4981</td></tr>
    <tr><td>80076</td><td>0.7125</td><td>1.0000</td><td>0.5901</td><td>0.8281</td><td>0.5080</td><td>0.7130</td></tr>
    <tr><td>80889</td><td>0.5500</td><td>1.0000</td><td>0.4745</td><td>0.8627</td><td>0.2810</td><td>0.5109</td></tr>
</table>


- **Overall**: IoU score considering all slices.
- **Slices with Prompts**: IoU score considering only slices where prompts were applied.
- **Average IoU Across 10 Cases**:
  - **Masks (Overall)**: 0.5246
  - **Bounding Box (Overall)**: 0.4388
  - **Points (Overall)**: 0.2670