# Brain-MRI-Segmentation

### To-Do List - Feb 26

- [ ] Check the code of video propogation
- [ ] Analyze slice-wise performance (use image processing directly)
- [ ] Create a table to visualize results from different prompts


# Mean IoU Scores for SAM Prompting Methods

**Mean IoU Scores** for different SAM prompting methods across 10 samples. Each method includes both the **overall mean IoU** and the **mean IoU from slices with prompts only**.

| Sample ID |       **Masks**        |       **Bounding Box**        |       **Points**        |
|-----------|------------------------|------------------------------|-------------------------|
|           | Overall | Slices with Prompts | Overall | Slices with Prompts | Overall | Slices with Prompts |
| 81088     | 0.4667  | 1.0000              | 0.3851  | 0.8251               | TBD     | TBD                 |
| 80593     | 0.6333  | 1.0000              | 0.5410  | 0.8542               | TBD     | TBD                 |
| 80139     | 0.4000  | 1.0000              | 0.3264  | 0.8160               | TBD     | TBD                 |
| 80421     | 0.6333  | 1.0000              | 0.5368  | 0.8476               | TBD     | TBD                 |
| 81099     | 0.4833  | 1.0000              | 0.4086  | 0.8454               | TBD     | TBD                 |
| 80900     | 0.4333  | 1.0000              | 0.3697  | 0.8531               | TBD     | TBD                 |
| 80179     | 0.5500  | 1.0000              | 0.4447  | 0.8086               | TBD     | TBD                 |
| 80701     | 0.3833  | 1.0000              | 0.3110  | 0.8112               | TBD     | TBD                 |
| 80076     | 0.7125  | 1.0000              | 0.5901  | 0.8281               | TBD     | TBD                 |
| 80889     | 0.5500  | 1.0000              | 0.4745  | 0.8627               | TBD     | TBD                 |


- **Overall**: IoU score considering all slices.
- **Slices with Prompts**: IoU score considering only slices where prompts were applied.
