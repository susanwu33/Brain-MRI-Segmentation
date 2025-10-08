import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from sam_refiner import sam_refiner
from segment_anything.utils.transforms import ResizeLongestSide
from pytorch3dunet.unet3d.trainer import UNetTrainer

class IntegratedUNetTrainer(UNetTrainer):
    def __init__(self, sam_model, image_slice_root, slice_suffix=".png", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam = sam_model
        self.sam.eval()
        for p in self.sam.parameters():
            p.requires_grad = False

        self.resize_transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        self.image_slice_root = image_slice_root  
        self.slice_suffix = slice_suffix

    def _forward_pass(self, inp: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 3D UNet forward pass
        coarse_mask, logits = self.model(inp, return_logits=True)  # (B, 1, D, H, W)
        batch_size, _, depth, H, W = coarse_mask.shape

        refined_outputs = []

        for b in range(batch_size):
            pid = self.loaders['train'].dataset.subjects[b]  # Assuming subjects[b] gives pid for b-th item
            pid_slice_dir = os.path.join(self.image_slice_root, pid, "image_slices")

            refined_volume = torch.zeros_like(coarse_mask[b, 0])
            for d in range(depth):
                pred_mask = coarse_mask[b, 0, d].detach().cpu().numpy()
                pred_mask_norm = (pred_mask > 0.5).astype(np.uint8)

                slice_name = f"{pid}_{d:03d}{self.slice_suffix}"
                slice_path = os.path.join(pid_slice_dir, slice_name)
                if not os.path.exists(slice_path):
                    continue

                refined_mask, _, _ = sam_refiner(
                    image_path=slice_path,
                    coarse_masks=[pred_mask_norm],
                    sam=self.sam,
                    resize_transform=self.resize_transform,
                    is_train=True,
                    use_point=True,
                    use_box=True,
                    use_mask=True,
                    gamma=4.0,
                    strength=30,
                )
                refined_mask_tensor = torch.from_numpy(refined_mask[0]).float()
                refined_volume[d] = refined_mask_tensor

            refined_outputs.append(refined_volume.unsqueeze(0))  # (1, D, H, W)

        refined_tensor = torch.stack(refined_outputs).to(inp.device)  # (B, 1, D, H, W)
        loss = self.loss_criterion(refined_tensor, target)
        return refined_tensor, loss