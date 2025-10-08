#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, gc, csv, glob, logging, argparse, random
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# -------- CUDA + autocast (optional) --------
torch.set_grad_enabled(False)  # ensure no autograd graphs are created
# Optional (can also be set in shell): helps fragmentation for large tensors
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if torch.cuda.is_available():
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# -------- Repro --------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# -------- Logging --------
def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(str(log_path), mode="w"), logging.StreamHandler(sys.stdout)],
    )

# -------- I/O helpers --------
def read_rgb(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def read_bin_mask(p):
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None: raise FileNotFoundError(p)
    return (m > 127).astype(np.uint8)

def overlay(image_rgb, mask_bin, alpha=0.45):
    color = np.zeros_like(image_rgb); color[...,0] = 255
    m3 = np.repeat((mask_bin>0)[...,None], 3, axis=2).astype(np.uint8)
    over = (image_rgb*(1-alpha) + color*alpha).astype(np.uint8)
    out = image_rgb.copy(); out[m3==1] = over[m3==1]
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def add_header(img_bgr, text):
    bar_h = 36
    out = cv2.copyMakeBorder(img_bgr, bar_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))
    cv2.putText(out, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out

# -------- Metrics --------
def dice_iou(pred, gt):
    pred = (pred>0).astype(np.uint8)
    gt = (gt>0).astype(np.uint8)
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    a = int(pred.sum()); b = int(gt.sum())
    dice = (2*inter)/(a+b) if (a+b)>0 else (1.0 if a==b==0 else 0.0)
    iou  = (inter/union)  if union>0  else (1.0 if a==b==0 else 0.0)
    return float(dice), float(iou)

def mean_std(vals):
    arr = np.array(vals, dtype=np.float64)
    return (float(arr.mean()) if arr.size else 0.0,
            float(arr.std(ddof=0)) if arr.size else 0.0)

# -------- Prompts --------
def mask_to_bbox(mask, pad=2):
    ys, xs = np.where(mask>0)
    if xs.size==0: return None
    x1, x2 = xs.min(), xs.max(); y1, y2 = ys.min(), ys.max()
    x1 = max(0, x1-pad); y1 = max(0, y1-pad)
    x2 = min(mask.shape[1]-1, x2+pad); y2 = min(mask.shape[0]-1, y2+pad)
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def sample_pos_points(mask, n=20):
    pos = np.argwhere(mask>0)
    if len(pos)==0: return None
    if len(pos)>n:
        idx = np.random.choice(len(pos), n, replace=False)
        pos = pos[idx]
    pts = np.stack([pos[:,1], pos[:,0]], axis=1).astype(np.float32)  # (x,y)
    lbs = np.ones(len(pts), dtype=np.int32)
    return pts, lbs

def sample_neg_ring_points(mask, n=20, ring=6):
    """Sample negatives in a ring around the tumor bbox."""
    if not mask.any() or n <= 0: return None
    yx = np.argwhere(mask>0)
    y1,x1 = yx.min(0); y2,x2 = yx.max(0)
    H,W = mask.shape
    y1 = max(0, y1-ring); y2 = min(H-1, y2+ring)
    x1 = max(0, x1-ring); x2 = min(W-1, x2+ring)
    cand = []
    for y in range(y1, y2+1):
        for x in range(x1, x2+1):
            if mask[y,x] == 0:
                cand.append([x,y])
    if not cand: return None
    if len(cand) > n:
        idx = np.random.choice(len(cand), n, replace=False)
        cand = [cand[k] for k in idx]
    pts = np.array(cand, dtype=np.float32)
    lbs = np.zeros(len(pts), dtype=np.int32)
    return pts, lbs

# -------- Visualization Helpers --------
def show_mask(mask, ax, color=(1, 0, 1, 0.6)):
    """Overlay predicted mask in purple (RGBA)."""
    if mask is None or not mask.any():
        return
    h, w = mask.shape[-2:]
    mask_img = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_img)

def show_gt_mask(mask, ax, color=(1, 0, 0, 0.4)):
    """Overlay GT mask in red (RGBA)."""
    if mask is None or not mask.any():
        return
    h, w = mask.shape[-2:]
    mask_img = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_img)

def save_overlay_pair(frame_image, pred_mask, gt_mask, out_path, frame_idx=None, forced=False):
    """Save side-by-side overlay of Prediction vs Ground Truth using matplotlib."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    if frame_idx is not None:
        suptitle = f"Frame {frame_idx}" + (" (Forced-Empty)" if forced else "")
        fig.suptitle(suptitle, fontsize=14)

    # Left: Prediction
    ax[0].imshow(frame_image)
    ax[0].set_title("Prediction", fontsize=12)
    show_mask(pred_mask, ax[0])

    # Right: Ground Truth
    ax[1].imshow(frame_image)
    ax[1].set_title("Ground Truth", fontsize=12)
    show_gt_mask(gt_mask, ax[1])

    for a in ax:
        a.axis("off")

    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# -------- SAM2 video predictor --------
def get_video_predictor(config_path, ckpt_path):
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor(config_path, ckpt_path)

# -------- Per-PID run --------
def run_pid_video(predictor, pid_dir: Path, out_pred_dir: Path, out_overlay_dir: Path,
                  prompt_mode: str, vague_root: Path|None, forced_empty_eval: bool,
                  pos_points: int, neg_points: int, neg_ring: int, logit_thr: float):
    """
    prompt_mode ∈ {"bbox","points","vague-mask"}
      - bbox: derive bbox from GT tumor_masks
      - points: sample positive/negative points from GT tumor_masks
      - vague-mask: read vague masks from {vague_root}/{PID}/{PID}_{slice}.png (0/255)
    Saves BOTH:
      - normal predictions/overlays
      - forced-empty predictions/overlays (empty on slices without GT/prompt)
    """
    pid = pid_dir.name
    img_dir = pid_dir/"image_slices"
    gt_dir  = pid_dir/"tumor_masks"
    slice_paths = sorted(glob.glob(str(img_dir/f"{pid}_*.jpg")))
    assert len(slice_paths)>0, f"No slices for {pid}"

    # subdirs: normal vs forced-empty
    pred_norm_dir   = out_pred_dir/"normal"
    pred_forced_dir = out_pred_dir/"forced_empty"
    ov_norm_dir     = out_overlay_dir/"normal"
    ov_forced_dir   = out_overlay_dir/"forced_empty"
    for d in [pred_norm_dir, pred_forced_dir, ov_norm_dir, ov_forced_dir]:
        d.mkdir(parents=True, exist_ok=True)

    frames, gts, stems = [], [], []
    for p in slice_paths:
        stem = Path(p).stem
        frames.append(read_rgb(p)); stems.append(stem)
        gt_path = gt_dir/f"{stem}.png"  # your masks are .png
        gts.append(read_bin_mask(str(gt_path)) if gt_path.exists()
                   else np.zeros(frames[-1].shape[:2], dtype=np.uint8))

    # Stage frames as 000.jpg, 001.jpg… for SAM2 API
    staging = out_pred_dir.parent/"_staging"
    staging.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(frames):
        cv2.imwrite(str(staging/f"{i:03d}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    state = predictor.init_state(video_path=str(staging))

    # Add prompts
    has_prompt = [False] * len(gts)
    any_prompt_added = False
    with torch.inference_mode():
        for i, gt in enumerate(gts):
            if not gt.any():  # no prompt for empty GT slice
                continue
            pts, lbs, box, mask_for_sam = None, None, None, None

            if prompt_mode == "bbox":
                box = mask_to_bbox(gt)

            elif prompt_mode == "points":
                pos = sample_pos_points(gt, n=max(0, pos_points))
                neg = sample_neg_ring_points(gt, n=max(0, neg_points), ring=max(1, neg_ring)) if neg_points>0 else None
                if pos is not None:
                    pts, lbs = pos
                    if neg is not None:
                        npts, nlbs = neg
                        if npts.size > 0:
                            pts = np.concatenate([pts, npts], axis=0)
                            lbs = np.concatenate([lbs, nlbs], axis=0)

            elif prompt_mode == "vague-mask":
                assert vague_root is not None, "vague-mask requires --vague_root"
                vpath = vague_root/pid/f"{stems[i]}.png"
                if not vpath.exists():
                    vpath = vague_root/pid/f"{stems[i]}.jpg"
                if not vpath.exists():
                    continue
                mask_for_sam = read_bin_mask(str(vpath)).astype(np.uint8)*255

            else:
                raise ValueError(f"Unknown prompt_mode {prompt_mode}")

            if mask_for_sam is not None:
                predictor.add_new_mask(state, frame_idx=i, obj_id=1, mask=mask_for_sam)
                any_prompt_added = True
                has_prompt[i] = True
            else:
                if (pts is not None and len(pts)>0) or (box is not None):
                    predictor.add_new_points_or_box(state, frame_idx=i, obj_id=1, points=pts, labels=lbs, box=box)
                    any_prompt_added = True
                    has_prompt[i] = True

    # Propagate (normal predictions)
    segs = {}
    if any_prompt_added:
        with torch.inference_mode():
            for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state, start_frame_idx=0):
                # binarize -> cpu -> numpy, then free ASAP
                m = (mask_logits[0] > float(logit_thr)).detach().cpu().numpy()
                m = np.squeeze(m).astype(np.uint8)  # (H, W)
                segs[fidx] = m

                # free per-iteration CUDA tensors to reduce peak memory
                del mask_logits
                # (obj_ids are small ints, but del anyway)
                del obj_ids
                # occasionally flush allocator
                if (fidx % 10) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    else:
        logging.info(f"[{pid}] No prompts added. Skipping propagation and using empty predictions.")

    # Save + metrics (normal + forced-empty)
    per_slice_rows = []
    dices, ious = [], []
    forced_dices, forced_ious = [], []

    for i, stem in enumerate(stems):
        # --- normal ---
        pred = segs.get(i, np.zeros_like(gts[i], dtype=np.uint8))
        if pred.ndim == 3:
            pred = np.squeeze(pred)
        pred = np.ascontiguousarray(pred.astype(np.uint8))

        d_norm, j_norm = dice_iou(pred, gts[i])
        dices.append(d_norm)
        ious.append(j_norm)

        # save normal mask (0/255) + overlay (Prediction | GT)
        cv2.imwrite(str(pred_norm_dir / f"{stem}.png"), pred * 255)
        save_overlay_pair(frames[i], pred, gts[i], str(ov_norm_dir / f"{stem}.png"), frame_idx=i)

        # --- forced-empty (only empty when GT is empty) ---
        if forced_empty_eval:
            if gts[i].any():
                pred_forced = pred  # keep normal pred when GT non-empty
            else:
                pred_forced = np.zeros_like(gts[i], dtype=np.uint8)

            d_forced, j_forced = dice_iou(pred_forced, gts[i])
            forced_dices.append(d_forced)
            forced_ious.append(j_forced)

            # save forced mask (0/255) + overlay (Prediction (Forced-Empty) | GT)
            cv2.imwrite(str(pred_forced_dir / f"{stem}.png"), pred_forced * 255)
            save_overlay_pair(frames[i], pred_forced, gts[i], str(ov_forced_dir / f"{stem}.png"), frame_idx=i, forced=True)

        # per-slice row (include forced_* if evaluated)
        row = {
            "pid": pid,
            "slice": stem,
            "prompt_mode": prompt_mode,
            "has_gt": int(gts[i].any()),
            "dice": d_norm,
            "iou": j_norm,
        }
        if forced_empty_eval:
            row.update({
                "forced_dice": d_forced,
                "forced_iou": j_forced,
            })
        per_slice_rows.append(row)

        # clean up a little bit
        if (i % 20) == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate forced stats if needed
    forced_stats = None
    if forced_empty_eval:
        f_md, f_sd = mean_std(forced_dices)
        f_mi, f_sj = mean_std(forced_ious)
        forced_stats = (f_md, f_sd, f_mi, f_sj)

    # Cleanup
    predictor.reset_state(state)
    del state
    try:
        for f in staging.glob("*.jpg"):
            f.unlink()
        staging.rmdir()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    md, sd = mean_std(dices)
    mi, sj = mean_std(ious)
    return per_slice_rows, (md, sd, mi, sj), forced_stats

# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/yw2692/preprocess_3Dimages/Prostate-MRI-US-Biopsy",
                    help="Dataset root")
    ap.add_argument("--exp_name", required=True,
                    help="Subfolder under /home/yw2692/SAM2_outputs")
    ap.add_argument("--out_base", default="/home/yw2692/SAM2_outputs")
    ap.add_argument("--prompt_mode", default="bbox", choices=["bbox","points","vague-mask"])
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--checkpoint", default="/home/yw2692/workspace/Brain-MRI-Segmentation/sam2/checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--forced_empty_eval", action="store_true",
                    help="Also compute forced-empty variant (slices without prompts are forced empty).")
    ap.add_argument("--vague_root", type=str, default="",
                    help="Root dir that contains precomputed vague masks per PID (required for vague-mask).")

    # NEW arguments for point prompts & threshold
    ap.add_argument("--pos_points", type=int, default=20, help="Number of positive points per slice (inside tumor).")
    ap.add_argument("--neg_points", type=int, default=10,  help="Number of negative points per slice (outside tumor).")
    ap.add_argument("--neg_ring",   type=int, default=25,  help="Ring (pixels) around tumor bbox to sample negatives from.")
    ap.add_argument("--logit_thr",  type=float, default=0.0, help="Mask logit threshold for binarization.")

    args = ap.parse_args()
    set_seed(args.seed)

    exp_root = Path(args.out_base)/args.exp_name
    (exp_root/"logs").mkdir(parents=True, exist_ok=True)
    (exp_root/"csv").mkdir(parents=True, exist_ok=True)
    (exp_root/"pred_masks").mkdir(parents=True, exist_ok=True)
    (exp_root/"overlays").mkdir(parents=True, exist_ok=True)
    setup_logging(exp_root/"logs"/"exp.log")

    logging.info(f"Experiment {args.exp_name} | prompt={args.prompt_mode} | forced_empty={args.forced_empty_eval}")
    logging.info(f"Dataset root: {args.root}")
    if args.prompt_mode == "vague-mask":
        assert args.vague_root, "--vague_root is required for vague-mask"
    predictor = get_video_predictor(args.config, args.checkpoint)
    logging.info("Loaded SAM2 video predictor.")

    if torch.cuda.is_available():
        logging.info(f"CUDA available. Using device: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("CUDA not available. Falling back to CPU.")

    root = Path(args.root)
    pid_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    # Collectors
    all_rows = []               # per-slice records across ALL PIDs (for single CSV)
    pid_summ = []               # per-PID normal stats
    forced_pid_summ = []        # per-PID forced-empty stats (if enabled)

    vague_root = Path(args.vague_root) if args.vague_root else None

    for pid_dir in pid_dirs:
        pid = pid_dir.name
        if not (pid_dir/"image_slices").exists() or not (pid_dir/"tumor_masks").exists():
            logging.info(f"[skip] {pid} (missing image_slices or tumor_masks)")
            continue

        out_pred = exp_root/"pred_masks"/pid
        out_ov   = exp_root/"overlays"/pid

        rows, (md, sd, mi, sj), forced_stats = run_pid_video(
            predictor, pid_dir, out_pred, out_ov,
            prompt_mode=args.prompt_mode, vague_root=vague_root,
            forced_empty_eval=args.forced_empty_eval,
            pos_points=args.pos_points, neg_points=args.neg_points,
            neg_ring=args.neg_ring, logit_thr=args.logit_thr
        )
        if not rows:
            logging.info(f"[warn] {pid} produced no slices.")
            continue

        # Add a PID column is already inside rows; just append
        all_rows.extend(rows)

        # Per-PID summary (normal)
        pid_summ.append({
            "pid": pid, "prompt_mode": args.prompt_mode,
            "mean_iou": mi, "std_iou": sj, "mean_dice": md, "std_dice": sd,
            "num_slices": len(rows)
        })
        logging.info(f"[PID {pid} - Original Pred] IoU {mi:.4f} ± {sj:.4f} | Dice {md:.4f} ± {sd:.4f} ({len(rows)} slices)")

        # Per-PID forced summary
        if args.forced_empty_eval and forced_stats is not None:
            f_md, f_sd, f_mi, f_sj = forced_stats
            forced_pid_summ.append({
                "pid": pid, "prompt_mode": args.prompt_mode,
                "forced_mean_iou": f_mi, "forced_std_iou": f_sj,
                "forced_mean_dice": f_md, "forced_std_dice": f_sd,
                "num_slices": len(rows)
            })
            logging.info(f"[PID {pid} - Forced Empty] IoU {f_mi:.4f} ± {f_sj:.4f} | Dice {f_md:.4f} ± {f_sd:.4f} ({len(rows)} slices)")

    # ---------- Write one unified per-slice CSV across ALL PIDs ----------
    if all_rows:
        # Ensure consistent columns (include forced_* if present)
        base_fields = ["pid","slice","prompt_mode","has_gt","dice","iou"]
        forced_fields = ["forced_dice","forced_iou"] if args.forced_empty_eval else []
        fields = base_fields + forced_fields

        per_slice_all = exp_root/"csv"/"per_slice_all.csv"
        with open(per_slice_all, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in all_rows:
                row = {k: r.get(k, "") for k in fields}
                w.writerow(row)
        logging.info(f"[ALL] wrote per-slice CSV → {per_slice_all}")

    # ---------- Normal summaries ----------
    if pid_summ:
        # Pooled (from all rows)
        all_iou  = [r["iou"]  for r in all_rows]
        all_dice = [r["dice"] for r in all_rows]
        o_mi, o_sj = mean_std(all_iou)
        o_md, o_sd = mean_std(all_dice)

        # Per-PID + GLOBAL
        per_pid_path = exp_root/"csv"/"summary_per_pid.csv"
        with open(per_pid_path, "w", newline="") as f:
            fieldnames = ["pid","prompt_mode","mean_iou","std_iou","mean_dice","std_dice","num_slices"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in pid_summ:
                w.writerow(r)
            w.writerow({
                "pid": "GLOBAL",
                "prompt_mode": args.prompt_mode,
                "mean_iou": o_mi, "std_iou": o_sj,
                "mean_dice": o_md, "std_dice": o_sd,
                "num_slices": len(all_rows),
            })

        # Single-row overall
        with open(exp_root/"csv"/"summary_overall.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "prompt_mode","overall_mean_iou","overall_std_iou",
                "overall_mean_dice","overall_std_dice","num_total_slices"
            ])
            w.writeheader()
            w.writerow({
                "prompt_mode": args.prompt_mode,
                "overall_mean_iou":  o_mi, "overall_std_iou":  o_sj,
                "overall_mean_dice": o_md, "overall_std_dice": o_sd,
                "num_total_slices": len(all_rows),
            })
        logging.info(f"[ALL] IoU {o_mi:.4f} ± {o_sj:.4f} | Dice {o_md:.4f} ± {o_sd:.4f} over {len(all_rows)} slices")

    # ---------- Forced-empty summaries ----------
    if args.forced_empty_eval and forced_pid_summ:
        # pooled mean/std from per-PID forced stats
        def pooled_mean_std(entries, mean_key, std_key, n_key="num_slices"):
            N = sum(e[n_key] for e in entries)
            if N == 0:
                return 0.0, 0.0
            mu = sum(e[mean_key] * e[n_key] for e in entries) / N
            var = 0.0
            for e in entries:
                n = e[n_key]
                m = e[mean_key]
                s = e[std_key]
                var += n * ((s ** 2) + (m - mu) ** 2)
            var /= N
            return mu, float(np.sqrt(var))

        g_f_mi, g_f_sj = pooled_mean_std(forced_pid_summ, "forced_mean_iou",  "forced_std_iou")
        g_f_md, g_f_sd = pooled_mean_std(forced_pid_summ, "forced_mean_dice", "forced_std_dice")
        total_forced_slices = sum(e["num_slices"] for e in forced_pid_summ)

        forced_path = exp_root/"csv"/"summary_per_pid_forced.csv"
        with open(forced_path, "w", newline="") as f:
            fieldnames = ["pid","prompt_mode","forced_mean_iou","forced_std_iou",
                          "forced_mean_dice","forced_std_dice","num_slices"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in forced_pid_summ:
                w.writerow(r)
            w.writerow({
                "pid": "GLOBAL",
                "prompt_mode": args.prompt_mode,
                "forced_mean_iou":  g_f_mi, "forced_std_iou":  g_f_sj,
                "forced_mean_dice": g_f_md, "forced_std_dice": g_f_sd,
                "num_slices": total_forced_slices,
            })
        logging.info(f"[FORCED GLOBAL] IoU {g_f_mi:.4f} ± {g_f_sj:.4f} | Dice {g_f_md:.4f} ± {g_f_sd:.4f}")

if __name__ == "__main__":
    main()