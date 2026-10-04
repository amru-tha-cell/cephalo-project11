"""
Training script for the ISBI 2015 Cephalometric X-ray Dataset.

Trains the exact Swin Transformer Encoder -> Attention U-Net Decoder used by the
inference API (app/ml/model.py), so the saved weights are drop-in compatible:
the best checkpoint is written to settings.WEIGHTS_PATH (weights/swin_attn_unet.pt),
which `build_model()` loads automatically.

Expected dataset layout under --data-root (see app/ml/dataset.py):
    RawImage/{TrainingData,Test1Data,Test2Data}/*.bmp
    AnnotationsByMD/{400_junior,400_senior}/*.txt

Run from the backend/ folder:
    python training/train.py --data-root data/raw --epochs 100
    python training/train.py --data-root data/raw --annotator mean --amp      # GPU, mixed precision
    python training/train.py --data-root data/raw --resume                    # continue after a stop
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import settings
from app.ml.coordinate_extraction import extract_coordinates
from app.ml.dataset import ISBICephalometricDataset
from app.ml.metrics import compute_mre, compute_sdr, landmark_wise_error
from app.ml.model import SwinAttentionUNet


# --------------------------------------------------------------------- losses
def gaussian_heatmap_loss(pred_heatmaps, target_heatmaps):
    """Penalty-reduced focal loss (CenterNet / CornerNet) on sigmoid(pred) vs the Gaussian
    targets. Heatmaps are ~99.9% background, so plain MSE gives almost no signal for the
    peaks; this loss is the standard fix. Normalised by the number of landmark peaks.
    The target's peak pixel is exactly 1.0 (guaranteed by the dataset)."""
    logits = pred_heatmaps.float()
    log_p = nn.functional.logsigmoid(logits)
    log_1mp = nn.functional.logsigmoid(-logits)
    p = torch.sigmoid(logits)
    pos = (target_heatmaps >= 1.0).float()
    pos_loss = -((1 - p) ** 2) * log_p * pos
    neg_loss = -((1 - target_heatmaps) ** 4) * (p ** 2) * log_1mp * (1 - pos)
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)


def offset_loss(pred_offsets, target_offsets, mask):
    """Smooth-L1 on (dx, dy), averaged over the supervised pixels only (mask == 1)."""
    per_pixel = nn.functional.smooth_l1_loss(pred_offsets, target_offsets, reduction="none") * mask
    return per_pixel.sum() / mask.sum().clamp(min=1.0)


# ----------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, device, offset_weight):
    """Runs the model in eval mode (dropout off) and scores it the same way the API does:
    coordinate extraction + sub-pixel refinement -> MRE / SDR in mm on ORIGINAL pixels."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    errors = {}                     # (image_idx, landmark_idx) -> error in mm
    per_landmark = {}               # landmark_idx -> [errors]
    img_counter = 0

    for batch in loader:
        image = batch["image"].to(device)
        heat_t = batch["heatmaps"].to(device)
        off_t = batch["offsets"].to(device)
        mask = batch["offset_mask"].to(device)

        heat_p, off_p = model(image)
        total_loss += (gaussian_heatmap_loss(heat_p, heat_t)
                       + offset_weight * offset_loss(off_p, off_t, mask)).item()
        n_batches += 1

        for b in range(image.shape[0]):
            sx, sy = batch["scale"][b].tolist()
            spacing = float(batch["pixel_spacing_mm"][b])
            preds = extract_coordinates(heat_p[b], off_p[b], sx, sy)
            gt = batch["orig_keypoints"][b]
            for p in preds:
                i = p["landmark_index"]
                err = float(np.hypot(p["x"] - gt[i, 0].item(), p["y"] - gt[i, 1].item())) * spacing
                errors[(img_counter, i)] = err
                per_landmark.setdefault(i, []).append(err)
            img_counter += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mre": compute_mre(errors),
        "sdr": compute_sdr(errors),
        "per_landmark": landmark_wise_error({i: float(np.mean(v)) for i, v in per_landmark.items()}),
    }


def print_metrics(tag, m):
    sdr = "  ".join(f"SDR@{k} {v * 100:5.1f}%" for k, v in m["sdr"].items())
    print(f"[{tag}] loss {m['loss']:.5f} | MRE {m['mre']:.3f} mm | {sdr}")


# ----------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Train Swin + Attention U-Net on ISBI 2015")
    parser.add_argument("--data-root", default=os.getenv("ISBI_ROOT", "./data/raw"),
                        help="Folder containing RawImage/ and AnnotationsByMD/")
    parser.add_argument("--annotator", default="senior", choices=["junior", "senior", "mean"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--offset-weight", type=float, default=1.0)
    parser.add_argument("--val-size", type=int, default=15,
                        help="Images held out of the 150 training images for validation (0 = use test1)")
    parser.add_argument("--image-size", type=int, default=settings.IMAGE_SIZE,
                        help="Keep 512 for the API; smaller values are for quick CPU tests only")
    parser.add_argument("--heatmap-sigma", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Use only N images per split (smoke test)")
    parser.add_argument("--amp", action="store_true", help="Mixed precision (CUDA only)")
    parser.add_argument("--resume", action="store_true", help="Resume from <out>.last.pt")
    parser.add_argument("--skip-test", action="store_true", help="Skip final test1/test2 evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=settings.WEIGHTS_PATH)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.amp and device == "cuda"
    print(f"[train] device={device} amp={use_amp} data_root={args.data_root} annotator={args.annotator}")

    # ---- data
    ds_kwargs = dict(data_root=args.data_root, annotator=args.annotator, image_size=args.image_size,
                     heatmap_sigma=args.heatmap_sigma, val_size=args.val_size, seed=args.seed,
                     limit=args.limit)
    train_ds = ISBICephalometricDataset(split="train", **ds_kwargs)
    val_split = "val" if args.val_size > 0 else "test1"
    val_ds = ISBICephalometricDataset(split=val_split, **ds_kwargs)
    print(f"[train] {len(train_ds)} train images | {len(val_ds)} '{val_split}' images for validation")

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    # ---- model / optimiser
    model = SwinAttentionUNet().to(device)
    if not args.resume:
        # start with sigmoid(heatmap) ~ 0.01 (standard for focal loss); avoids a huge initial loss
        nn.init.constant_(model.decoder.heatmap_head.bias, -4.6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    last_path = args.out + ".last.pt"
    start_epoch, best_mre = 0, float("inf")
    if args.resume and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_mre = ckpt["epoch"] + 1, ckpt["best_mre"]
        print(f"[train] resumed from {last_path} at epoch {start_epoch} (best MRE {best_mre:.3f} mm)")

    # ---- training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            heat_t = batch["heatmaps"].to(device, non_blocking=True)
            off_t = batch["offsets"].to(device, non_blocking=True)
            mask = batch["offset_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                heat_p, off_p = model(image)
            # losses in fp32 for stability
            loss = (gaussian_heatmap_loss(heat_p.float(), heat_t)
                    + args.offset_weight * offset_loss(off_p.float(), off_t, mask))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()

        metrics = evaluate(model, val_loader, device, args.offset_weight)
        print(f"epoch {epoch + 1:3d}/{args.epochs} | train loss {running / len(train_loader):.5f} | "
              f"val MRE {metrics['mre']:.3f} mm | SDR@2.0mm {metrics['sdr'].get('2.0mm', 0) * 100:5.1f}% | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s")

        if metrics["mre"] < best_mre:
            best_mre = metrics["mre"]
            torch.save(model.state_dict(), args.out)       # plain state_dict -> build_model() loads it
            print(f"           ** new best (MRE {best_mre:.3f} mm) -> saved {args.out}")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best_mre": best_mre}, last_path)

    # ---- final evaluation with the best weights
    if os.path.isfile(args.out):
        model.load_state_dict(torch.load(args.out, map_location=device))
    print("\n[train] Final evaluation with best weights")
    print_metrics(f"{val_split}", evaluate(model, val_loader, device, args.offset_weight))
    if not args.skip_test:
        for split in ("test1", "test2"):
            try:
                ds = ISBICephalometricDataset(split=split, **{**ds_kwargs, "val_size": 0})
            except FileNotFoundError as e:
                print(f"[{split}] skipped: {e.args[0].splitlines()[0]}")
                continue
            m = evaluate(model, DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers), device,
                         args.offset_weight)
            print_metrics(split, m)
            worst = list(m["per_landmark"].items())[:3]
            print("         hardest landmarks: " + ", ".join(f"{k} ({v:.2f} mm)" for k, v in worst))
    print(f"\n[train] Done. Best weights: {args.out}")


if __name__ == "__main__":
    main()