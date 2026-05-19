# -*- coding: utf-8 -*-
"""
Small-sample fine-tuning for the user's real laptop/camera environment.

Labels:
    data/me/attack -> 0 (Attack)
    data/me/real   -> 1 (Genuine)

This script intentionally saves a separate adapted model instead of replacing
the original CASIA-SURF baseline model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.model import DualHeadFASNet


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class FinetuneConfig:
    data_dir: str
    checkpoint: str
    output_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    val_ratio: float
    seed: int
    train_mode: str
    num_workers: int
    export_onnx: bool
    opset: int
    face_crop: bool
    face_margin: float
    context_margin: float
    crop_mode: str
    eval_crop_mode: str
    augment: bool
    attack_weight: float


class MeFASDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        train: bool,
        face_crop: bool,
        face_margin: float,
        context_margin: float,
        crop_mode: str,
        augment: bool,
    ):
        self.samples = samples
        self.train = train
        self.face_crop = face_crop
        self.face_margin = face_margin
        self.context_margin = context_margin
        self.crop_mode = crop_mode
        self.augment = augment
        self.face_detector = None
        if face_crop:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self.face_detector = cv2.CascadeClassifier(str(cascade_path))
            if self.face_detector.empty():
                self.face_detector = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        if self.face_crop:
            image = self._crop_by_mode(image, label)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self._augment(image) if self.train and self.augment else image
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (2, 0, 1))

        # The depth branch is kept for checkpoint compatibility, but the
        # fine-tuning loss only uses classification.
        depth = np.zeros((1, 32, 32), dtype=np.float32)
        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(depth).float(),
            torch.tensor(label, dtype=torch.long),
            image_path,
        )

    def _choose_crop_mode(self, label: int) -> str:
        if self.crop_mode != "mixed" or not self.train:
            return self.crop_mode

        # Attack cues often live outside the tight face crop: phone border,
        # screen glare, moire, and re-capture context. Keep those views frequent.
        if label == 0:
            return random.choices(
                ["context", "full", "face"],
                weights=[0.55, 0.30, 0.15],
                k=1,
            )[0]
        return random.choices(
            ["context", "face", "full"],
            weights=[0.45, 0.45, 0.10],
            k=1,
        )[0]

    def _crop_by_mode(self, image: np.ndarray, label: int) -> np.ndarray:
        mode = self._choose_crop_mode(label)
        if mode == "full":
            return image
        if mode == "face":
            return self._crop_face_or_center(image, self.face_margin)
        if mode == "context":
            return self._crop_face_or_center(image, self.context_margin)
        raise ValueError(f"Unsupported crop mode: {mode}")

    def _crop_face_or_center(self, image: np.ndarray, margin_ratio: float) -> np.ndarray:
        h, w = image.shape[:2]
        face = None
        if self.face_detector is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )
            if len(faces) > 0:
                face = max(faces, key=lambda box: int(box[2]) * int(box[3]))

        if face is None:
            side = min(h, w)
            x1 = max(0, (w - side) // 2)
            y1 = max(0, (h - side) // 2)
            return image[y1:y1 + side, x1:x1 + side]

        x, y, fw, fh = [int(v) for v in face]
        cx = x + fw / 2.0
        cy = y + fh / 2.0
        size = max(fw, fh) * margin_ratio
        x1 = max(0, int(round(cx - size / 2.0)))
        y1 = max(0, int(round(cy - size / 2.0)))
        x2 = min(w, int(round(cx + size / 2.0)))
        y2 = min(h, int(round(cy + size / 2.0)))
        if x2 <= x1 or y2 <= y1:
            return image
        return image[y1:y2, x1:x2]

    def _augment(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        if random.random() < 0.5:
            image = cv2.flip(image, 1)

        if random.random() < 0.8:
            alpha = random.uniform(0.60, 1.35)
            beta = random.uniform(-35, 28)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        if random.random() < 0.35:
            blur_kernel = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

        if random.random() < 0.35:
            noise = np.random.normal(0, random.uniform(3.0, 10.0), image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if random.random() < 0.5:
            angle = random.uniform(-8.0, 8.0)
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(
                image,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        if random.random() < 0.6:
            scale = random.uniform(0.82, 1.0)
            crop_h = max(16, int(h * scale))
            crop_w = max(16, int(w * scale))
            y0 = random.randint(0, max(0, h - crop_h))
            x0 = random.randint(0, max(0, w - crop_w))
            image = image[y0:y0 + crop_h, x0:x0 + crop_w]

        return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune FAS model on data/me samples.")
    parser.add_argument("--data_dir", default="data/me", help="Folder containing attack/ and real/.")
    parser.add_argument("--checkpoint", default="weights/best_model.pth", help="Base checkpoint.")
    parser.add_argument("--output_dir", default=None, help="Output folder. Default: weights/me_finetune_TIMESTAMP.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_mode",
        choices=["cls_head", "last_blocks", "all"],
        default="last_blocks",
        help="Trainable scope. last_blocks is recommended for the 100-photo adaptation set.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_export_onnx", action="store_true", help="Skip ONNX export.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no_face_crop", action="store_true", help="Disable Haar face crop before training.")
    parser.add_argument("--face_margin", type=float, default=1.50, help="Tight face crop expansion ratio.")
    parser.add_argument("--context_margin", type=float, default=2.60, help="Context crop expansion ratio for PAD cues.")
    parser.add_argument(
        "--crop_mode",
        choices=["mixed", "context", "face", "full"],
        default="mixed",
        help="Training crop mode. mixed uses context/full/face views.",
    )
    parser.add_argument(
        "--eval_crop_mode",
        choices=["context", "face", "full"],
        default="context",
        help="Validation crop mode. context should match GUI Face Crop Margin 0.80.",
    )
    parser.add_argument("--no_augment", action="store_true", help="Disable training-time image augmentation.")
    parser.add_argument(
        "--attack_weight",
        type=float,
        default=1.0,
        help="Class weight for Attack(label=0). Increase to reduce false genuine accepts.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def list_images(folder: Path, label: int) -> list[tuple[str, int]]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    samples = [
        (str(path), label)
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not samples:
        raise RuntimeError(f"No images found in: {folder}")
    return samples


def stratified_split(
    attack_samples: list[tuple[str, int]],
    real_samples: list[tuple[str, int]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    rng = random.Random(seed)
    attack_samples = attack_samples[:]
    real_samples = real_samples[:]
    rng.shuffle(attack_samples)
    rng.shuffle(real_samples)

    attack_val_count = max(1, int(round(len(attack_samples) * val_ratio)))
    real_val_count = max(1, int(round(len(real_samples) * val_ratio)))
    val_samples = attack_samples[:attack_val_count] + real_samples[:real_val_count]
    train_samples = attack_samples[attack_val_count:] + real_samples[real_val_count:]
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def load_base_model(checkpoint_path: Path, device: torch.device) -> tuple[DualHeadFASNet, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing model_state_dict")
    model = DualHeadFASNet(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    return model, checkpoint


def configure_trainable_layers(model: DualHeadFASNet, train_mode: str) -> list[str]:
    for param in model.parameters():
        param.requires_grad = False

    enabled: list[str] = []
    if train_mode == "cls_head":
        modules: Iterable[tuple[str, nn.Module]] = [("cls_head", model.cls_head)]
    elif train_mode == "last_blocks":
        backbone_children = list(model.backbone.named_children())
        modules = [(f"backbone.{name}", module) for name, module in backbone_children[-3:]]
        modules = list(modules) + [("cls_head", model.cls_head)]
    else:
        modules = [("model", model)]

    for name, module in modules:
        for param in module.parameters():
            param.requires_grad = True
        enabled.append(name)

    return enabled


def compute_pad_metrics(labels: list[int], preds: list[int]) -> dict[str, float]:
    labels_arr = np.array(labels, dtype=np.int64)
    preds_arr = np.array(preds, dtype=np.int64)
    total = max(1, len(labels_arr))
    correct = int((labels_arr == preds_arr).sum())

    attack_mask = labels_arr == 0
    real_mask = labels_arr == 1
    n_attack = max(1, int(attack_mask.sum()))
    n_real = max(1, int(real_mask.sum()))
    apcer = float(((preds_arr == 1) & attack_mask).sum() / n_attack)
    bpcer = float(((preds_arr == 0) & real_mask).sum() / n_real)
    acer = (apcer + bpcer) / 2.0
    return {
        "acc": correct / total,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
    }


def run_epoch(
    model: DualHeadFASNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if is_train:
        # Small personal datasets make BatchNorm running statistics unstable.
        # Keep BN layers in eval mode while still training the selected weights.
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

    total_loss = 0.0
    labels_all: list[int] = []
    preds_all: list[int] = []

    for images, _depths, labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits, _depth_pred = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        preds = torch.argmax(logits.detach(), dim=1)
        labels_all.extend(labels.detach().cpu().tolist())
        preds_all.extend(preds.cpu().tolist())

    metrics = compute_pad_metrics(labels_all, preds_all)
    metrics["loss"] = total_loss / max(1, len(labels_all))
    return metrics


def save_checkpoint(
    path: Path,
    model: DualHeadFASNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: dict[str, float],
    best_acer: float,
    config: FinetuneConfig,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_acc": metrics["acc"],
            "val_loss": metrics["loss"],
            "val_apcer": metrics["apcer"],
            "val_bpcer": metrics["bpcer"],
            "val_acer": metrics["acer"],
            "best_val_acer": best_acer,
            "config": asdict(config),
        },
        path,
    )


def export_to_onnx(checkpoint_path: Path, onnx_path: Path, opset: int) -> None:
    model, _checkpoint = load_base_model(checkpoint_path, torch.device("cpu"))
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["rgb"],
        output_names=["cls_logits", "depth_pred"],
        dynamic_axes={
            "rgb": {0: "batch"},
            "cls_logits": {0: "batch"},
            "depth_pred": {0: "batch"},
        },
    )


def verify_onnx(onnx_path: Path) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime is not installed, skip ONNX runtime verification.")
        return

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    cls_logits, depth_pred = session.run(None, {"rgb": dummy})
    if cls_logits.shape != (1, 2):
        raise RuntimeError(f"Unexpected cls_logits shape: {cls_logits.shape}")
    if depth_pred.shape != (1, 1, 32, 32):
        raise RuntimeError(f"Unexpected depth_pred shape: {depth_pred.shape}")
    print(f"[ONNX] verified: cls_logits={cls_logits.shape}, depth_pred={depth_pred.shape}")


def write_split_manifest(path: Path, train_samples: list[tuple[str, int]], val_samples: list[tuple[str, int]]) -> None:
    payload = {
        "label_map": {"attack": 0, "real": 1},
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train_samples": [{"path": item[0], "label": item[1]} for item in train_samples],
        "val_samples": [{"path": item[0], "label": item[1]} for item in val_samples],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parent
    data_dir = (project_root / args.data_dir).resolve() if not os.path.isabs(args.data_dir) else Path(args.data_dir)
    checkpoint_path = (project_root / args.checkpoint).resolve() if not os.path.isabs(args.checkpoint) else Path(args.checkpoint)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "weights" / f"me_finetune_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = FinetuneConfig(
        data_dir=str(data_dir),
        checkpoint=str(checkpoint_path),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_mode=args.train_mode,
        num_workers=args.num_workers,
        export_onnx=not args.no_export_onnx,
        opset=args.opset,
        face_crop=not args.no_face_crop,
        face_margin=args.face_margin,
        context_margin=args.context_margin,
        crop_mode=args.crop_mode,
        eval_crop_mode=args.eval_crop_mode,
        augment=not args.no_augment,
        attack_weight=args.attack_weight,
    )
    (output_dir / "config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")

    attack_samples = list_images(data_dir / "attack", 0)
    real_samples = list_images(data_dir / "real", 1)
    train_samples, val_samples = stratified_split(attack_samples, real_samples, args.val_ratio, args.seed)
    write_split_manifest(output_dir / "split_manifest.json", train_samples, val_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, base_checkpoint = load_base_model(checkpoint_path, device)
    trainable_modules = configure_trainable_layers(model, args.train_mode)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Check --train_mode.")

    train_loader = DataLoader(
        MeFASDataset(
            train_samples,
            train=True,
            face_crop=config.face_crop,
            face_margin=config.face_margin,
            context_margin=config.context_margin,
            crop_mode=config.crop_mode,
            augment=config.augment,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        MeFASDataset(
            val_samples,
            train=False,
            face_crop=config.face_crop,
            face_margin=config.face_margin,
            context_margin=config.context_margin,
            crop_mode=config.eval_crop_mode,
            augment=False,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    class_weights = torch.tensor([args.attack_weight, 1.0], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    log_path = output_dir / "finetune_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch", "lr",
            "train_loss", "train_acc", "train_apcer", "train_bpcer", "train_acer",
            "val_loss", "val_acc", "val_apcer", "val_bpcer", "val_acer",
        ])

    print("=" * 78)
    print("FAS personal-environment fine-tuning")
    print("=" * 78)
    print(f"Data:       {data_dir}")
    print(f"Attack:     {len(attack_samples)}")
    print(f"Real:       {len(real_samples)}")
    print(f"Split:      train={len(train_samples)}, val={len(val_samples)}")
    print(f"Base ckpt:  {checkpoint_path} (epoch={base_checkpoint.get('epoch')})")
    print(f"Output:     {output_dir}")
    print(f"Device:     {device}")
    print(f"Train mode: {args.train_mode} -> {', '.join(trainable_modules)}")
    print(f"Face crop:  {config.face_crop} (face_margin={config.face_margin}, context_margin={config.context_margin})")
    print(f"Crop mode:  train={config.crop_mode}, val={config.eval_crop_mode}")
    print(f"Augment:    {config.augment}")
    print(f"Class wt:   attack={args.attack_weight}, genuine=1.0")
    print("=" * 78)

    best_acer = float("inf")
    best_acc = -1.0
    best_path = output_dir / "best_model_me.pth"
    last_path = output_dir / "last_model_me.pth"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, None)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        is_best = (val_metrics["acer"] < best_acer - 1e-12) or (
            abs(val_metrics["acer"] - best_acer) <= 1e-12 and val_metrics["acc"] > best_acc
        )
        if is_best:
            best_acer = val_metrics["acer"]
            best_acc = val_metrics["acc"]
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, val_metrics, best_acer, config)

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, val_metrics, best_acer, config)

        with log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch, f"{lr:.8f}",
                f"{train_metrics['loss']:.6f}", f"{train_metrics['acc']:.6f}",
                f"{train_metrics['apcer']:.6f}", f"{train_metrics['bpcer']:.6f}", f"{train_metrics['acer']:.6f}",
                f"{val_metrics['loss']:.6f}", f"{val_metrics['acc']:.6f}",
                f"{val_metrics['apcer']:.6f}", f"{val_metrics['bpcer']:.6f}", f"{val_metrics['acer']:.6f}",
            ])

        mark = "*" if is_best else " "
        print(
            f"[{epoch:03d}/{args.epochs}] lr={lr:.7f} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"APCER={val_metrics['apcer']:.4f} BPCER={val_metrics['bpcer']:.4f} "
            f"ACER={val_metrics['acer']:.4f}{mark}"
        )

    if not best_path.exists():
        raise RuntimeError("Best checkpoint was not saved.")

    onnx_path = output_dir / "best_model_me.onnx"
    if config.export_onnx:
        try:
            export_to_onnx(best_path, onnx_path, args.opset)
            verify_onnx(onnx_path)
        except Exception as exc:
            print(f"[WARN] ONNX export/verification failed: {exc}")

    latest_ckpt = project_root / "weights" / "best_model_me.pth"
    latest_onnx = project_root / "weights" / "best_model_me.onnx"
    shutil.copy2(best_path, latest_ckpt)
    if onnx_path.exists():
        shutil.copy2(onnx_path, latest_onnx)

    print("=" * 78)
    print("[DONE] Fine-tuning finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Latest checkpoint copy: {latest_ckpt}")
    if onnx_path.exists():
        print(f"Best ONNX: {onnx_path}")
        print(f"Latest ONNX copy: {latest_onnx}")
    print(f"Log CSV: {log_path}")
    print(f"Split manifest: {output_dir / 'split_manifest.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
