"""
Adaptive Headlights Control with YOLO (Ultralytics)

Commands:
  python adaptive_headlights_yolo.py train
  python adaptive_headlights_yolo.py val
  python adaptive_headlights_yolo.py infer
  python adaptive_headlights_yolo.py figures

Requirements:
  pip install ultralytics opencv-python

For figures:
  pip install pandas matplotlib
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import cv2
from ultralytics import YOLO


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class TrainConfig:
    data_yaml: str = "data.yaml"
    pretrained_model: str = "yolov8n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    device: str = "cpu"  # "0" for first GPU, "cpu" for CPU
    project: str = "runs_headlight"
    name: str = "yolo_headlight"


@dataclass
class InferenceConfig:
    weights_path: str = "..."
    source: str = "..."
    conf: float = 0.10
    iou: float = 0.5
    imgsz: int = 640

    # Labels in YOUR dataset that mean "oncoming high beam detected"
    high_beam_labels: tuple[str, ...] = ("high beam",)

    # Output folder for annotated images
    out_dir: str = "predictions"


# ----------------------------
# Train / Validate
# ----------------------------

def train_yolo(cfg: TrainConfig) -> str:
    data_yaml = Path(cfg.data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Could not find dataset YAML at: {data_yaml.resolve()}\n"
            f"Tip: set TrainConfig.data_yaml to your Roboflow-exported data.yaml."
        )

    model = YOLO(cfg.pretrained_model)

    model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.project,
        name=cfg.name,
    )

    # Best weights path (note: Ultralytics may create name2/name3 folders if name exists)
    best_path = Path(cfg.project) / cfg.name / "weights" / "best.pt"
    if best_path.exists():
        print(f"Saved best weights to: {best_path.resolve()}")
        return str(best_path)

    print("Warning: best.pt not found where expected. Use 'figures' helper or search your run folder.")
    return str(best_path)


def validate_yolo(weights_path: str, data_yaml: str, device: str = "cpu") -> None:
    w = Path(weights_path)
    if not w.exists():
        raise FileNotFoundError(f"weights not found: {w.resolve()}")

    model = YOLO(str(w))
    metrics = model.val(data=data_yaml, device=device)
    print("Validation done. Metrics object:", metrics)


# ----------------------------
# Inference (folder of images) + "HIGH BEAM: TURN OFF/OK"
# ----------------------------

def run_inference(cfg: InferenceConfig) -> None:
    weights = Path(cfg.weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    src_path = Path(cfg.source)
    if not src_path.is_dir():
        raise RuntimeError(f"Source folder not found: {cfg.source}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    # Collect images recursively (handles .JPG etc.)
    images = [p for p in src_path.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
    if not images:
        raise RuntimeError(f"No image files found in: {cfg.source}")

    print(f"Found {len(images)} images. Running inference...")

    high_set = {s.lower().strip() for s in cfg.high_beam_labels}

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print("Could not read:", img_path)
            continue

        results = model.predict(img, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz, verbose=False)
        r0 = results[0]

        detected_names: List[str] = []
        if r0.boxes is not None and len(r0.boxes) > 0:
            for b in r0.boxes:
                cls_id = int(b.cls.item())
                name = model.names.get(cls_id, str(cls_id))
                detected_names.append(str(name))

        must_turn_off = any(n.lower().strip() in high_set for n in detected_names)

        annotated = r0.plot()

        decision_text = "HIGH BEAM: TURN OFF" if must_turn_off else "HIGH BEAM: OK"
        color = (0, 0, 255) if must_turn_off else (0, 255, 0)

        cv2.putText(
            annotated,
            decision_text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
            cv2.LINE_AA,
        )

        if detected_names:
            cv2.putText(
                annotated,
                "Detected: " + ", ".join(detected_names[:6]),
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), annotated)

    print(f"Done! Saved annotated images to: {out_dir.resolve()}")


# ----------------------------
# Figures (graphs for documentation)
# ----------------------------

def _find_latest_run_dir(project_dir: str, name_prefix: str) -> Path:
    """
    Finds newest run folder inside project_dir starting with name_prefix
    (e.g., yolo_headlight, yolo_headlight2, yolo_headlight3).
    """
    p = Path(project_dir)
    if not p.exists():
        raise FileNotFoundError(f"Project dir not found: {p}")

    candidates = [d for d in p.iterdir() if d.is_dir() and d.name.startswith(name_prefix)]
    if not candidates:
        raise FileNotFoundError(f"No run folders starting with '{name_prefix}' in {p}")

    return max(candidates, key=lambda d: d.stat().st_mtime)


def generate_doc_figures(run_dir: str, out_dir: str = "docs_figures") -> None:
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")  # <-- IMPORTANT: no GUI / no Tkinter needed
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("For figures, install: pip install pandas matplotlib") from e

    run_path = Path(run_dir)
    results_csv = run_path / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(
            f"results.csv not found at: {results_csv}\n"
            f"Point to the correct run folder (runs_headlight/<run_name>/)."
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)

    def plot_cols(title: str, cols: list[str], filename: str, ylabel: str = ""):
        available = [c for c in cols if c in df.columns]
        if not available:
            print(f"Skip {filename}: none of these columns exist: {cols}")
            return

        plt.figure()
        for c in available:
            plt.plot(df.index + 1, df[c], label=c)

        plt.title(title)
        plt.xlabel("Epoch")
        if ylabel:
            plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(out_path / filename, dpi=200)
        plt.close()
        print(f"Saved: {(out_path / filename).resolve()}")

    # Loss curves
    plot_cols(
        "Training Loss Curves",
        ["train/box_loss", "train/cls_loss", "train/dfl_loss", "train/seg_loss"],
        "loss_train.png",
        ylabel="Loss",
    )
    plot_cols(
        "Validation Loss Curves",
        ["val/box_loss", "val/cls_loss", "val/dfl_loss", "val/seg_loss"],
        "loss_val.png",
        ylabel="Loss",
    )

    # Metrics (newer names)
    plot_cols(
        "Detection Metrics",
        ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"],
        "metrics.png",
        ylabel="Score",
    )

    # Metrics (fallback names)
    plot_cols(
        "Detection Metrics (fallback columns)",
        ["metrics/precision", "metrics/recall", "metrics/mAP50", "metrics/mAP50-95", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95"],
        "metrics_fallback.png",
        ylabel="Score",
    )

    # Learning rate (if present)
    plot_cols(
        "Learning Rate",
        ["lr/pg0", "lr/pg1", "lr/pg2", "lr0"],
        "learning_rate.png",
        ylabel="LR",
    )

    # Copy Ultralytics auto-generated plots if they exist
    builtins = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "labels.jpg",
        "labels_correlogram.jpg",
    ]
    for b in builtins:
        src = run_path / b
        if src.exists():
            dst = out_path / b
            dst.write_bytes(src.read_bytes())
            print(f"Copied: {dst.resolve()}")

    print(f"\nAll figures saved in: {out_path.resolve()}")


# ----------------------------
# Main
# ----------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python adaptive_headlights_yolo.py [train|val|infer|figures]")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    train_cfg = TrainConfig(
        data_yaml=r"F:\adaptive_headlights_yolo\dataset\Yolo\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="cpu",
        project=r"F:\adaptive_headlights_yolo\runs_headlight",
        name="yolo_headlight",
    )

    infer_cfg = InferenceConfig(
        weights_path=r"F:\adaptive_headlights_yolo\runs_headlight\yolo_headlight4\weights\best.pt",  # best.pt also OK
        source=r"F:\adaptive_headlights_yolo\dataset\Yolo\test\images",
        conf=0.10,
        iou=0.5,
        imgsz=640,
        high_beam_labels=("high beam",),
        out_dir="predictions",
    )

    if cmd == "train":
        best = train_yolo(train_cfg)
        print("Training complete. Best weights (expected):", best)
        return

    if cmd == "val":
        validate_yolo(
            weights_path=infer_cfg.weights_path,  # validate the same weights you infer with
            data_yaml=train_cfg.data_yaml,
            device=train_cfg.device,
        )
        return

    if cmd == "infer":
        run_inference(infer_cfg)
        return

    if cmd == "figures":
        latest_run = _find_latest_run_dir(train_cfg.project, train_cfg.name)
        print(f"Using run folder: {latest_run}")
        generate_doc_figures(str(latest_run), out_dir="docs_figures")
        return

    raise ValueError("Unknown command. Use: train | val | infer | figures")


if __name__ == "__main__":
    main()
