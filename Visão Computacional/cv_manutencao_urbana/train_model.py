"""
Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696
GS FIAP 2026.2 - CitizenTech - Visão Computacional

train_model.py - Treinamento do detector YOLO11-nano (transfer learning a partir de pesos COCO).

Por que YOLO11n? Detector de estágio único com ~2,6 M de parâmetros e ~6,5 GFLOPs @640: roda em CPU, localiza
(bounding boxes) e classifica no mesmo passe, e os pesos COCO fornecem filtros de borda/textura reutilizáveis.
Saídas: weights/best.pt (melhor checkpoint por fitness = 0.1*mAP50 + 0.9*mAP50-95 na validação),
runs/<name>/ (curvas, results.csv), reports/training_curves.png.

Uso (CPU):
  python train_model.py --epochs 30 --imgsz 640 --batch 8
"""
import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import (AUTHOR, BEST_PT, PRETRAINED, REPORTS_DIR, RUNS_DIR, SEED, WEIGHTS_DIR, default_yolo_dir, ensure_pretrained,
                    write_dataset_yaml)


def plot_curves(results_csv: Path, out_png: Path):
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    specs = [(["train/box_loss", "val/box_loss"], "Perda de caixa (box)"),
             (["train/cls_loss", "val/cls_loss"], "Perda de classificação (cls)"),
             (["train/dfl_loss", "val/dfl_loss"], "Perda DFL"),
             (["metrics/precision(B)", "metrics/recall(B)"], "Precisão / Recall (validação)"),
             (["metrics/mAP50(B)"], "mAP@50 (validação)"),
             (["metrics/mAP50-95(B)"], "mAP@50-95 (validação)")]
    for a, (cols, title) in zip(ax.ravel(), specs):
        for c in cols:
            if c in df:
                a.plot(df["epoch"], df[c], marker="o", ms=3, label=c.split("/")[0] if "loss" in c else c.split("/")[1])
        a.set_title(title); a.set_xlabel("época"); a.grid(alpha=.3); a.legend()
    fig.suptitle("Curvas de treinamento - YOLO11n (buracos e rachaduras)")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


PRESETS = {
    "base": {},
    # augmentations mais fortes p/ generalizar a outros ângulos, escalas e iluminações
    "robust": dict(degrees=5.0, translate=0.1, scale=0.6, shear=2.0, perspective=0.0005, hsv_h=0.02, hsv_s=0.7,
                   hsv_v=0.5, mixup=0.1, erasing=0.3, cos_lr=True),
}


def main():
    ap = argparse.ArgumentParser(description=f"Treino YOLO | {AUTHOR}")
    ap.add_argument("--yolo_dir", default=None, help="padrão: data/yolo_mix se existir, senão data/yolo")
    ap.add_argument("--model", default="yolo11n", help="yolo11n (padrão), yolo11s, ... (pesos COCO baixados se faltarem)")
    ap.add_argument("--weights", default=None, help="caminho de pesos iniciais (sobrepõe --model)")
    ap.add_argument("--preset", choices=list(PRESETS), default="base", help="robust = augmentation mais forte")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0, help="fração do TREINO usada (acelera testes em CPU)")
    ap.add_argument("--patience", type=int, default=10, help="early stopping (épocas sem melhora)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="cpu ou 0 (GPU)")
    ap.add_argument("--name", default="train")
    ap.add_argument("--project", default=None, help="pasta das corridas (padrão: runs/). No Colab use uma pasta do DRIVE: "
                    "os checkpoints (last.pt a cada época) sobrevivem a quedas de sessão e permitem --resume")
    ap.add_argument("--backup_dir", default=None, help="copia best.pt/curvas para esta pasta ao final (ex.: Drive)")
    ap.add_argument("--resume", action="store_true", help="retoma runs/<name>/weights/last.pt")
    ap.add_argument("--no_default", action="store_true", help="não sobrescreve weights/best.pt (só weights/best_<name>.pt)")
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "False"])
    a = ap.parse_args()

    from ultralytics import YOLO
    runs_dir = Path(a.project) if a.project else RUNS_DIR
    run_dir = runs_dir / a.name
    yolo_dir = Path(a.yolo_dir) if a.yolo_dir else default_yolo_dir()
    data_yaml = write_dataset_yaml(yolo_dir)
    print(f"[train] dados: {yolo_dir} | modelo: {a.model} | preset: {a.preset} | device: {a.device}")
    if a.resume:
        YOLO(str(run_dir / "weights" / "last.pt")).train(resume=True)
    else:
        weights = Path(a.weights) if a.weights else PRETRAINED.parent / f"{a.model}.pt"
        weights = ensure_pretrained(weights)
        model = YOLO(str(weights))  # transfer learning: rede inteira é ajustada; cabeça de 2 classes reinicializada
        model.train(data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                    workers=a.workers, seed=SEED, deterministic=True, patience=a.patience, fraction=a.fraction,
                    project=str(runs_dir), name=a.name, exist_ok=True, plots=True, pretrained=True,
                    cache=False if a.cache == "False" else a.cache, close_mosaic=max(1, min(10, a.epochs // 3)),
                    fliplr=0.5, flipud=0.0, **PRESETS[a.preset])
    best = run_dir / "weights" / "best.pt"
    WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy2(best, WEIGHTS_DIR / f"best_{a.name}.pt")
    if not a.no_default:
        shutil.copy2(best, BEST_PT)
    REPORTS_DIR.mkdir(exist_ok=True)
    plot_curves(run_dir / "results.csv", REPORTS_DIR / f"training_curves{'' if a.name == 'train' else '_' + a.name}.png")
    if a.backup_dir:
        bk = Path(a.backup_dir); bk.mkdir(parents=True, exist_ok=True)
        for f in (WEIGHTS_DIR / f"best_{a.name}.pt", run_dir / "results.csv", run_dir / "args.yaml", run_dir / "results.png",
                  REPORTS_DIR / f"training_curves{'' if a.name == 'train' else '_' + a.name}.png"):
            if f.exists():
                shutil.copy2(f, bk / f.name)
        print(f"[backup] pesos e curvas copiados para {bk}")
    print(f"[ok] pesos: weights/best_{a.name}.pt" + ("" if a.no_default else " e weights/best.pt"))


if __name__ == "__main__":
    main()
