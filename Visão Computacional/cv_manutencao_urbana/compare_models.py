"""
Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696
GS FIAP 2026.2 - CitizenTech - Visão Computacional

compare_models.py - compara 2+ modelos treinados no MESMO conjunto de teste (qualidade x velocidade), para justificar a
arquitetura (Atividade 5). Mede mAP@50, mAP@50-95, Precision, Recall, F1, parâmetros, tamanho do arquivo e latência/FPS
(ponta a ponta, warmup + iterações, time.perf_counter) no dispositivo escolhido. Saída: reports/COMPARACAO_MODELOS.md/.json
Uso: python compare_models.py --weights weights/best_train.pt weights/best_train_s.pt --labels YOLO11n YOLO11s --device 0
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common import AUTHOR, REPORTS_DIR, ROOT, default_yolo_dir, hardware_info, write_dataset_yaml
from evaluate import ultralytics_val


def latency(model, img, imgsz, device, n=50, warmup=10):
    for _ in range(warmup):
        model.predict(img, imgsz=imgsz, conf=0.25, device=device, verbose=False)
    ts = []
    for _ in range(n):
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, conf=0.25, device=device, verbose=False)
        if device != "cpu":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts)), float(1000 / np.mean(ts))


def main():
    ap = argparse.ArgumentParser(description=f"Comparação de modelos | {AUTHOR}")
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--yolo_dir", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from ultralytics import YOLO
    yd = Path(a.yolo_dir) if a.yolo_dir else default_yolo_dir()
    yaml_path = write_dataset_yaml(yd)
    sample = sorted((yd / "images" / "test").glob("*.jpg"))[0]
    img = cv2.imread(str(sample))
    labels = a.labels or [Path(w).stem for w in a.weights]
    rows = []
    for w, lab in zip(a.weights, labels):
        wp = Path(w) if Path(w).is_absolute() else ROOT / w
        m = YOLO(str(wp)); imgsz = int(m.ckpt.get("train_args", {}).get("imgsz", 640))
        u = ultralytics_val(m, yaml_path, "test", imgsz, f"cmp_{lab}", a.device)
        ms, fps = latency(m, img, imgsz, a.device)
        rows.append({"modelo": lab, "imgsz": imgsz, "parametros_M": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2),
                     "arquivo_MB": round(wp.stat().st_size / 1e6, 1), "mAP50": round(u["mAP50"], 4), "mAP50_95": round(u["mAP50_95"], 4),
                     "precision": round(u["precision"], 4), "recall": round(u["recall"], 4), "f1": round(u["f1"], 4),
                     "latencia_ms": round(ms, 1), "fps": round(fps, 1)})
    hw = hardware_info()
    dev = f"GPU {hw['gpu']}" if a.device != "cpu" else f"CPU {hw['cpu']}"
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "comparacao_modelos.json").write_text(json.dumps({"dispositivo": dev, "modelos": rows}, indent=2, ensure_ascii=False))
    cols = list(rows[0])
    md = [f"# Comparação de modelos (teste, {dev})\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    md.append("\nP/R no ponto de máximo F1 (Ultralytics, conf=0.001); latência ponta a ponta de 1 imagem no dispositivo indicado.")
    (REPORTS_DIR / "COMPARACAO_MODELOS.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
