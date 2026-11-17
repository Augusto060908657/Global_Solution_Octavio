"""
Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696
GS FIAP 2026.2 - CitizenTech - Visão Computacional

demo_gallery.py - gera as DEMONSTRAÇÕES visuais (reais, produzidas pelo modelo treinado) para o relatório e o vídeo:
  reports/demo/teste_<fonte>.jpg   mosaico de imagens de TESTE de cada fonte (Brasil, RDD2022, ...), real (verde) x previsto
  reports/demo/externas_<pasta>.jpg mosaico das fotos externas (buracos, rachaduras, sem_danos) e de `minhas_fotos/`
  reports/demo/dashboard.png       painel-resumo: mAP por fonte, P/R/F1 por classe, latência/FPS por equipamento, curva do limiar
Uso: python demo_gallery.py [--yolo_dir data/yolo_mix] [--n 12] [--conf 0.25]
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import AUTHOR, CLASS_COLORS_BGR, CLASS_NAMES, CLASS_PT, EXTERNAL_DIR, REPORTS_DIR, ROOT, default_yolo_dir
from cv_service import default_conf, detect_image, draw_detections, get_device, get_model, model_imgsz
from evaluate import load_gt

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
CELL = (560, 360)


def letterbox(img, size=CELL):
    h, w = img.shape[:2]
    s = min(size[0] / w, size[1] / h)
    r = cv2.resize(img, (int(w * s), int(h * s)))
    canvas = np.full((size[1], size[0], 3), 30, np.uint8)
    canvas[(size[1] - r.shape[0]) // 2:(size[1] - r.shape[0]) // 2 + r.shape[0], (size[0] - r.shape[1]) // 2:(size[0] - r.shape[1]) // 2 + r.shape[1]] = r
    return canvas


def caption(cell, text):
    cv2.rectangle(cell, (0, 0), (cell.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(cell, text[:70], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def mosaic(cells, cols=3):
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    return np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])


def draw_gt(img, gt):
    for g in gt:
        cv2.rectangle(img, (int(g[1]), int(g[2])), (int(g[3]), int(g[4])), (0, 200, 0), 2)
    return img


def test_mosaics(yolo_dir: Path, n, conf, imgsz, out_dir, seed=42):
    rng = random.Random(seed)
    files = sorted((yolo_dir / "images" / "test").glob("*.jpg"))
    names = sorted((p.stem[len("test_"):] for p in yolo_dir.glob("test_*.txt")), key=len, reverse=True)
    sources = {}
    for f in files:
        src = next((n for n in names if f.name.startswith(n + "_")), "teste")
        sources.setdefault(src, []).append(f)
    for src, fs in sources.items():
        pick = rng.sample(fs, min(n, len(fs)))
        cells = []
        for f in pick:
            img = cv2.imread(str(f))
            dets, _ = detect_image(img, conf=conf, imgsz=imgsz)
            gt = load_gt(yolo_dir / "labels" / "test" / f"{f.stem}.txt", img.shape[1], img.shape[0])
            vis = draw_detections(draw_gt(img.copy(), gt), dets)
            cells.append(caption(letterbox(vis), f"{f.stem[:34]} | real {len(gt)} | previsto {len(dets)}"))
        cv2.imwrite(str(out_dir / f"teste_{src}.jpg"), mosaic(cells))
        print(f"[demo] teste_{src}.jpg ({len(cells)} imagens; verde = real, laranja/vermelho = previsto)")


def folder_mosaics(folders, n, conf, imgsz, out_dir, seed=42):
    rng = random.Random(seed)
    for name, folder in folders:
        fs = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)
        if not fs:
            continue
        cells = []
        for f in rng.sample(fs, min(n, len(fs))):
            img = cv2.imread(str(f))
            dets, timing = detect_image(img, conf=conf, imgsz=imgsz)
            cells.append(caption(letterbox(draw_detections(img, dets)), f"{f.name[:32]} | {len(dets)} defeito(s) | {timing.get('inference')} ms"))
        cv2.imwrite(str(out_dir / f"externas_{name}.jpg"), mosaic(cells))
        print(f"[demo] externas_{name}.jpg ({len(cells)} imagens)")


def dashboard(out_dir):
    mf = REPORTS_DIR / "metrics.json"
    if not mf.exists():
        print("[demo] metrics.json ausente: rode evaluate.py antes do painel"); return
    m = json.loads(mf.read_text())
    u, o = m["teste_ultralytics_conf0.001"], m["teste_limiar_operacional"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    # (a) mAP por fonte
    src = m.get("teste_por_fonte") or {"teste": {"mAP50": u["mAP50"], "mAP50_95": u["mAP50_95"]}}
    x = np.arange(len(src)); w = 0.38
    ax[0, 0].bar(x - w / 2, [v["mAP50"] for v in src.values()], w, label="mAP@50", color="#2a9d8f")
    ax[0, 0].bar(x + w / 2, [v["mAP50_95"] for v in src.values()], w, label="mAP@50-95", color="#e9c46a")
    ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(list(src), rotation=0); ax[0, 0].set_ylim(0, 1)
    ax[0, 0].set_title("mAP no TESTE por fonte de dados"); ax[0, 0].legend(); ax[0, 0].grid(alpha=.3, axis="y")
    # (b) P/R/F1 por classe
    cls = [CLASS_PT[k] for k in o["por_classe"]]; xx = np.arange(len(cls)); w = 0.26
    for i, (k, c) in enumerate((("precision", "#264653"), ("recall", "#e76f51"), ("f1", "#8ab17d"))):
        vals = [v[k] for v in o["por_classe"].values()]
        ax[0, 1].bar(xx + (i - 1) * w, vals, w, label=k, color=c)
        for xi, val in zip(xx + (i - 1) * w, vals):
            ax[0, 1].text(xi, val + 0.01, f"{val:.2f}", ha="center", fontsize=8)
    ax[0, 1].set_xticks(xx); ax[0, 1].set_xticklabels(cls); ax[0, 1].set_ylim(0, 1.1)
    ax[0, 1].set_title(f"Precision / Recall / F1 por classe (conf={o['conf']:.2f}, IoU≥0.5)"); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3, axis="y")
    # (c) latência
    bm = m.get("benchmark", {})
    if bm:
        labs = [(f"GPU {v['hardware']['gpu']}" if str(v["dispositivo"]).startswith("GPU") else f"CPU ({v['hardware']['cpu_threads']} threads)") + f"\n{v['imgsz']} px" for v in bm.values()]
        lat = [v["ponta_a_ponta"]["latencia_media_ms"] for v in bm.values()]
        fps = [v["ponta_a_ponta"]["fps"] for v in bm.values()]
        b = ax[1, 0].bar(labs, lat, color=["#457b9d", "#e63946"][:len(lat)])
        for rect, l, f in zip(b, lat, fps):
            ax[1, 0].text(rect.get_x() + rect.get_width() / 2, l, f"{l:.1f} ms\n{f:.1f} FPS", ha="center", va="bottom", fontsize=10)
        ax[1, 0].set_ylim(0, max(lat) * 1.35)
    ax[1, 0].set_title("Latência média por imagem (ponta a ponta)"); ax[1, 0].set_ylabel("ms"); ax[1, 0].grid(alpha=.3, axis="y")
    # (d) F1 x limiar
    cv = m["f1_vs_conf_validacao"]
    ax[1, 1].plot(cv["grid"], cv["macro_f1"], color="k")
    ax[1, 1].axvline(m["limiar_confianca_escolhido_na_validacao"], color="r", ls="--", label=f"escolhido = {m['limiar_confianca_escolhido_na_validacao']:.3f}")
    ax[1, 1].set_title("F1 médio × limiar de confiança (validação)"); ax[1, 1].set_xlabel("limiar"); ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)
    fig.suptitle(f"CitizenTech · Manutenção Urbana · YOLO11 ({m['weights']}, {m['imgsz']} px)  —  {AUTHOR}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out_dir / "dashboard.png", dpi=130); plt.close(fig)
    print("[demo] dashboard.png")


def main():
    ap = argparse.ArgumentParser(description=f"Demonstrações | {AUTHOR}")
    ap.add_argument("--yolo_dir", default=None)
    ap.add_argument("--n", type=int, default=12, help="imagens por mosaico")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--weights", default=None)
    a = ap.parse_args()
    out = REPORTS_DIR / "demo"; out.mkdir(parents=True, exist_ok=True)
    conf = default_conf() if a.conf is None else a.conf
    model = get_model(a.weights); imgsz = model_imgsz(model)
    yd = Path(a.yolo_dir) if a.yolo_dir else default_yolo_dir()
    print(f"[demo] dispositivo {get_device()} | imgsz {imgsz} | conf {conf:.3f}")
    test_mosaics(yd, a.n, conf, imgsz, out)
    folders = [(p.name, p) for p in sorted(EXTERNAL_DIR.glob("*")) if p.is_dir()]
    if (ROOT / "minhas_fotos").exists():
        folders.append(("minhas_fotos", ROOT / "minhas_fotos"))
    folder_mosaics(folders, a.n, conf, imgsz, out)
    dashboard(out)
    print(f"[ok] demonstrações em {out}/")


if __name__ == "__main__":
    main()
