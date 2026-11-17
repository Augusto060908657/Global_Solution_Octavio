"""
Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696
GS FIAP 2026.2 - CitizenTech - Visão Computacional

sequence_demo.py - "monitoramento urbano" por SEQUÊNCIA DE IMAGENS (Atividade 4 do enunciado: "vídeo curto OU sequência
de imagens"). Processa quadro a quadro, desenha caixas/categoria/confiança, contagem e FPS de processamento, e salva:
  results/sequence/frames/*.jpg   quadros anotados
  results/sequence/sequence_log.csv  (quadro, defeitos, buracos, rachaduras, confiança máx., ms por quadro)
  results/sequence/timeline.png, contact_sheet.jpg, sequence_summary.json
  (opcional) results/sequence/sequence.mp4 com --mp4

Fontes (--source):
  * omitido: usa quadros CONSECUTIVOS do mesmo trecho do conjunto de TESTE (nunca vistos no treino), escolhendo a
    janela de --n_frames quadros mais próximos entre si (fotos reais de levantamento de rodovia, em ordem);
  * pasta com imagens em ordem alfabética (ex.: fotos suas tiradas em sequência);
  * arquivo .mp4/.avi (opcional; --stride N processa 1 a cada N quadros).
Uso: python sequence_demo.py [--source pasta_ou_video] [--n_frames 40] [--conf 0.25]
"""
import argparse
import json
import re
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import AUTHOR, ROOT, default_yolo_dir
from cv_service import default_conf, detect_image, draw_detections, get_device, get_model, model_imgsz

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def seg_frame(stem: str):
    stem = re.sub(r"^[a-z0-9]+?_(?=\d+_[A-Z]{2}_)", "", stem)  # remove prefixo de fonte (ex.: br_)
    parts = stem.split("_")
    m = re.search(r"(\d+)$", parts[-1])
    return (parts[3] if len(parts) >= 5 else "unk"), (int(m.group(1)) if m else 0)


def pick_test_sequence(yolo_dir: Path, n_frames: int):
    files = sorted((yolo_dir / "images" / "test").glob("*.jpg"))
    if (yolo_dir / "test_br.txt").exists():          # dataset combinado: só quadros do Brasil (sequências reais de levantamento)
        files = [f for f in files if f.name.startswith("br_")]
    by_seg = {}
    for f in files:
        seg, fr = seg_frame(f.stem)
        by_seg.setdefault(seg, []).append((fr, f))
    best = None
    for seg, items in by_seg.items():
        items.sort()
        for i in range(0, max(1, len(items) - n_frames + 1)):
            win = items[i:i + n_frames]
            span = win[-1][0] - win[0][0]
            if best is None or (len(win), -span) > (len(best[1]), -best[2]):
                best = (seg, win, span)
    if best is None:
        raise SystemExit("Sem imagens de teste do dataset brasileiro em " + str(yolo_dir))
    print(f"[sequência] trecho {best[0]}: {len(best[1])} quadros consecutivos (números {best[1][0][0]} a {best[1][-1][0]})")
    return [f for _, f in best[1]]


def frames_from_source(src: Path, stride: int):
    if src.is_dir():
        for f in sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)[::stride]:
            yield f.name, cv2.imread(str(f))
    elif src.suffix.lower() in VID_EXT:
        cap = cv2.VideoCapture(str(src)); i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield f"frame_{i:05d}", fr
            i += 1
        cap.release()
    else:
        raise SystemExit("--source deve ser uma pasta de imagens ou um arquivo de vídeo")


def header(img, idx, n, dets, fps, ms):
    h, w = img.shape[:2]
    bar = max(34, h // 18)
    col = (0, 0, 200) if dets else (0, 150, 0)
    cv2.rectangle(img, (0, 0), (w, bar), col, -1)
    sc = bar / 60
    mx = max((d["confidence"] for d in dets), default=0)
    txt = f"Quadro {idx}/{n} | Defeitos: {len(dets)} | Conf. max: {mx * 100:.0f}%" if dets else f"Quadro {idx}/{n} | Sem defeitos detectados"
    cv2.putText(img, txt, (10, int(bar * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, sc * 0.9), (255, 255, 255), 2, cv2.LINE_AA)
    r = f"{fps:4.1f} FPS | {ms:4.0f} ms"
    (tw, _), _ = cv2.getTextSize(r, cv2.FONT_HERSHEY_SIMPLEX, max(0.5, sc * 0.9), 2)
    cv2.putText(img, r, (w - tw - 10, int(bar * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, sc * 0.9), (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=f"Sequência de imagens | {AUTHOR}")
    ap.add_argument("--source", default=None)
    ap.add_argument("--yolo_dir", default=None)
    ap.add_argument("--n_frames", type=int, default=40)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="results/sequence")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--mp4", action="store_true", help="também grava sequence.mp4 (opcional)")
    a = ap.parse_args()

    out = Path(a.out); (out / "frames").mkdir(parents=True, exist_ok=True)
    conf = default_conf() if a.conf is None else a.conf
    model = get_model(a.weights)
    imgsz = a.imgsz or model_imgsz(model)
    if a.source:
        src = Path(a.source)
        items = list(frames_from_source(src, a.stride))
    else:
        yd = Path(a.yolo_dir) if a.yolo_dir else default_yolo_dir()
        items = [(f.name, cv2.imread(str(f))) for f in pick_test_sequence(yd, a.n_frames)[::a.stride]]
    n = len(items)
    log, writer, fps_ema, t_total = [], None, None, 0.0
    for i, (name, img) in enumerate(items, 1):
        t0 = time.perf_counter()
        dets, timing = detect_image(img, conf=conf, imgsz=imgsz, weights=a.weights)
        ann = draw_detections(img, dets)
        dt = time.perf_counter() - t0                      # decodificação já feita; inclui inferência, NMS e desenho
        t_total += dt
        fps_ema = 1 / dt if fps_ema is None else 0.8 * fps_ema + 0.2 * (1 / dt)
        ann = header(ann, i, n, dets, fps_ema, dt * 1000)
        cv2.imwrite(str(out / "frames" / f"{i:04d}.jpg"), ann)
        if a.mp4:
            if writer is None:
                writer = cv2.VideoWriter(str(out / "sequence.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (ann.shape[1], ann.shape[0]))
            writer.write(ann)
        log.append({"quadro": i, "arquivo": name, "defeitos": len(dets), "buracos": sum(d["class_en"] == "pothole" for d in dets),
                    "rachaduras": sum(d["class_en"] == "crack" for d in dets),
                    "conf_max": max((d["confidence"] for d in dets), default=0.0),
                    "inferencia_ms": timing.get("inference"), "tempo_quadro_ms": round(dt * 1000, 1)})
    if writer:
        writer.release()
    df = pd.DataFrame(log)
    df.to_csv(out / "sequence_log.csv", index=False)

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].bar(df.quadro, df.rachaduras, label="Rachaduras", color="orange")
    ax[0].bar(df.quadro, df.buracos, bottom=df.rachaduras, label="Buracos", color="red")
    ax[0].set_ylabel("defeitos por quadro"); ax[0].legend(); ax[0].set_title("Monitoramento por sequência de imagens")
    ax[1].plot(df.quadro, df.tempo_quadro_ms, color="k"); ax[1].set_ylabel("ms por quadro"); ax[1].set_xlabel("quadro")
    for x in ax:
        x.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out / "timeline.png", dpi=140); plt.close()

    sel = sorted(set(np.linspace(1, n, min(12, n)).astype(int).tolist()))
    thumbs = [cv2.resize(cv2.imread(str(out / "frames" / f"{k:04d}.jpg")), (480, 300)) for k in sel]
    while len(thumbs) % 3:
        thumbs.append(np.zeros_like(thumbs[0]))
    cv2.imwrite(str(out / "contact_sheet.jpg"), np.vstack([np.hstack(thumbs[j:j + 3]) for j in range(0, len(thumbs), 3)]))

    summ = {"quadros": n, "quadros_com_defeito": int((df.defeitos > 0).sum()), "total_defeitos": int(df.defeitos.sum()),
            "fps_processamento_medio": float(n / t_total), "latencia_media_ms": float(t_total / n * 1000),
            "inferencia_media_ms": float(df.inferencia_ms.mean()), "conf_usada": conf, "imgsz": imgsz,
            "dispositivo": get_device(), "observacao": "tempo por quadro inclui inferência, NMS e desenho; não inclui gravação em disco"}
    (out / "sequence_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print(json.dumps(summ, indent=2, ensure_ascii=False))
    print(f"[ok] {out}/ (frames/, sequence_log.csv, timeline.png, contact_sheet.jpg)")


if __name__ == "__main__":
    main()
