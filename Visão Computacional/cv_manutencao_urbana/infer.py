"""
Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696
GS FIAP 2026.2 - CitizenTech - Visão Computacional

infer.py - inferência em IMAGENS (arquivo único ou pasta, recursivo). Sem vídeo/webcam.

Para cada imagem salva em --out/<nome>/:
  original.jpg | deteccao.jpg (bounding boxes + categoria + confiança) | comparativo.jpg (lado a lado) | resultado.json
(categoria de cada defeito, caixas, confiança, quantidade de problemas, tempo de inferência) e, na raiz de --out,
summary.csv. Se as imagens estiverem em subpastas buracos/ rachaduras/ sem_danos/, gera também a tabela
"esperado x detectado" (--expected_from_folders) e imprime o resumo.

Uso:
  python infer.py --source tests/external_images --out results/external --expected_from_folders
  python infer.py --source foto.jpg
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import AUTHOR, EXTERNAL_DIR, ROOT
from cv_service import decode_image, default_conf, detect_image, draw_detections, get_model

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
EXPECT = {"buracos": "pothole", "rachaduras": "crack", "sem_danos": None}


def side_by_side(orig, det, height=520):
    def rs(im):
        return cv2.resize(im, (int(im.shape[1] * height / im.shape[0]), height))
    a, b = rs(orig), rs(det)
    for im, t in ((a, "ORIGINAL"), (b, "DETECCAO")):
        cv2.rectangle(im, (0, 0), (150, 28), (0, 0, 0), -1)
        cv2.putText(im, t, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return np.hstack([a, np.full((height, 6, 3), 255, np.uint8), b])


def main():
    ap = argparse.ArgumentParser(description=f"Inferência em imagens | {AUTHOR}")
    ap.add_argument("--source", default=str(EXTERNAL_DIR.relative_to(ROOT)))
    ap.add_argument("--out", default="results/inference")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=None, help="padrão: weights/threshold.json (ou 0.25)")
    ap.add_argument("--imgsz", type=int, default=None, help="padrão: o usado no treino (lido do best.pt)")
    ap.add_argument("--expected_from_folders", action="store_true")
    a = ap.parse_args()

    src = Path(a.source)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in EXT)
    if not files:
        raise SystemExit(f"Nenhuma imagem em {src}.")
    out = Path(a.out)
    conf = default_conf() if a.conf is None else a.conf
    from cv_service import model_imgsz
    a.imgsz = a.imgsz or model_imgsz(get_model(a.weights))
    rows = []
    for f in files:
        try:
            img = decode_image(str(f))
        except ValueError:
            print(f"[aviso] não foi possível ler {f}"); continue
        t0 = time.perf_counter()
        dets, timing = detect_image(img, conf=conf, imgsz=a.imgsz, weights=a.weights)
        total_ms = (time.perf_counter() - t0) * 1000
        ann = draw_detections(img, dets)
        d = out / f"{f.parent.name}__{f.stem}"
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "original.jpg"), img)
        cv2.imwrite(str(d / "deteccao.jpg"), ann)
        cv2.imwrite(str(d / "comparativo.jpg"), side_by_side(img, ann))
        n_pot = sum(x["class_en"] == "pothole" for x in dets)
        n_crk = sum(x["class_en"] == "crack" for x in dets)
        rec = {"arquivo": f.name, "pasta": f.parent.name, "largura": img.shape[1], "altura": img.shape[0],
               "conf_usada": conf, "imgsz": a.imgsz, "quantidade_defeitos": len(dets), "buracos": n_pot,
               "rachaduras": n_crk, "inference_ms": timing.get("inference"), "tempo_total_ms": round(total_ms, 1),
               "deteccoes": dets}
        (d / "resultado.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        rows.append({k: v for k, v in rec.items() if k != "deteccoes"} |
                    {"conf_maxima": max((x["confidence"] for x in dets), default=0.0)})
        print(f"{f.parent.name}/{f.name}: {len(dets)} defeito(s) [buraco={n_pot}, rachadura={n_crk}] "
              f"inferência={timing.get('inference')} ms")
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)

    if a.expected_from_folders:
        df = df[df.pasta.isin(EXPECT)].copy()
        def hit(r):
            exp = EXPECT[r.pasta]
            return (r.quantidade_defeitos == 0) if exp is None else (getattr(r, "buracos" if exp == "pothole" else "rachaduras") > 0)
        df["esperado"] = df.pasta.map(lambda p: EXPECT[p] or "sem dano")
        df["resultado_ok"] = df.apply(hit, axis=1)
        df.to_csv(out / "expected_vs_detected.csv", index=False)
        print("\n=== esperado (pasta) x detectado ===")
        print(df.groupby("pasta").resultado_ok.agg(["sum", "count"]).rename(columns={"sum": "acertos", "count": "imagens"}))
    cards = []
    for r in rows:
        cards.append(f'<div class="c"><img src="{r["pasta"]}__{Path(r["arquivo"]).stem}/comparativo.jpg">'
                     f'<p><b>{r["pasta"]}/{r["arquivo"]}</b><br>defeitos: {r["quantidade_defeitos"]} '
                     f'(buracos {r["buracos"]}, rachaduras {r["rachaduras"]}) | inferencia {r["inference_ms"]} ms</p></div>')
    (out / "galeria.html").write_text("<!doctype html><meta charset='utf-8'><title>Galeria de deteccoes</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#eee}.c{margin:14px 0}img{max-width:100%}</style>"
        f"<h2>Deteccoes em imagens - {AUTHOR}</h2>" + "".join(cards), encoding="utf-8")
    print(f"\n[ok] {len(rows)} imagens processadas | tempo médio de inferência: "
          f"{pd.DataFrame(rows).inference_ms.mean():.1f} ms | saída: {out}")


if __name__ == "__main__":
    main()
