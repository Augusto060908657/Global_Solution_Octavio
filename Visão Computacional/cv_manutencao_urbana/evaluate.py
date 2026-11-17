"""
Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696
GS FIAP 2026.2 - CitizenTech - Visão Computacional

evaluate.py - avaliação do detector no conjunto de TESTE (nunca usado em treino/seleção) + latência/FPS.

Etapas:
  1. Seleção do limiar de confiança no conjunto de VALIDAÇÃO (maximiza o F1 médio das classes, IoU>=0.5) ->
     weights/threshold.json e reports/f1_vs_conf.png. O teste NÃO participa dessa escolha.
  2. Métricas Ultralytics no TESTE: Precision, Recall, mAP@50, mAP@50-95 (P/R no ponto de máximo F1) e F1 = 2PR/(P+R).
  3. Métricas no limiar operacional (caixas IoU>=0.5): Precision/Recall/F1 por classe e matriz de confusão
     (reports/confusion_matrix.png) + presença por imagem (há buraco? há rachadura?).
  4. Galeria de acertos/erros do teste (reports/error_analysis/).
  5. Benchmark de latência (média ms) e FPS em CPU e/ou GPU (--bench_devices cpu,0): forward puro e ponta a ponta,
     10 warmup + 100 iterações, time.perf_counter(), com equipamento (nome da GPU/CPU) e resolução, e o efeito do limiar
     de confiança no tempo.
  5b. Fatores de erro: recall/FP por tercil de brilho, área escura (sombra), textura e tamanho do objeto.
  3b. No dataset combinado, métricas por fonte (Brasil, RDD2022...).
  6. (opcional) --external_pothole: buracos de fotos externas com rótulo (tests/external_labels).

Uso: python evaluate.py --yolo_dir data/yolo --imgsz 512 [--external_pothole]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image

from common import (AUTHOR, BEST_PT, CLASS_COLORS_BGR, CLASS_NAMES, CLASS_PT, EXTERNAL_DIR, REPORTS_DIR, ROOT, RUNS_DIR,
                    SEED, WEIGHTS_DIR, default_yolo_dir, hardware_info, write_dataset_yaml)

NC = len(CLASS_NAMES)
IMG_EXT = {".jpg", ".jpeg", ".png"}


# ----------------------------------------------------------------------------- GT / predições
def load_gt(label_path: Path, w: int, h: int) -> np.ndarray:
    rows = []
    if label_path.exists():
        for ln in label_path.read_text().splitlines():
            if ln.strip():
                c, cx, cy, bw, bh = ln.split()[:5]
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                rows.append([int(c), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return np.array(rows, dtype=float).reshape(-1, 5)


def collect(model, images_dir: Path, labels_dir: Path, imgsz: int, conf_min: float = 0.05, iou_nms: float = 0.6,
            device: str = "cpu", only_prefix: str = None, names: list = None):
    """Prediz todas as imagens uma vez (conf mínimo baixo); filtros por limiar são feitos depois em numpy."""
    recs = []
    for p in sorted(q for q in images_dir.iterdir() if q.suffix.lower() in IMG_EXT):
        if only_prefix and not p.name.startswith(only_prefix):
            continue
        w, h = Image.open(p).size
        r = model.predict(str(p), imgsz=imgsz, conf=conf_min, iou=iou_nms, device=device, verbose=False)[0]
        pred = np.concatenate([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None],
                               r.boxes.cls.cpu().numpy()[:, None]], axis=1) if len(r.boxes) else np.zeros((0, 6))
        recs.append({"path": p, "gt": load_gt(labels_dir / f"{p.stem}.txt", w, h), "pred": pred})
    return recs


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(gt, pr, iou_thr=0.5):
    """Casamento guloso por IoU (independente de classe). Retorna pares (i_gt, j_pred)."""
    if len(gt) == 0 or len(pr) == 0:
        return []
    iou = iou_matrix(gt[:, 1:5], pr[:, :4])
    cand = np.argwhere(iou >= iou_thr)
    cand = cand[np.argsort(-iou[cand[:, 0], cand[:, 1]])]
    ug, up, pairs = set(), set(), []
    for i, j in cand:
        if i not in ug and j not in up:
            ug.add(i); up.add(j); pairs.append((int(i), int(j)))
    return pairs


def score(recs, conf: float, iou_thr: float = 0.5):
    cm = np.zeros((NC + 1, NC + 1), int)           # linhas = real (última = sem objeto), colunas = previsto
    tp, fp, fn = np.zeros(NC), np.zeros(NC), np.zeros(NC)
    img = {c: np.zeros(4, int) for c in range(NC)}  # TP, FP, FN, TN por imagem (presença da classe)
    for r in recs:
        gt, pr = r["gt"], r["pred"][r["pred"][:, 4] >= conf]
        pairs = match(gt, pr, iou_thr)
        mg, mp = {i for i, _ in pairs}, {j for _, j in pairs}
        for i, j in pairs:
            gc, pc = int(gt[i, 0]), int(pr[j, 5])
            cm[gc, pc] += 1
            if gc == pc:
                tp[gc] += 1
            else:
                fn[gc] += 1; fp[pc] += 1
        for i in range(len(gt)):
            if i not in mg:
                cm[int(gt[i, 0]), NC] += 1; fn[int(gt[i, 0])] += 1
        for j in range(len(pr)):
            if j not in mp:
                cm[NC, int(pr[j, 5])] += 1; fp[int(pr[j, 5])] += 1
        for c in range(NC):
            g, p = (gt[:, 0] == c).any() if len(gt) else False, (pr[:, 5] == c).any() if len(pr) else False
            img[c][0 if g and p else 1 if p else 2 if g else 3] += 1
    prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return {"cm": cm, "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1, "img": img}


def select_conf(val_recs, grid=np.arange(0.05, 0.76, 0.025)):
    curves = {c: [] for c in range(NC)}; macro = []
    for t in grid:
        s = score(val_recs, t)
        for c in range(NC):
            curves[c].append(s["f1"][c])
        macro.append(float(np.mean(s["f1"])))
    best = float(grid[int(np.argmax(macro))])
    plt.figure(figsize=(7, 4.5))
    for c in range(NC):
        plt.plot(grid, curves[c], label=CLASS_PT[CLASS_NAMES[c]])
    plt.plot(grid, macro, "k--", label="F1 médio")
    plt.axvline(best, color="gray", ls=":", label=f"limiar escolhido = {best:.3f}")
    plt.xlabel("limiar de confiança"); plt.ylabel("F1 (IoU>=0.5) na VALIDAÇÃO"); plt.grid(alpha=.3); plt.legend()
    plt.title("Seleção do limiar de confiança (validação)")
    plt.tight_layout(); plt.savefig(REPORTS_DIR / "f1_vs_conf.png", dpi=150); plt.close()
    return best, {"grid": [round(float(g), 3) for g in grid], "macro_f1": [round(m, 4) for m in macro]}


# ----------------------------------------------------------------------------- gráficos / galeria
def plot_confusion(cm, path: Path, conf: float):
    labels_true = [CLASS_PT[n] for n in CLASS_NAMES] + ["Sem objeto (falso alarme)"]
    labels_pred = [CLASS_PT[n] for n in CLASS_NAMES] + ["Não detectado"]
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    annot = np.array([[f"{cm[i, j]}\n({pct[i, j]:.1f}%)" if i < NC else f"{cm[i, j]}" for j in range(NC + 1)]
                      for i in range(NC + 1)])
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", linewidths=1, linecolor="white",
                xticklabels=labels_pred, yticklabels=labels_true, annot_kws={"size": 12})
    plt.xlabel("Previsto"); plt.ylabel("Real")
    plt.title(f"Matriz de confusão - TESTE (IoU>=0.5, conf>={conf:.3f})")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def draw(img, boxes, colors, thick):
    for (x1, y1, x2, y2, txt), col in zip(boxes, colors):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, thick)
        if txt:
            cv2.putText(img, txt, (int(x1) + 2, max(14, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def save_gallery(recs, conf, out_dir: Path, n_bad=8, n_good=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()
    scored = []
    for r in recs:
        pr = r["pred"][r["pred"][:, 4] >= conf]
        pairs = match(r["gt"], pr)
        ok = sum(int(r["gt"][i, 0]) == int(pr[j, 5]) for i, j in pairs)
        errors = (len(r["gt"]) - ok) + (len(pr) - ok)
        scored.append((errors, ok, r, pr))
    bad = sorted(scored, key=lambda s: -s[0])[:n_bad]
    good = sorted([s for s in scored if s[0] == 0 and s[1] > 0], key=lambda s: -s[1])[:n_good]
    for tag, items in (("erro", bad), ("acerto", good)):
        for errors, ok, r, pr in items:
            img = cv2.imread(str(r["path"]))
            left, right = img.copy(), img.copy()
            draw(left, [(*g[1:5], CLASS_PT[CLASS_NAMES[int(g[0])]]) for g in r["gt"]], [(0, 200, 0)] * len(r["gt"]), 2)
            draw(right, [(*p[:4], f"{CLASS_PT[CLASS_NAMES[int(p[5])]]} {p[4]:.2f}") for p in pr],
                 [CLASS_COLORS_BGR[CLASS_NAMES[int(p[5])]] for p in pr], 2)
            for im, t in ((left, "REAL (verde)"), (right, "PREVISTO")):
                cv2.rectangle(im, (0, 0), (190, 26), (0, 0, 0), -1)
                cv2.putText(im, t, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / f"{tag}_{errors:02d}err_{r['path'].stem}.jpg"), np.hstack([left, right]))


# ----------------------------------------------------------------------------- benchmark
def _sync(device: str):
    if device != "cpu":
        torch.cuda.synchronize()


def benchmark(wpath: Path, img_path: Path, imgsz: int, device: str = "cpu", n_iter=100, warmup=10,
              conf_grid=(0.05, 0.10, 0.25, 0.50), conf_iters=30):
    """Latência/FPS por imagem em `device` ('cpu' ou '0' = GPU). Ponta a ponta (pré-proc + modelo + NMS) e forward puro,
    com warmup e time.perf_counter(). Também mede o efeito do limiar de confiança no tempo (NMS/pós-processamento)."""
    from ultralytics import YOLO
    model = YOLO(str(wpath))   # instância NOVA: o Ultralytics fixa o dispositivo no 1º predict, então cada dispositivo precisa da sua
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    pred = lambda c: model.predict(img, imgsz=imgsz, conf=c, device=device, verbose=False)[0]
    for _ in range(warmup):
        pred(0.25)
    _sync(device)
    used = str(next(model.predictor.model.model.parameters()).device) if hasattr(model.predictor.model, "model") else str(model.predictor.model.device)
    assert (device == "cpu") == (used == "cpu"), f"benchmark pediu '{device}' mas o modelo está em '{used}'"
    e2e, parts = [], []
    for _ in range(n_iter):
        _sync(device); t0 = time.perf_counter()
        r = pred(0.25)
        _sync(device); e2e.append((time.perf_counter() - t0) * 1000)
        parts.append(r.speed)
    net = model.model.eval().float().to("cpu" if device == "cpu" else "cuda")
    x = torch.zeros(1, 3, int(np.ceil(imgsz * h / max(h, w) / 32) * 32), int(np.ceil(imgsz * w / max(h, w) / 32) * 32))
    x = x.to("cpu" if device == "cpu" else "cuda")
    fw = []
    with torch.inference_mode():
        for _ in range(warmup):
            net(x)
        _sync(device)
        for _ in range(n_iter):
            _sync(device); t0 = time.perf_counter(); net(x); _sync(device); fw.append((time.perf_counter() - t0) * 1000)
    net.to("cpu")
    conf_rows = []
    for c in conf_grid:
        ts, nd = [], 0
        for _ in range(conf_iters):
            _sync(device); t0 = time.perf_counter(); r = pred(c); _sync(device)
            ts.append((time.perf_counter() - t0) * 1000); nd = len(r.boxes)
        conf_rows.append({"conf": c, "latencia_media_ms": float(np.mean(ts)), "fps": float(1000 / np.mean(ts)), "deteccoes": nd})
    e2e, fw = np.array(e2e), np.array(fw)
    hw = hardware_info()
    return {"dispositivo": "GPU: " + str(hw["gpu"]) if device != "cpu" else "CPU: " + str(hw["cpu"]), "dispositivo_verificado": used,
            "imagem_original": [w, h], "imgsz": imgsz, "tensor_forward": list(x.shape), "warmup": warmup, "iteracoes": n_iter,
            "hardware": hw, "torch_threads": torch.get_num_threads(),
            "ponta_a_ponta": {"latencia_media_ms": float(e2e.mean()), "p95_ms": float(np.percentile(e2e, 95)),
                              "desvio_ms": float(e2e.std()), "fps": float(1000 / e2e.mean()),
                              "preprocess_ms_medio": float(np.mean([p["preprocess"] for p in parts])),
                              "inference_ms_medio": float(np.mean([p["inference"] for p in parts])),
                              "postprocess_ms_medio": float(np.mean([p["postprocess"] for p in parts]))},
            "forward_puro": {"latencia_media_ms": float(fw.mean()), "p95_ms": float(np.percentile(fw, 95)),
                             "fps": float(1000 / fw.mean())},
            "limiar_x_latencia": conf_rows}


# ----------------------------------------------------------------------------- fatores de erro (sombra, iluminação, textura, tamanho)
def image_factors(path: Path) -> dict:
    g = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (320, int(320 * g.shape[0] / g.shape[1])))
    return {"brilho": float(g.mean()), "contraste": float(g.std()),
            "area_escura": float((g < 60).mean()),                      # proxy de sombra / subexposição
            "textura": float((cv2.Canny(g, 80, 160) > 0).mean())}       # proxy de textura/ruído do asfalto


def error_factors(recs, conf, n_bins=3):
    """Recall e falsos positivos por tercil de brilho, sombra, textura e tamanho do objeto (IoU>=0.5, mesma classe)."""
    rows = []
    for r in recs:
        pr = r["pred"][r["pred"][:, 4] >= conf]
        pairs = match(r["gt"], pr)
        ok = [(i, j) for i, j in pairs if int(r["gt"][i, 0]) == int(pr[j, 5])]
        f = image_factors(r["path"])
        h, w = cv2.imread(str(r["path"])).shape[:2]
        rows.append({**f, "gt": len(r["gt"]), "tp": len(ok), "fp": len(pr) - len(ok)})
    df = pd.DataFrame(rows)
    out = {}
    for fac in ("brilho", "area_escura", "textura", "contraste"):
        if df[fac].nunique() < n_bins:
            continue
        df["_bin"] = pd.qcut(df[fac], n_bins, labels=["baixo", "médio", "alto"], duplicates="drop")
        g = df.groupby("_bin", observed=True).agg(imagens=("gt", "size"), objetos=("gt", "sum"), acertos=("tp", "sum"),
                                                 fp=("fp", "sum"), faixa_min=(fac, "min"), faixa_max=(fac, "max"))
        g["recall"] = g.acertos / g.objetos.clip(lower=1)
        g["fp_por_imagem"] = g.fp / g.imagens
        out[fac] = {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in g.round(4).to_dict("index").items()}
    # por tamanho do objeto (área relativa)
    sizes = []
    for r in recs:
        w_h = cv2.imread(str(r["path"])).shape[:2]
        pr = r["pred"][r["pred"][:, 4] >= conf]
        matched = {i for i, j in match(r["gt"], pr) if int(r["gt"][i, 0]) == int(pr[j, 5])}
        for i, g in enumerate(r["gt"]):
            sizes.append({"area_rel": (g[3] - g[1]) * (g[4] - g[2]) / (w_h[0] * w_h[1]), "ok": i in matched})
    sd = pd.DataFrame(sizes)
    if len(sd) >= 9:
        sd["_bin"] = pd.qcut(sd.area_rel, 3, labels=["pequeno", "médio", "grande"], duplicates="drop")
        g = sd.groupby("_bin", observed=True).agg(objetos=("ok", "size"), recall=("ok", "mean"),
                                                 area_min=("area_rel", "min"), area_max=("area_rel", "max"))
        out["tamanho_objeto"] = {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in g.round(5).to_dict("index").items()}
    return out


# ----------------------------------------------------------------------------- Ultralytics val
def ultralytics_val(model, yaml_path: Path, split: str, imgsz: int, name: str, device: str = "cpu"):
    m = model.val(data=str(yaml_path), split=split, imgsz=imgsz, batch=8, conf=0.001, iou=0.6, device=device,
                  workers=1, plots=True, project=str(RUNS_DIR), name=name, exist_ok=True, verbose=False)
    b = m.box
    per = {}
    for k, ci in enumerate(b.ap_class_index):
        p, r, ap50, ap = b.class_result(k)
        per[CLASS_NAMES[int(ci)]] = {"precision": float(p), "recall": float(r), "f1": float(2 * p * r / max(p + r, 1e-9)),
                                     "mAP50": float(ap50), "mAP50_95": float(ap)}
    p, r = float(b.mp), float(b.mr)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "mAP50": float(b.map50),
            "mAP50_95": float(b.map), "por_classe": per}


def main():
    ap = argparse.ArgumentParser(description=f"Avaliação | {AUTHOR}")
    ap.add_argument("--yolo_dir", default=None, help="padrão: data/yolo_mix se existir, senão data/yolo")
    ap.add_argument("--weights", default=str(BEST_PT.relative_to(ROOT)))
    ap.add_argument("--imgsz", type=int, default=None, help="padrão: o usado no treino (lido do best.pt)")
    ap.add_argument("--device", default="cpu", help="dispositivo das predições de avaliação: cpu ou 0 (GPU)")
    ap.add_argument("--bench_devices", default="cpu", help="benchmark de latência/FPS: ex. 'cpu' ou 'cpu,0' (T4 no Colab)")
    ap.add_argument("--skip_benchmark", action="store_true")
    ap.add_argument("--external_pothole", action="store_true", help="avalia buracos em tests/external_*")
    a = ap.parse_args()

    from ultralytics import YOLO
    wpath = Path(a.weights) if Path(a.weights).is_absolute() else ROOT / a.weights
    model = YOLO(str(wpath))
    a.imgsz = a.imgsz or int(model.ckpt.get("train_args", {}).get("imgsz", 640))
    yolo_dir = Path(a.yolo_dir) if a.yolo_dir else default_yolo_dir()
    yaml_path = write_dataset_yaml(yolo_dir)
    REPORTS_DIR.mkdir(exist_ok=True)
    try:
        rel = str(yolo_dir.resolve().relative_to(ROOT))
    except ValueError:
        rel = str(yolo_dir)
    out = {"author": AUTHOR, "weights": wpath.name, "imgsz": a.imgsz, "dados": rel}
    print(f"[eval] dados: {yolo_dir} | imgsz {a.imgsz} | device {a.device}")

    # 1) limiar escolhido na validação
    val_recs = collect(model, yolo_dir / "images" / "val", yolo_dir / "labels" / "val", a.imgsz, device=a.device)
    conf, curve = select_conf(val_recs)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    (WEIGHTS_DIR / "threshold.json").write_text(json.dumps({"conf": conf, "criterio": "max F1 medio (IoU>=0.5) na validacao",
                                                            "imgsz": a.imgsz}, indent=2))
    out["limiar_confianca_escolhido_na_validacao"] = conf
    out["f1_vs_conf_validacao"] = curve
    print(f"[limiar] escolhido na validação: conf={conf:.3f}")

    # 2) mAP Ultralytics (teste completo e por fonte) e 3) limiar operacional
    out["teste_ultralytics_conf0.001"] = ultralytics_val(model, yaml_path, "test", a.imgsz, "eval_test", a.device)
    test_recs = collect(model, yolo_dir / "images" / "test", yolo_dir / "labels" / "test", a.imgsz, device=a.device)
    s = score(test_recs, conf)
    plot_confusion(s["cm"], REPORTS_DIR / "confusion_matrix.png", conf)
    out["teste_limiar_operacional"] = {
        "conf": conf, "iou_match": 0.5,
        "por_classe": {CLASS_NAMES[c]: {"TP": int(s["tp"][c]), "FP": int(s["fp"][c]), "FN": int(s["fn"][c]),
                                        "precision": float(s["precision"][c]), "recall": float(s["recall"][c]),
                                        "f1": float(s["f1"][c])} for c in range(NC)},
        "matriz_confusao_linhas_real_colunas_previsto": s["cm"].tolist(),
        "presenca_por_imagem": {CLASS_NAMES[c]: {"TP": int(v[0]), "FP": int(v[1]), "FN": int(v[2]), "TN": int(v[3]),
                                                 "precision": float(v[0] / max(v[0] + v[1], 1)),
                                                 "recall": float(v[0] / max(v[0] + v[2], 1))} for c, v in s["img"].items()}}

    # 3b) por fonte (Brasil, RDD2022, ...): só existe no dataset combinado
    por_fonte = {}
    for y in sorted(yolo_dir.glob("dataset_test_*.yaml")):
        src = y.stem.replace("dataset_test_", "")
        sub = [r for r in test_recs if r["path"].name.startswith(src + "_")]
        if not sub:
            continue
        u = ultralytics_val(model, y, "test", a.imgsz, f"eval_test_{src}", a.device)
        ss = score(sub, conf)
        por_fonte[src] = {"imagens": len(sub), "mAP50": u["mAP50"], "mAP50_95": u["mAP50_95"], "precision_ult": u["precision"],
                          "recall_ult": u["recall"], "f1_ult": u["f1"],
                          "limiar": {CLASS_NAMES[c]: {"precision": float(ss["precision"][c]), "recall": float(ss["recall"][c]),
                                                      "f1": float(ss["f1"][c])} for c in range(NC)},
                          "presenca_por_imagem": {CLASS_NAMES[c]: {"precision": float(v[0] / max(v[0] + v[1], 1)),
                                                                   "recall": float(v[0] / max(v[0] + v[2], 1))}
                                                  for c, v in ss["img"].items()}}
    if por_fonte:
        out["teste_por_fonte"] = por_fonte

    # 4) análise de erros: galeria + fatores (iluminação, sombra, textura, tamanho)
    save_gallery(test_recs, conf, REPORTS_DIR / "error_analysis")
    out["fatores_de_erro"] = error_factors(test_recs, conf)

    # 5) latência / FPS
    if not a.skip_benchmark:
        out["benchmark"] = {}
        for d in [x.strip() for x in a.bench_devices.split(",") if x.strip()]:
            if d != "cpu" and not torch.cuda.is_available():
                print(f"[aviso] GPU solicitada ({d}) indisponível; benchmark ignorado.")
                continue
            out["benchmark"]["cpu" if d == "cpu" else "gpu"] = benchmark(wpath, test_recs[0]["path"], a.imgsz, d)

    # 6) buracos em fotos externas
    if a.external_pothole:
        ext = EXTERNAL_DIR / "buracos"
        recs = collect(model, ext, EXTERNAL_DIR.parent / "external_labels" / "buracos", a.imgsz, device=a.device)
        for r in recs:  # só a classe buraco tem rótulo nestas imagens: ignora previsões de rachadura
            r["pred"] = r["pred"][r["pred"][:, 5] == 0]
        se = score(recs, conf)
        vistos = any(yolo_dir.glob("images/train/pothole_github_*"))
        out["externo_buracos"] = {"imagens": len(recs), "fonte": "Muhammad-aa/Pothole-Detector (test; estas imagens NÃO foram usadas no treino" +
                                  (", mas train/valid da MESMA fonte entraram no treino: mede desempenho nessa fonte, não domínio novo)" if vistos else ")"),
                                  "TP": int(se["tp"][0]), "FP": int(se["fp"][0]), "FN": int(se["fn"][0]),
                                  "precision": float(se["precision"][0]), "recall": float(se["recall"][0]),
                                  "f1": float(se["f1"][0]), "conf": conf, "iou_match": 0.5}

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    rows = [{"classe": k, **v} for k, v in out["teste_ultralytics_conf0.001"]["por_classe"].items()]
    rows.append({"classe": "todas", **{k: out["teste_ultralytics_conf0.001"][k] for k in
                                       ("precision", "recall", "f1", "mAP50", "mAP50_95")}})
    pd.DataFrame(rows).round(4).to_csv(REPORTS_DIR / "metrics_test.csv", index=False)
    print(json.dumps({k: v for k, v in out.items() if k not in ("f1_vs_conf_validacao", "fatores_de_erro")}, indent=2, ensure_ascii=False))
    print("[ok] reports/metrics.json, metrics_test.csv, confusion_matrix.png, f1_vs_conf.png, error_analysis/")


if __name__ == "__main__":
    main()
