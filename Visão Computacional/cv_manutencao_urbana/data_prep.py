"""
Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696
GS FIAP 2026.2 - CitizenTech - Visão Computacional

data_prep.py - Preparação de dados: máscaras -> anotações YOLO, split sem data leakage, inspeção de formatos.

Dataset (brasileiro): "Cracks and Potholes in Road Images" (Passos et al.; Kaggle, rodovias de RS, DF e ES).
Estrutura: uma pasta por imagem com <id>_RAW.jpg e máscaras binárias <id>_CRACK.png, <id>_POTHOLE.png, <id>_LANE.png.
As máscaras NÃO estão em formato YOLO: este script converte cada máscara em bounding boxes
(componentes conectados após dilatação morfológica, que une fragmentos próximos de uma mesma rachadura).
A máscara LANE (faixas de sinalização) não é dano e é ignorada.

Anti-leakage (2 camadas): o nome do arquivo é <id>_<UF>_<rodovia>_<trecho>_<quadro>. Quadros consecutivos do mesmo trecho são
quase idênticos (mesma via, ângulo e iluminação). Por isso o grupo é (trecho + bloco de --stretch_frames quadros
consecutivos) e CADA GRUPO FICA EM UM ÚNICO SPLIT (70/15/15, seed=42), verificado por assert. Além disso, quadros de TREINO a <= --buffer_frames de um quadro de val/test no mesmo
trecho (fronteiras entre blocos) são descartados (zona-tampão).

Uso:
  python data_prep.py --raw_dir data/raw --out_dir data/yolo [--audit] [--max_images N]
"""
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import AUTHOR, CLASS_NAMES, REPORTS_DIR, SEED, write_dataset_yaml

# parâmetros da conversão máscara -> caixa (px na resolução original ~1024x640)
BOX_PARAMS = {
    "crack":   {"dilate": 21, "min_area": 60,  "min_side": 20},
    "pothole": {"dilate": 9,  "min_area": 200, "min_side": 12},
}


# ----------------------------------------------------------------------------- descoberta e inspeção
def discover_samples(raw_dir: Path) -> pd.DataFrame:
    rows = []
    seen = set()
    for raw in sorted(Path(raw_dir).rglob("*_RAW.jpg")):
        if "preview" in str(raw.relative_to(raw_dir)).lower():  # pasta 'Preview Images' duplica uma amostra
            continue
        if raw.name in seen:
            continue
        seen.add(raw.name)
        stem = raw.name[:-len("_RAW.jpg")]
        rows.append({"stem": stem, "raw": raw,
                     "crack": raw.with_name(f"{stem}_CRACK.png"),
                     "pothole": raw.with_name(f"{stem}_POTHOLE.png")})
    if not rows:
        raise FileNotFoundError(f"Nenhum *_RAW.jpg em {raw_dir}. Descompacte o dataset (ver README).")
    df = pd.DataFrame(rows)
    missing = ~(df.crack.map(Path.exists) & df.pothole.map(Path.exists))
    if missing.any():
        print(f"[aviso] {int(missing.sum())} amostras sem máscara CRACK/POTHOLE foram descartadas.")
        df = df[~missing].reset_index(drop=True)
    return df


def inspect_formats(df: pd.DataFrame, n: int = 60) -> dict:
    """Verifica formatos de imagens e máscaras antes de usar (amostra determinística)."""
    rng = np.random.RandomState(SEED)
    sample = df.iloc[rng.permutation(len(df))[:n]]
    shapes, mask_shapes, mask_vals, dtypes = Counter(), Counter(), Counter(), Counter()
    for r in sample.itertuples():
        shapes[str(cv2.imread(str(r.raw)).shape)] += 1
        for k in ("crack", "pothole"):
            m = cv2.imread(str(getattr(r, k)), cv2.IMREAD_UNCHANGED)
            mask_shapes[str(m.shape)] += 1
            mask_vals[str(sorted(np.unique(m).tolist()))] += 1
            dtypes[str(m.dtype)] += 1
    return {"n_amostras_total": len(df), "n_inspecionadas": len(sample),
            "formatos_imagem_RAW_HxWxC": dict(shapes), "formatos_mascara": dict(mask_shapes),
            "valores_unicos_mascara": dict(mask_vals), "dtype_mascara": dict(dtypes),
            "conclusao": "RAW = JPEG BGR; máscaras = PNG 3 canais binárias (0/255) do mesmo tamanho da RAW "
                         "-> convertidas para caixas YOLO normalizadas (class cx cy w h)."}


# ----------------------------------------------------------------------------- máscara -> caixas
def mask_to_boxes(mask: np.ndarray, dilate: int, min_area: int, min_side: int):
    """Retorna lista de (x1, y1, x2, y2) em pixels, uma por região conectada da máscara."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return []
    d = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        sub = (m[y:y + h, x:x + w] > 0) & (lab[y:y + h, x:x + w] == i)  # só pixels originais da máscara
        if sub.sum() < min_area:
            continue
        ys, xs = np.nonzero(sub)
        x1, x2, y1, y2 = x + xs.min(), x + xs.max() + 1, y + ys.min(), y + ys.max() + 1
        if max(x2 - x1, y2 - y1) < min_side:
            continue
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def sample_to_yolo(row, img_shape):
    h, w = img_shape[:2]
    lines, counts = [], Counter()
    for cls_id, name in enumerate(CLASS_NAMES):
        mask = cv2.imread(str(getattr(row, name)), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape[:2] != (h, w):
            continue
        for x1, y1, x2, y2 in mask_to_boxes(mask, **BOX_PARAMS[name]):
            lines.append(f"{cls_id} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            counts[name] += 1
    return lines, counts


# ----------------------------------------------------------------------------- grupos e split
def group_of(stem: str, stretch_frames: int) -> str:
    parts = stem.split("_")
    seg = parts[3] if len(parts) >= 5 else "unk"
    m = re.search(r"(\d+)$", parts[-1])
    frame = int(m.group(1)) if m else 0
    return f"{seg}:{frame // stretch_frames}"


def parse_seg_frame(stem: str):
    parts = stem.split("_")
    seg = parts[3] if len(parts) >= 5 else "unk"
    m = re.search(r"(\d+)$", parts[-1])
    return seg, int(m.group(1)) if m else 0


def apply_buffer(df: pd.DataFrame, buffer_frames: int) -> pd.DataFrame:
    """Zona-tampão: remove do TREINO quadros do mesmo trecho a <= buffer_frames de qualquer quadro de val/test.
    Evita que quadros vizinhos nas fronteiras entre blocos (quase idênticos) vazem entre splits."""
    df = df.copy()
    if buffer_frames <= 0:
        return df
    held = df[df.split.isin(["val", "test"])]
    frames = {seg: np.sort(g.frame.values) for seg, g in held.groupby("seg")}
    def near(r):
        f = frames.get(r.seg)
        if f is None or r.split != "train":
            return False
        i = np.searchsorted(f, r.frame)
        return any(abs(int(f[j]) - r.frame) <= buffer_frames for j in (i - 1, i) if 0 <= j < len(f))
    mask = df.apply(near, axis=1)
    df.loc[mask, "split"] = "buffer"
    return df


def split_by_group(df: pd.DataFrame, seed: int = SEED, ratios=(0.70, 0.15, 0.15), trials: int = 500) -> pd.DataFrame:
    """Atribui GRUPOS (nunca imagens soltas) a train/val/test. Entre `trials` sorteios com seed fixa, escolhe o que
    melhor aproxima 70/15/15 em imagens e mantém proporções parecidas de imagens com buraco/rachadura."""
    rng = np.random.RandomState(seed)
    g = df.groupby("group").agg(n=("stem", "size"), pot=("n_pothole", lambda s: int((s > 0).sum())),
                                crk=("n_crack", lambda s: int((s > 0).sum())))
    names, sizes = list(g.index), g.n.to_dict()
    splits = ["train", "val", "test"]
    tot = g[["n", "pot", "crk"]].sum()
    best, best_cost = None, 1e18
    for _ in range(trials):
        order = list(rng.permutation(names))
        got = {s: np.zeros(3) for s in splits}
        assign = {}
        for name in order:
            deficit = {s: ratios[i] - got[s][0] / tot.n for i, s in enumerate(splits)}
            s = max(deficit, key=deficit.get)
            assign[name] = s
            got[s] += g.loc[name, ["n", "pot", "crk"]].values.astype(float)
        cost = sum(abs(got[s][0] / tot.n - ratios[i]) * 4 + abs(got[s][1] / tot.pot - ratios[i]) +
                   abs(got[s][2] / tot.crk - ratios[i]) for i, s in enumerate(splits))
        if cost < best_cost:
            best, best_cost = assign, cost
    df = df.copy()
    df["split"] = df.group.map(best)
    gs = {s: set(df[df.split == s].group) for s in splits}
    assert not (gs["train"] & gs["val"]) and not (gs["train"] & gs["test"]) and not (gs["val"] & gs["test"]), \
        "DATA LEAKAGE: grupo presente em mais de um split!"
    return df


# ----------------------------------------------------------------------------- auditoria (dHash)
def _dhash(path: Path) -> int:
    g = cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (9, 8), interpolation=cv2.INTER_AREA)
    return int(np.packbits((g[:, 1:] > g[:, :-1]).flatten()).view(">u8")[0])


def audit_near_duplicates(df: pd.DataFrame, yolo_dir: Path, max_n: int = 800, max_dist: int = 3) -> dict:
    """Quase-duplicatas (dHash) entre train e val/test: evidência empírica extra contra leakage."""
    rng = np.random.RandomState(SEED)

    def hashes(split):
        sub = df[df.split == split]
        sub = sub.iloc[rng.permutation(len(sub))[:max_n]]
        return np.array([_dhash(yolo_dir / "images" / split / f"{s}.jpg") for s in sub.stem], dtype=np.uint64)

    tr, out = hashes("train"), {}
    for split in ("val", "test"):
        h = hashes(split)
        x = np.bitwise_xor(h[:, None], tr[None, :])
        dist = np.unpackbits(x.view(np.uint8), axis=-1).reshape(*x.shape, 64).sum(-1)
        out[split] = {"verificadas": int(len(h)), "quase_duplicatas_vs_train(dist<=%d)" % max_dist:
                      int((dist.min(axis=1) <= max_dist).sum())}
    return out


# ----------------------------------------------------------------------------- overlay p/ conferir conversão
def save_label_overlays(df, yolo_dir: Path, out_dir: Path, n: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {0: (0, 0, 230), 1: (0, 165, 255)}
    sub = df[df.split == "train"].iloc[np.random.RandomState(SEED).permutation((df.split == "train").sum())[:n]]
    for s in sub.stem:
        img = cv2.imread(str(yolo_dir / "images" / "train" / f"{s}.jpg"))
        h, w = img.shape[:2]
        for ln in (yolo_dir / "labels" / "train" / f"{s}.txt").read_text().splitlines():
            c, cx, cy, bw, bh = ln.split()
            c, cx, cy, bw, bh = int(c), float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            cv2.rectangle(img, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), colors[c], 2)
        cv2.imwrite(str(out_dir / f"{s}.jpg"), img)


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=f"Preparação de dados | {AUTHOR}")
    ap.add_argument("--raw_dir", default="data/raw", help="pasta com as amostras do dataset (busca recursiva)")
    ap.add_argument("--out_dir", default="data/yolo")
    ap.add_argument("--stretch_frames", type=int, default=200, help="quadros consecutivos por grupo")
    ap.add_argument("--buffer_frames", type=int, default=30,
                    help="zona-tampão (quadros) removida do treino ao redor de val/test no mesmo trecho")
    ap.add_argument("--max_images", type=int, default=0, help="0 = usa todas (útil p/ testes rápidos)")
    ap.add_argument("--audit", action="store_true", help="auditoria de quase-duplicatas (dHash)")
    a = ap.parse_args()

    raw_dir, out_dir = Path(a.raw_dir), Path(a.out_dir)
    df = discover_samples(raw_dir)
    if a.max_images:
        df = df.iloc[np.random.RandomState(SEED).permutation(len(df))[:a.max_images]].reset_index(drop=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    insp = inspect_formats(df)
    (REPORTS_DIR / "dataset_inspection.json").write_text(json.dumps(insp, indent=2, ensure_ascii=False))
    print("[inspeção]", json.dumps(insp, indent=2, ensure_ascii=False))

    # 1) anotações YOLO (em memória) e contagens por imagem
    labels, n_crack, n_pot = {}, [], []
    for r in df.itertuples():
        img = cv2.imread(str(r.raw))
        lines, cnt = sample_to_yolo(r, img.shape)
        labels[r.stem] = lines
        n_crack.append(cnt["crack"]); n_pot.append(cnt["pothole"])
    df["n_crack"], df["n_pothole"] = n_crack, n_pot
    df["group"] = df.stem.map(lambda s: group_of(s, a.stretch_frames))
    df["seg"], df["frame"] = zip(*df.stem.map(parse_seg_frame))

    # 2) split por grupo + zona-tampão + escrita no formato YOLO
    df = split_by_group(df)
    df = apply_buffer(df, a.buffer_frames)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)
    for r in df[df.split != "buffer"].itertuples():
        shutil.copy2(r.raw, out_dir / "images" / r.split / f"{r.stem}.jpg")
        (out_dir / "labels" / r.split / f"{r.stem}.txt").write_text("\n".join(labels[r.stem]))
    df.drop(columns=["raw", "crack", "pothole"]).to_csv(out_dir / "splits.csv", index=False)
    yaml_path = write_dataset_yaml(out_dir)

    report = {"author": AUTHOR, "seed": SEED, "imagens": len(df), "grupos": int(df.group.nunique()),
              "stretch_frames": a.stretch_frames, "buffer_frames": a.buffer_frames,
              "imagens_descartadas_zona_tampao": int((df.split == "buffer").sum()), "box_params": BOX_PARAMS, "grupos_em_mais_de_um_split": 0}
    report["splits"] = {s: {"imagens": int((df.split == s).sum()), "grupos": int(df[df.split == s].group.nunique()),
                            "caixas_buraco": int(df[df.split == s].n_pothole.sum()),
                            "caixas_rachadura": int(df[df.split == s].n_crack.sum()),
                            "imgs_com_buraco": int((df[df.split == s].n_pothole > 0).sum()),
                            "imgs_com_rachadura": int((df[df.split == s].n_crack > 0).sum()),
                            "imgs_sem_dano": int(((df.split == s) & (df.n_pothole + df.n_crack == 0)).sum())}
                       for s in ("train", "val", "test")}
    if a.audit:
        report["auditoria_quase_duplicatas"] = audit_near_duplicates(df, out_dir)
    save_label_overlays(df, out_dir, REPORTS_DIR / "label_check")
    (REPORTS_DIR / "split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[ok] dataset YOLO em {out_dir} | yaml: {yaml_path}")


if __name__ == "__main__":
    main()
