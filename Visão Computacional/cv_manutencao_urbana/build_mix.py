"""
Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696
GS FIAP 2026.2 - CitizenTech - Visão Computacional

build_mix.py - combina o dataset brasileiro (já preparado por data_prep.py, com seu split anti-leakage) com as fontes
extras de data/extra/ em UM dataset YOLO de 2 classes (0 = buraco, 1 = rachadura): data/yolo_mix/.

 * Brasil: mantém o split de data_prep.py (grupos de trechos + zona-tampão).
 * Fontes com split "group" (RDD2022): split 70/15/15 por GRUPO (país + bloco de números consecutivos), seed 42.
 * Fontes com split "given" (Roboflow/GitHub): usam o split original (train -> train, valid -> val).
 * Arquivos recebem prefixo da fonte (br_, rdd2022_, pothole_github_...) para permitir avaliar CADA fonte separadamente
   (test_<fonte>.txt + yaml) e checar que nenhum grupo aparece em mais de um split.
Uso: python build_mix.py [--brazil_dir data/yolo] [--extra_dir data/extra] [--out_dir data/yolo_mix]
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from common import AUTHOR, CLASS_NAMES, EXTRA_DIR, MIX_DIR, REPORTS_DIR, YOLO_DIR, write_dataset_yaml
from data_prep import split_by_group


def read_labels(p: Path):
    return [ln.split() for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def to_bbox(r):
    """Linha YOLO -> (cx, cy, w, h). Aceita caixa (5 colunas) ou polígono de segmentação (classe + pares x y)."""
    v = [float(x) for x in r[1:]]
    if len(v) == 4:
        return v
    xs, ys = v[0::2], v[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def remap(rows, idmap):
    out = []
    for r in rows:
        new = idmap.get(int(float(r[0])))
        if new is not None and len(r) >= 5:
            cx, cy, w, h = to_bbox(r)
            if w > 0 and h > 0:
                out.append(f"{new} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def write_item(out: Path, split: str, name: str, img_src: Path, lines):
    shutil.copy2(img_src, out / "images" / split / f"{name}.jpg")
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=f"Dataset combinado | {AUTHOR}")
    ap.add_argument("--brazil_dir", default=str(YOLO_DIR.relative_to(YOLO_DIR.parents[1])))
    ap.add_argument("--extra_dir", default=str(EXTRA_DIR.relative_to(EXTRA_DIR.parents[1])))
    ap.add_argument("--out_dir", default=str(MIX_DIR.relative_to(MIX_DIR.parents[1])))
    ap.add_argument("--max_train_per_source", type=int, default=2500,
                    help="teto de imagens de TREINO por fonte extra (evita que uma fonte domine); 0 = sem teto")
    a = ap.parse_args()
    brazil, extra, out = Path(a.brazil_dir), Path(a.extra_dir), Path(a.out_dir)

    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val", "test"):
        (out / "images" / s).mkdir(parents=True); (out / "labels" / s).mkdir(parents=True)
    records = []  # (fonte, split, grupo, nome)

    # 1) Brasil
    df = pd.read_csv(brazil / "splits.csv")
    df = df[df.split.isin(["train", "val", "test"])]
    for r in df.itertuples():
        name = f"br_{r.stem}"
        write_item(out, r.split, name, brazil / "images" / r.split / f"{r.stem}.jpg",
                   (brazil / "labels" / r.split / f"{r.stem}.txt").read_text().splitlines())
        records.append(("br", r.split, f"br:{r.group}", name))

    # 2) extras
    fontes = [d for d in sorted(extra.glob("*")) if d.is_dir() and not d.name.startswith("_") and (d / "source.json").exists()]
    for d in fontes:
        meta = json.loads((d / "source.json").read_text())
        idmap = {i: (CLASS_NAMES.index(meta["map"][n]) if meta["map"].get(n) else None) for i, n in enumerate(meta["names"])}
        idx = pd.read_csv(d / "index.csv", keep_default_na=False)
        idx["lines"] = idx.stem.map(lambda s: remap(read_labels(d / "labels" / f"{s}.txt"), idmap))
        idx["n_pothole"] = idx.lines.map(lambda L: sum(x.startswith("0 ") for x in L))
        idx["n_crack"] = idx.lines.map(lambda L: sum(x.startswith("1 ") for x in L))
        if meta["split_mode"] == "group":
            idx = split_by_group(idx.assign(group=idx.group.astype(str)))
        else:
            idx["split"] = idx.split.replace({"valid": "val"})
            idx = idx[idx.split.isin(["train", "val", "test"])]
        if a.max_train_per_source:
            tr = idx[idx.split == "train"]
            if len(tr) > a.max_train_per_source:
                drop = tr.sample(len(tr) - a.max_train_per_source, random_state=42).index
                idx = idx.drop(drop)
                print(f"[mix] {d.name}: treino limitado a {a.max_train_per_source} imagens (aleatório, seed 42)")
        for r in idx.itertuples():
            name = f"{d.name}_{r.stem}"
            write_item(out, r.split, name, d / "images" / f"{r.stem}.jpg", r.lines)
            records.append((d.name, r.split, f"{d.name}:{r.group}", name))
        print(f"[mix] {d.name}: {len(idx)} imagens | classes: {meta['map']}")

    rec = pd.DataFrame(records, columns=["fonte", "split", "group", "name"])
    gs = {s: set(rec[rec.split == s].group) for s in ("train", "val", "test")}
    assert not (gs["train"] & gs["val"]) and not (gs["train"] & gs["test"]) and not (gs["val"] & gs["test"]), \
        "DATA LEAKAGE: grupo em mais de um split!"
    rec.to_csv(out / "splits.csv", index=False)

    # 3) listas de teste por fonte + yamls
    for f in rec.fonte.unique():
        lines = [str((out / "images" / "test" / f"{n}.jpg").resolve()) for n in rec[(rec.fonte == f) & (rec.split == "test")].name]
        if lines:
            (out / f"test_{f}.txt").write_text("\n".join(lines))
            write_dataset_yaml(out, test_list=f"test_{f}.txt", name=f"dataset_test_{f}.yaml")
    write_dataset_yaml(out)

    report = {"author": AUTHOR, "grupos_em_mais_de_um_split": 0, "fontes": {}}
    for f in rec.fonte.unique():
        report["fontes"][f] = {}
        for s in ("train", "val", "test"):
            names = rec[(rec.fonte == f) & (rec.split == s)].name
            cnt = Counter()
            bg = 0
            for n in names:
                L = [x for x in (out / "labels" / s / f"{n}.txt").read_text().splitlines() if x.strip()]
                bg += not L
                cnt.update(x.split()[0] for x in L)
            report["fontes"][f][s] = {"imagens": int(len(names)), "caixas_buraco": cnt["0"], "caixas_rachadura": cnt["1"],
                                      "imagens_sem_dano": bg}
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "mix_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[ok] dataset combinado em {out}")


if __name__ == "__main__":
    main()
