"""
Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696
GS FIAP 2026.2 - CitizenTech - Visão Computacional

data_extra.py - baixa FONTES EXTRAS (opcionais) para complementar o dataset brasileiro e melhorar a generalização.
Cada fonte vai para data/extra/<nome>/ em formato normalizado:
    images/*.jpg  labels/*.txt (YOLO, ids ORIGINAIS)  source.json (nomes/mapeamento/licença)  index.csv (stem,group,split)
O mapeamento para as classes do projeto (0 = buraco, 1 = rachadura) é feito depois por build_mix.py.

Fontes:
  rdd2022   RDD2022 (Arya et al., 2022) - base SUGERIDA no enunciado - via Hugging Face `dronefreak/RDD2022`
            (reexportação YOLO, CC BY-SA 4.0). Baixa apenas um SUBCONJUNTO por GRUPOS de quadros consecutivos
            (por país), sem o subconjunto de drone (vista aérea). Classes D00/D10/D20 -> rachadura; D40 -> buraco.
  pothole_github   Buracos (origem Roboflow) do repositório GitHub Muhammad-aa/Pothole-Detector: usa SÓ train e valid;
            o `test` desse repositório fica reservado como teste EXTERNO (tests/external_images/buracos).
  roboflow  Qualquer projeto do Roboflow Universe (ou SEU projeto anotado) via API: precisa de chave gratuita
            (use variável de ambiente ROBOFLOW_API_KEY; nunca escreva a chave no notebook).

Uso:
  python data_extra.py rdd2022 --groups_per_country 8
  python data_extra.py pothole_github
  python data_extra.py roboflow --workspace W --project P --version N --alias rf_rua   (chave em ROBOFLOW_API_KEY)
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # evita centenas de linhas de log no Colab
from common import AUTHOR, EXTRA_DIR, SEED

RDD_REPO = "dronefreak/RDD2022"
RDD_NAMES = ["D00", "D10", "D20", "D40"]
RDD_MAP = {"D00": "crack", "D10": "crack", "D20": "crack", "D40": "pothole"}


def write_source(out: Path, names, cmap, origin, license_, split_mode, rows):
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.json").write_text(json.dumps({"names": names, "map": cmap, "origin": origin, "license": license_,
                                                 "split_mode": split_mode}, indent=2, ensure_ascii=False))
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stem", "group", "split"]); w.writerows(rows)


# ----------------------------------------------------------------------------- RDD2022 (Hugging Face)
def plan_rdd(files, countries=None, exclude=("China_Drone",), groups_per_country=8, block=50, min_imgs=8, seed=SEED):
    """Escolhe GRUPOS (país + bloco de `block` números consecutivos) e devolve [(img_path, label_path, group)].
    Os blocos vêm de QUALQUER split original do repositório: o split final é refeito por grupo em build_mix.py."""
    fileset = set(files)
    groups = defaultdict(list)
    for f in files:
        m = re.match(r"data/images/(train|valid|test)/[^/]+/(.+)_(\d+)\.jpg$", f)
        if not m:
            continue
        country, num = m.group(2), int(m.group(3))
        if country in exclude or (countries and country not in countries):
            continue
        lab = f.replace("data/images/", "data/labels/").rsplit(".", 1)[0] + ".txt"
        if lab in fileset:
            groups[(country, num // block)].append((f, lab))
    by_country = defaultdict(list)
    for (c, b), items in groups.items():
        if len(items) >= min_imgs:
            by_country[c].append((c, b))
    rng = np.random.RandomState(seed)
    plan = []
    for c in sorted(by_country):
        keys = sorted(by_country[c])
        pick = [keys[i] for i in rng.permutation(len(keys))[:groups_per_country]]
        for k in pick:
            plan += [(i, l, f"{k[0]}:{k[1]}") for i, l in groups[k]]
    return plan


def _dl(repo, filename, local_dir, retries=3):
    from huggingface_hub import hf_hub_download
    for t in range(retries):
        try:
            return hf_hub_download(repo, filename, repo_type="dataset", local_dir=local_dir)
        except Exception as e:  # rede instável
            if t == retries - 1:
                raise e


def ingest_rdd(plan, local_dir: Path, out: Path):
    (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
    rows = []
    for img, lab, group in plan:
        stem = Path(img).stem
        shutil.copy2(local_dir / img, out / "images" / f"{stem}.jpg")
        shutil.copy2(local_dir / lab, out / "labels" / f"{stem}.txt")
        rows.append([stem, group, ""])
    write_source(out, RDD_NAMES, RDD_MAP, f"huggingface.co/datasets/{RDD_REPO} (RDD2022, Arya et al., arXiv:2209.08538)",
                 "CC BY-SA 4.0", "group", rows)


def cmd_rdd2022(a):
    from huggingface_hub import HfApi
    out = EXTRA_DIR / "rdd2022"
    print("[rdd2022] listando arquivos do repositório (pode levar alguns segundos)...")
    files = HfApi().list_repo_files(RDD_REPO, repo_type="dataset")
    plan = plan_rdd(files, groups_per_country=a.groups_per_country, block=a.block,
                    exclude=() if a.include_drone else ("China_Drone",))
    if not plan:
        raise SystemExit("[erro] nenhuma imagem planejada; o layout do repositório pode ter mudado.")
    tmp = EXTRA_DIR / "_hf_tmp"
    want = [x for p in plan for x in p[:2]]
    print(f"[rdd2022] baixando {len(plan)} imagens + rótulos ({len(want)} arquivos) em {len(set(p[2] for p in plan))} grupos...")
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda f: _dl(RDD_REPO, f, str(tmp)), want))
    ingest_rdd(plan, tmp, out)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[ok] {len(plan)} imagens em {out}")


# ----------------------------------------------------------------------------- Pothole-Detector (GitHub / Roboflow)
def ingest_pothole_github(repo_dir: Path, out: Path):
    (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
    rows = []
    for sp, tag in (("train", "train"), ("valid", "val")):  # o 'test' NÃO é usado: é o teste externo
        for img in sorted((repo_dir / "dataset" / "images" / sp).glob("*")):
            lab = repo_dir / "dataset" / "labels" / sp / f"{img.stem}.txt"
            if not lab.exists():
                continue
            base = img.stem.split("_jpg")[0]          # versões aumentadas da mesma foto compartilham o prefixo
            stem = f"{sp}_{img.stem}"[:90]
            shutil.copy2(img, out / "images" / f"{stem}.jpg")
            shutil.copy2(lab, out / "labels" / f"{stem}.txt")
            rows.append([stem, f"{sp}:{base}", tag])
    write_source(out, ["pothole"], {"pothole": "pothole"},
                 "github.com/Muhammad-aa/Pothole-Detector (procedência original não declarada no repositório; "
                 "tutorial associado usa dataset do Roboflow - verificar)", "verificar na fonte", "given", rows)


def cmd_pothole_github(a):
    clone = EXTRA_DIR / "_pothole_clone"
    shutil.rmtree(clone, ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/Muhammad-aa/Pothole-Detector", str(clone)],
                   check=True)
    ingest_pothole_github(clone, EXTRA_DIR / "pothole_github")
    shutil.rmtree(clone, ignore_errors=True)
    print("[ok] pothole_github pronto")


# ----------------------------------------------------------------------------- Roboflow (API)
POTHOLE_WORDS = ("pothole", "buraco", "hole", "cratera", "d40")
CRACK_WORDS = ("crack", "rachadura", "trinca", "fissura", "d00", "d10", "d20")


def guess_map(names):
    cmap = {}
    for n in names:
        low = str(n).lower()
        cmap[n] = "pothole" if any(w in low for w in POTHOLE_WORDS) else "crack" if any(w in low for w in CRACK_WORDS) else None
    return cmap


def cmd_roboflow(a):
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("[erro] pip install roboflow")
    key = a.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("[erro] informe --api_key ou a variável de ambiente ROBOFLOW_API_KEY (chave gratuita em roboflow.com). "
                         "Nunca grave a chave dentro do notebook/código.")
    alias = a.alias or re.sub(r"\W+", "_", a.project)[:24]
    tmp = EXTRA_DIR / f"_rf_{alias}"
    shutil.rmtree(tmp, ignore_errors=True)
    Roboflow(api_key=key).workspace(a.workspace).project(a.project).version(a.version).download("yolov8", location=str(tmp))
    import yaml
    names = yaml.safe_load(open(tmp / "data.yaml"))["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    cmap = guess_map(names)
    for kv in (a.map or "").split(","):                       # --map "classe=pothole,outra=crack,lixo=none"
        if "=" in kv:
            k, v = kv.split("=", 1)
            cmap[k.strip()] = None if v.strip() in ("", "none") else v.strip()
    print("[roboflow] classes do projeto e mapeamento:", cmap, "(None = descartada)")
    if not any(cmap.values()):
        raise SystemExit(f"[erro] nenhuma classe reconhecida em {names}. Use --map \"nome=pothole\" ou \"nome=crack\".")
    out = EXTRA_DIR / alias
    shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
    rows = []
    for sp, tag in (("train", "train"), ("valid", "val"), ("test", "test")):
        imgdir = tmp / sp / "images"
        for img in sorted(imgdir.glob("*")) if imgdir.exists() else []:
            lab = tmp / sp / "labels" / f"{img.stem}.txt"
            stem = f"{sp}_{img.stem}"[:90]
            shutil.copy2(img, out / "images" / f"{stem}.jpg")
            if lab.exists():
                shutil.copy2(lab, out / "labels" / f"{stem}.txt")
            else:
                (out / "labels" / f"{stem}.txt").write_text("")
            rows.append([stem, f"{alias}:{sp}:{img.stem.split('.rf.')[0].rsplit('_', 1)[0]}", tag])
    write_source(out, names, cmap, f"roboflow {a.workspace}/{a.project} v{a.version}", "ver página do projeto no Roboflow", "given", rows)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[ok] {len(rows)} imagens em {out}")


def main():
    ap = argparse.ArgumentParser(description=f"Fontes extras | {AUTHOR}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rdd2022"); r.add_argument("--groups_per_country", type=int, default=8)
    r.add_argument("--block", type=int, default=50, help="números consecutivos por grupo")
    r.add_argument("--include_drone", action="store_true")
    sub.add_parser("pothole_github")
    f = sub.add_parser("roboflow")
    for k in ("workspace", "project"):
        f.add_argument(f"--{k}", required=True)
    f.add_argument("--version", type=int, required=True); f.add_argument("--api_key", default=None)
    f.add_argument("--alias", default=None, help="nome curto da fonte (padrão: derivado do projeto)")
    f.add_argument("--map", default=None, help='mapeamento manual de classes: "nome=pothole,outro=crack,lixo=none"')
    a = ap.parse_args()
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    {"rdd2022": cmd_rdd2022, "pothole_github": cmd_pothole_github, "roboflow": cmd_roboflow}[a.cmd](a)


if __name__ == "__main__":
    main()
