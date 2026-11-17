"""
Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696
GS FIAP 2026.2 - CitizenTech - Visão Computacional

run.py - ponto de entrada ÚNICO. Faz tudo na ordem certa, com caminhos relativos.

  python run.py check                 # verifica dependências, dataset e imagens externas
  python run.py all                   # prep -> extra -> mix -> train -> eval -> demo -> sequence -> report
  python run.py all --quick           # versão rápida (416 px, 10 épocas, 40% do treino) para testar
  python run.py all --device 0        # com GPU (ex.: T4 do Colab): 40 épocas e benchmark de latência em GPU e CPU
  python run.py all --no-extra        # só o dataset brasileiro (sem RDD2022/Roboflow)
  python run.py all --compare         # também treina YOLO11s e gera a comparação de arquiteturas
  python run.py all --smoke --no-extra   # TESTE DE FUMAÇA (~minutos): roda TODAS as etapas com treino minúsculo p/ achar erros
  python run.py clean                 # apaga saídas geradas (pesos, relatórios, results) - use após o smoke
  python run.py prep|extra|mix|train|eval|gallery|demo|sequence|report|app
  --rf "workspace/projeto/versao"     # inclui SEU projeto Roboflow (chave em ROBOFLOW_API_KEY)

Fontes extras (data_extra.py): RDD2022 (Hugging Face) e buracos Roboflow (GitHub). Se o download falhar (sem internet),
o pipeline avisa e segue só com o dataset brasileiro.
O dataset brasileiro (.zip) pode estar em dataset/ ou na raiz; é descompactado automaticamente em data/raw/.
"""
import argparse
import importlib
import os
import shutil
import subprocess
import sys
import webbrowser
import zipfile
from pathlib import Path

from common import VERSION

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def sh(*args, fatal=True):
    print("\n$", "python", *map(str, args), flush=True)
    r = subprocess.run([PY, *map(str, args)], cwd=ROOT)
    if r.returncode and fatal:
        sys.exit(f"[erro] etapa falhou: {args[0]}")
    return r.returncode


def check_deps(need_gradio=False):
    mods = ["ultralytics", "cv2", "pandas", "matplotlib", "seaborn", "PIL", "numpy"] + (["gradio"] if need_gradio else [])
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit(f"[erro] faltam pacotes: {missing}. Rode: pip install -r requirements.txt")


def extract_recursive(zip_path: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    for inner in list(dest.rglob("*.zip")):
        extract_recursive(inner, inner.with_suffix(""))
        inner.unlink()


def ensure_raw(raw: Path = ROOT / "data" / "raw", search=(ROOT / "dataset", ROOT)):
    if any(raw.rglob("*_RAW.jpg")):
        return raw
    for d in search:
        for z in sorted(Path(d).glob("*.zip")):
            print(f"[dataset] descompactando {z.name} em {raw} (pode demorar)...")
            extract_recursive(z, raw)
            if any(raw.rglob("*_RAW.jpg")):
                return raw
    sys.exit("[erro] dataset não encontrado. Coloque o .zip em dataset/ (ver dataset/LEIA-ME.txt).")


def main():
    ap = argparse.ArgumentParser(description="CitizenTech CV - pipeline YOLO")
    ap.add_argument("step", choices=["check", "prep", "extra", "mix", "train", "eval", "gallery", "demo", "sequence", "report", "app", "clean", "all"])
    ap.add_argument("--quick", action="store_true", help="416 px, 10 épocas, 40%% do treino")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--device", default="cpu", help="cpu (padrão) ou 0 = GPU (ex.: Colab T4)")
    ap.add_argument("--model", default="yolo11n", help="yolo11n (padrão) ou yolo11s ...")
    ap.add_argument("--preset", default="robust", choices=["base", "robust"], help="augmentation do treino")
    ap.add_argument("--no-extra", dest="extra", action="store_false", help="não usar RDD2022/Roboflow")
    ap.add_argument("--compare", action="store_true", help="treina também YOLO11s e compara as arquiteturas")
    ap.add_argument("--smoke", action="store_true", help="teste de fumaça: 1 época, 320 px, 3%% do treino")
    ap.add_argument("--rf", default=None, help='SEU projeto Roboflow: "workspace/projeto/versao" (env ROBOFLOW_API_KEY)')
    ap.add_argument("--rf_alias", default="rf_rua")
    ap.add_argument("--project", default=None, help="pasta das corridas de treino (ex.: pasta do Drive no Colab)")
    ap.add_argument("--backup_dir", default=None, help="copia pesos/curvas para esta pasta ao fim do treino")
    a = ap.parse_args()
    gpu = a.device != "cpu"
    epochs = a.epochs or (1 if a.smoke else 10 if a.quick else (40 if gpu else 30))
    imgsz = a.imgsz or (320 if a.smoke else 416 if a.quick else 640)
    fraction = a.fraction or (0.03 if a.smoke else 0.4 if a.quick else 1.0)
    patience = 15 if epochs >= 30 else 10

    if a.step == "clean":
        for p in ["runs", "results", "weights/threshold.json", "reports/metrics.json", "reports/metrics_test.csv", "reports/RESULTADOS.md",
                  "reports/confusion_matrix.png", "reports/f1_vs_conf.png", "reports/error_analysis", "reports/demo",
                  "reports/COMPARACAO_MODELOS.md", "reports/comparacao_modelos.json"] + [str(x.relative_to(ROOT)) for x in ROOT.glob("weights/best*.pt")] \
                + [str(x.relative_to(ROOT)) for x in ROOT.glob("reports/training_curves*.png")]:
            t = ROOT / p
            shutil.rmtree(t, ignore_errors=True) if t.is_dir() else (t.unlink() if t.exists() else None)
        print("[clean] saídas geradas removidas (dados em data/ preservados)")
        return
    check_deps(need_gradio=(a.step == "app"))
    print(f"[versão] projeto {VERSION}")
    ext = len(list((ROOT / "tests" / "external_images").rglob("*.jp*g")))
    print(f"[check] dependências OK | imagens externas: {ext} | "
          f"sem_danos: {len(list((ROOT / 'tests/external_images/sem_danos').glob('*.*')))} (vazia = pendente)")
    if a.step == "check":
        print("[check] dataset:", "extraído" if any((ROOT / "data/raw").rglob("*_RAW.jpg")) else "ainda não extraído (será no 'prep')")
        print("[check] dados extras:", sorted(p.name for p in (ROOT / "data/extra").glob("*") if p.is_dir() and not p.name.startswith("_")) or "nenhum ainda")
        return
    yd = [] if a.extra else ["--yolo_dir", "data/yolo"]       # --no-extra força só o dataset brasileiro
    bench = "cpu,0" if gpu else "cpu"
    proj = (["--project", a.project] if a.project else []) + (["--backup_dir", a.backup_dir] if a.backup_dir else [])
    steps = (["prep"] + (["extra", "mix"] if a.extra else []) + ["train", "eval", "gallery", "demo", "sequence", "report"]) if a.step == "all" else [a.step]
    for s in steps:
        if s == "prep":
            ensure_raw()
            sh("data_prep.py", "--raw_dir", "data/raw", "--out_dir", "data/yolo", "--audit")
        elif s == "extra":
            ok1 = sh("data_extra.py", "rdd2022", fatal=False) == 0
            ok2 = sh("data_extra.py", "pothole_github", fatal=False) == 0
            if a.rf:
                ws, pj, ver = a.rf.split("/")
                sh("data_extra.py", "roboflow", "--workspace", ws, "--project", pj, "--version", ver, "--alias", a.rf_alias, fatal=False)
            if not (ok1 or ok2):
                print("[aviso] não foi possível baixar fontes extras (internet?). Seguindo só com o dataset brasileiro.")
                a.extra = False; yd = ["--yolo_dir", "data/yolo"]
        elif s == "mix":
            if a.extra:
                sh("build_mix.py")
        elif s == "train":
            sh("train_model.py", "--epochs", epochs, "--imgsz", imgsz, "--batch", 16 if gpu else 8, "--fraction", fraction,
               "--workers", 2, "--device", a.device, "--model", a.model, "--preset", a.preset, "--patience", patience, *yd, *proj)
            if a.compare and a.model == "yolo11n":
                sh("train_model.py", "--epochs", epochs, "--imgsz", imgsz, "--batch", 16 if gpu else 8, "--fraction", fraction,
                   "--workers", 2, "--device", a.device, "--model", "yolo11s", "--preset", a.preset, "--patience", patience,
                   "--name", "train_s", "--no_default", *yd)
        elif s == "eval":
            sh("evaluate.py", "--external_pothole", "--device", a.device, "--bench_devices", bench, *yd)
            if a.compare and a.model == "yolo11n":
                sh("compare_models.py", "--weights", "weights/best_train.pt", "weights/best_train_s.pt",
                   "--labels", "YOLO11n", "YOLO11s", "--device", a.device, *yd)
        elif s == "gallery":
            sh("demo_gallery.py", *yd)
        elif s == "demo":
            sh("infer.py", "--source", "tests/external_images", "--out", "results/external", "--expected_from_folders")
        elif s == "sequence":
            sh("sequence_demo.py", "--n_frames", 40, *yd)
        elif s == "report":
            sh("make_report.py")
        elif s == "app":
            sh("app_gradio.py")
    if "demo" in steps and (ROOT / "results/external/galeria.html").exists():
        try:
            webbrowser.open((ROOT / "results/external/galeria.html").as_uri())
        except Exception:
            pass
    print("\n[ok] concluído. Veja reports/RESULTADOS.md, reports/*.png, results/external/galeria.html, results/sequence/ "
          "e rode `python run.py app` para a interface (roteiro: ROTEIRO_VIDEO.md)")


if __name__ == "__main__":
    main()
