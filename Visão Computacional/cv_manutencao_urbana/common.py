"""
Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696
GS FIAP 2026.2 - CitizenTech "Cidade Para Todos" - Módulo de Visão Computacional (Manutenção Urbana)

common.py - constantes e utilitários compartilhados (evoluído da versão MobileNetV3 para YOLO).
Todos os caminhos são relativos à pasta do projeto (ROOT).
"""
import platform
from pathlib import Path

AUTHOR = "Octavio Augusto Ramalho Brandão Pires | RM 555696"
VERSION = "3.1.0"   # confira no notebook: se divergir, o zip do Drive está desatualizado
ROOT = Path(__file__).resolve().parent

SEED = 42
IMGSZ = 640                      # resolução padrão de inferência/treino (ajustável por CLI)
CLASS_NAMES = ["pothole", "crack"]                      # ids YOLO 0 e 1
CLASS_PT = {"pothole": "Buraco", "crack": "Rachadura"}  # nomes exibidos ao cidadão/gestor
CLASS_COLORS_BGR = {"pothole": (0, 0, 230), "crack": (0, 165, 255)}  # vermelho / laranja

RAW_DIR = ROOT / "data" / "raw"          # descompactar aqui o dataset (ver README)
YOLO_DIR = ROOT / "data" / "yolo"        # saída de data_prep.py
WEIGHTS_DIR = ROOT / "weights"
PRETRAINED = WEIGHTS_DIR / "yolo11n.pt"  # YOLO11-nano pré-treinado no COCO
BEST_PT = WEIGHTS_DIR / "best.pt"        # pesos treinados (cópia do melhor checkpoint)
RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "reports"
EXTERNAL_DIR = ROOT / "tests" / "external_images"


MIX_DIR = ROOT / "data" / "yolo_mix"     # saída de build_mix.py (Brasil + RDD2022 + Roboflow)
EXTRA_DIR = ROOT / "data" / "extra"      # fontes extras baixadas por data_extra.py


def write_dataset_yaml(yolo_dir: Path = YOLO_DIR, test_list: str = None, name: str = "dataset.yaml") -> Path:
    """(Re)gera o yaml com o caminho absoluto da máquina atual (exigência do Ultralytics).
    test_list: arquivo .txt (relativo a yolo_dir) com imagens de teste, para avaliar uma fonte isolada."""
    yolo_dir = Path(yolo_dir).resolve()
    yaml_path = yolo_dir / name
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    test = test_list or "images/test"
    yaml_path.write_text(f"path: {yolo_dir.as_posix()}\ntrain: images/train\nval: images/val\ntest: {test}\n"
                         f"names:\n{names}\n", encoding="utf-8")
    return yaml_path


def default_yolo_dir() -> Path:
    """data/yolo_mix se existir (dados combinados), senão data/yolo (só dataset brasileiro)."""
    return MIX_DIR if (MIX_DIR / "images" / "train").exists() else YOLO_DIR


def ensure_pretrained(path: Path = PRETRAINED) -> Path:
    """Garante o arquivo de pesos pré-treinados (baixa do release oficial da Ultralytics se faltar)."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(str(path))
    return path


def hardware_info() -> dict:
    import torch
    cpu = platform.processor() or platform.machine()
    try:  # nome amigável do CPU no Linux
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    import os
    gpu = None
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = "cuda"
    return {"cpu": cpu, "gpu": gpu, "cpu_threads": os.cpu_count(), "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available())}
