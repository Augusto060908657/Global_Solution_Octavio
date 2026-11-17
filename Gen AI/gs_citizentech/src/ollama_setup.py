"""Prepara o Ollama (LLM local e GRATUITO). Não depende de config.py (pode rodar antes dele).

Uso no notebook:  import ollama_setup; ollama_setup.ensure_ollama()
Uso local: instale o Ollama (https://ollama.com/download), rode `ollama pull qwen2.5:3b` e defina LLM_PROVIDER=ollama.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/").removesuffix("/v1")


def _tags():
    try:
        with urllib.request.urlopen(BASE + "/api/tags", timeout=2) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return None


def ensure_ollama(model: str = None, timeout_start: int = 90) -> None:
    """Garante: binário instalado, servidor no ar e modelo baixado. Lança RuntimeError com instruções se não der."""
    model = model or MODEL
    if _tags() is None:
        if shutil.which("ollama") is None:
            if sys.platform.startswith("linux") and (os.path.exists("/content") or os.getenv("OLLAMA_AUTO_INSTALL") == "1"):
                print("Instalando o Ollama (gratuito)... pode levar 1-2 min")
                subprocess.run("(command -v zstd >/dev/null || (apt-get update -qq && apt-get install -y -qq zstd >/dev/null 2>&1)); "
                               "curl -fsSL https://ollama.com/install.sh | sh >/dev/null 2>&1", shell=True, check=False)
            if shutil.which("ollama") is None:
                raise RuntimeError("Ollama não está instalado. Baixe em https://ollama.com/download, instale, abra o app (ou rode "
                                   f"'ollama serve') e execute 'ollama pull {model}'. Depois rode de novo.")
        print("Iniciando o servidor do Ollama...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(timeout_start):
            if _tags() is not None:
                break
            time.sleep(1)
        else:
            raise RuntimeError("O servidor do Ollama não respondeu em 90 s. Rode 'ollama serve' em outro terminal.")
    nomes = _tags() or []
    if model not in nomes and f"{model}:latest" not in nomes:
        print(f"Baixando o modelo {model} (uma vez só, ~2 GB)...")
        if subprocess.run(["ollama", "pull", model]).returncode != 0:
            raise RuntimeError(f"Falha ao baixar o modelo. Rode manualmente: ollama pull {model}")
    print(f"Ollama pronto: modelo {model} em {BASE}")


if __name__ == "__main__":
    ensure_ollama()
