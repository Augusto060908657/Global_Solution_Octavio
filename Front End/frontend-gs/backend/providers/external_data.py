"""Dados abertos externos (diferencial): enriquece o relato com o endereço do CEP via ViaCEP."""
import logging
from typing import Optional

import httpx

logger = logging.getLogger("citizentech.external")


def buscar_cep(cep: str, timeout: float = 4.0) -> Optional[dict]:
    """Retorna {logradouro,bairro,cidade,uf} ou None. Falha silenciosa: dado externo é opcional."""
    try:
        r = httpx.get(f"https://viacep.com.br/ws/{cep}/json/", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("erro"):
            return None
        return {
            "logradouro": data.get("logradouro") or None,
            "bairro": data.get("bairro") or None,
            "cidade": data.get("localidade") or None,
            "uf": data.get("uf") or None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.info("ViaCEP indisponível (%s); seguindo sem enriquecimento.", exc)
        return None
