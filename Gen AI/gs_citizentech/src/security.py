"""Guardrails de segurança: prompt injection, mascaramento de PII, vazamento de prompt e chunks envenenados."""
import re
from typing import List, Tuple

from text_utils import normalize

# Token secreto embutido no system prompt. Se aparecer numa resposta, houve vazamento do prompt.
CANARY = "CPT-CANARY-7f3a91"

# Padrões aplicados sobre texto normalizado (minúsculo, sem acentos).
_INJECTION_PATTERNS = [
    r"\bignor\w*\s+(todas?\s+)?(as\s+|suas\s+|os\s+|seus\s+|a\s+)?(instruc\w+|regras?|ordens|comandos?|diretrizes|restric\w+)",
    r"\bignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|prompts?)",
    r"\b(esqueca|desconsidere|desobedeca|descarte|sobrescreva)\b.{0,40}\b(instruc\w+|regras?|prompt|diretrizes)",
    r"\b(revele|mostre|exiba|imprima|repita|diga|vaze|liste|escreva|copie|traduza)\b.{0,40}\b(system\s*prompt|prompt\s+(de\s+)?sistema|prompt\s+inicial|seu\s+prompt|suas\s+instruc\w+|instruc\w+\s+(internas|iniciais|do\s+sistema|anteriores|ocultas))",
    r"\b(system|developer)\s*(prompt|message|instructions?)\b",
    r"\b(reveal|show|print|repeat)\b.{0,30}\b(your|the)\b.{0,20}\b(prompt|instructions?)\b",
    r"\bvoce\s+(agora\s+)?(e|sera|vai ser)\s+(um|uma|o|a)\s+(?!cidadao)\w+",
    r"\b(finja|simule|aja\s+como|atue\s+como|haja\s+como|assuma\s+o\s+papel|pretend\s+to\s+be|act\s+as)\b",
    r"\b(jailbreak|do\s+anything\s+now)\b|\bdan\s+mode\b|\bmodo\s+(desenvolvedor|deus|dev|sem\s+restric\w+)\b",
    r"\b(senha|password|chave|api\s*key|token|credenciais?|segredo|secret)\s+(do|da|de|dos|das)\s+(sistema|servidor|admin\w*|banco\s+de\s+dados|ia|assistente|aplicacao)",
    r"\bchave\s+(de\s+)?api\b|\bapi[\s_-]*key\b",
    r"\bvariaveis?\s+de\s+ambiente\b|\b\.env\b|\bopenai_api_key\b",
    r"\b(dados|informacoes|cpf|senhas?)\s+(de\s+)?(outros|todos\s+os|demais)\s+(usuarios|cidadaos|pessoas|clientes)",
]
_COMPILED_INJECTION = [re.compile(p) for p in _INJECTION_PATTERNS]

_PII_PATTERNS = [
    ("CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("CNPJ", re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")),
    ("E-MAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("TELEFONE", re.compile(r"(?<!\d)(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}[-\s]?\d{4}(?!\d)")),
    ("CARTAO", re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b")),
]

_LEAK_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),  # formato de chave de API
    re.compile(r"(?i)system prompt"),
    re.compile(r"(?i)regras inviolaveis|regras inviolá"),
]


def detect_prompt_injection(text: str) -> Tuple[bool, str]:
    """Retorna (detectado, padrão). Pensado para precisão: tenta não bloquear dúvidas legítimas."""
    norm = re.sub(r"\s+", " ", normalize(text))
    for pattern in _COMPILED_INJECTION:
        if pattern.search(norm):
            return True, pattern.pattern[:60]
    return False, ""


def mask_pii(text: str) -> Tuple[str, List[str]]:
    """Mascara CPF, CNPJ, e-mail, telefone e cartão ANTES de enviar ao LLM ou registrar logs."""
    found: List[str] = []
    for label, pattern in _PII_PATTERNS:
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(f"[{label} OCULTADO]", text)
    return text, found


def contains_leak(answer: str) -> bool:
    """Guardrail de saída: detecta vazamento do canário, de chaves ou de trechos do prompt."""
    if CANARY.lower() in answer.lower():
        return True
    return any(p.search(answer) for p in _LEAK_PATTERNS)


def is_poisoned_chunk(chunk_text: str) -> bool:
    """Injeção INDIRETA: documento recuperado contendo instruções dirigidas ao modelo."""
    detected, _ = detect_prompt_injection(chunk_text)
    return detected
