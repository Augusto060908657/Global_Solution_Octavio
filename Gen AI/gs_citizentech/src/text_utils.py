"""Utilitários de texto compartilhados (normalização e tokenização leve para PT-BR)."""
import re
import unicodedata
from typing import List

STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "por", "para", "pra", "com", "sem", "sobre", "e", "ou", "mas", "que", "se", "ao", "aos",
    "ja", "mais", "muito", "meu", "minha", "meus", "minhas", "seu", "sua", "seus", "suas", "isso", "isto",
    "esse", "essa", "este", "esta", "aquele", "aquela", "eu", "voce", "nos", "ele", "ela", "me", "te",
    "como", "qual", "quais", "quanto", "quantos", "quanta", "quando", "onde", "quem", "porque", "posso",
    "pode", "podem", "preciso", "precisa", "quero", "quer", "fazer", "faco", "devo", "deve", "tenho", "tem",
    "ter", "gostaria", "saber", "existe", "ha", "foi", "ser", "sao", "eh", "e", "esta", "estao", "estou",
    "ai", "entao", "tambem", "so", "ate", "entre", "uso", "usar", "fica", "ficar", "vai", "vou", "ir",
    "caso", "casos", "conta", "propria", "proprio", "alguma", "algum", "qualquer", "todo", "toda",
    "rua", "ruas", "bairro", "cidade", "prefeitura", "municipio", "municipal", "ajuda", "informacao",
}

# Sinônimos mínimos do domínio (aplicados a documentos e perguntas) para aproximar vocabulário
# do cidadão e vocabulário oficial. Em produção, substituir por reranker/embeddings melhores.
SYNONYMS = {
    "consertar": "reparar", "conserto": "reparo", "arrumar": "reparar", "tapar": "reparar",
    "cadeirante": "deficiencia", "deficiente": "deficiencia", "pcd": "deficiencia",
    "onibus": "transporte", "passagem": "transporte", "velho": "velhos", "lampada": "lampada",
}


# Etiquetas inseridas pelo mascaramento de PII (ex.: "[CPF OCULTADO]"). Não são conteúdo da pergunta:
# precisam sair antes de tokenizar/embutir, senão derrubam a cobertura lexical e a similaridade.
_PII_TAG = re.compile(r"\[[A-Za-z\-]+\s+OCULTAD[OA]\]")


# "meu CPF é [CPF OCULTADO]" -> o rótulo ("meu CPF é") também é só moldura do dado, não a intenção da pergunta.
_PII_FRAME = re.compile(
    r"(?:\b(?:meu|minha|o|a|seu|sua)\s+)?\b(?:cpf|cnpj|e-?mail|telefone|celular|cart[aã]o(?:\s+de\s+cr[eé]dito)?)\s*"
    r"(?:\u00e9|e|eh|:|=)?\s*\[[A-Za-z\-]+\s+OCULTAD[OA]\]", re.IGNORECASE)


def strip_pii_tags(text: str) -> str:
    """Remove as etiquetas de sanitização ([CPF OCULTADO]...) e a moldura que as acompanha ('meu CPF é')."""
    text = _PII_FRAME.sub(" ", text)
    return re.sub(r"\s{2,}", " ", _PII_TAG.sub(" ", text)).strip()


def normalize(text: str) -> str:
    """Minúsculas, sem acentos e sem caracteres invisíveis (zero-width)."""
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def stem(token: str) -> str:
    """Stemming grosseiro por prefixo (5 caracteres): une singular/plural e flexões verbais."""
    return token[:5] if len(token) > 5 and not token.isdigit() else token


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    tokens = [SYNONYMS.get(t, t) for t in re.findall(r"[a-z0-9]+", normalize(strip_pii_tags(text)))]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return [stem(t) for t in tokens]
