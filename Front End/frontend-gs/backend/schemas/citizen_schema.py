"""Modelos Pydantic de entrada e saída (contrato HTTP + saída estruturada do LLM)."""
import re
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------- vocabulário
Categoria = Literal[
    "iluminacao_publica",
    "limpeza_urbana_entulho",
    "buraco_pavimentacao",
    "saude",
    "seguranca",
    "saneamento_agua_esgoto",
    "arvores_areas_verdes",
    "transporte_mobilidade",
    "acessibilidade",
    "outros",
]
Urgencia = Literal["baixa", "media", "alta", "critica"]

CATEGORIA_LABELS: dict[str, str] = {
    "iluminacao_publica": "Iluminação pública",
    "limpeza_urbana_entulho": "Limpeza urbana e entulho",
    "buraco_pavimentacao": "Buracos e pavimentação",
    "saude": "Saúde",
    "seguranca": "Segurança",
    "saneamento_agua_esgoto": "Água, esgoto e drenagem",
    "arvores_areas_verdes": "Árvores e áreas verdes",
    "transporte_mobilidade": "Transporte e mobilidade",
    "acessibilidade": "Acessibilidade",
    "outros": "Outros serviços públicos",
}


def _clip(value, limit: int):
    return value[:limit].rstrip() if isinstance(value, str) else value


# ---------------------------------------------------------------- entrada
class AnalysisRequest(BaseModel):
    """Relato do cidadão. Validado no back-end mesmo que o front já valide."""

    relato: str = Field(..., min_length=20, max_length=1000,
                        description="Descrição do problema em linguagem natural.")
    bairro: Optional[str] = Field(None, max_length=80)
    cep: Optional[str] = Field(None, description="CEP com 8 dígitos (opcional).")
    categoria_sugerida: Optional[Categoria] = None

    @field_validator("relato", "bairro", "cep", mode="before")
    @classmethod
    def _strip(cls, v):
        if isinstance(v, str):
            v = re.sub(r"\s+", " ", v).strip()
            return v or None
        return v

    @field_validator("relato")
    @classmethod
    def _relato_com_conteudo(cls, v):
        if v is None or len(re.findall(r"[A-Za-zÀ-ÿ]{2,}", v)) < 3:
            raise ValueError("O relato precisa ter pelo menos 3 palavras.")
        return v

    @field_validator("cep")
    @classmethod
    def _cep_valido(cls, v):
        if v is None:
            return v
        digits = re.sub(r"\D", "", v)
        if len(digits) != 8:
            raise ValueError("CEP deve ter 8 dígitos.")
        return digits


# ---------------------------------------------------------------- saída do LLM
class LLMAnalysis(BaseModel):
    """Estrutura que o LLM é OBRIGADO a devolver (também enviada como JSON Schema ao Ollama)."""

    categoria: Categoria
    urgencia: Urgencia
    resumo: str = Field(..., min_length=5, max_length=300,
                        description="Resumo do relato em 1 frase simples.")
    justificativa_urgencia: str = Field(..., min_length=3, max_length=300)
    orgao_responsavel: str = Field(..., min_length=2, max_length=120)
    orientacoes: list[str] = Field(..., min_length=1, max_length=6,
                                   description="Passos práticos, em linguagem simples.")
    prazo_estimado: str = Field(..., min_length=2, max_length=80)
    risco_imediato: bool
    confianca: float = Field(..., ge=0, le=1)

    @field_validator("categoria", "urgencia", mode="before")
    @classmethod
    def _normaliza_enum(cls, v):
        if isinstance(v, str):
            v = v.strip().lower().replace("é", "e").replace("í", "i")
        return v

    @field_validator("resumo", "justificativa_urgencia", "orgao_responsavel",
                     "prazo_estimado", mode="before")
    @classmethod
    def _clip_textos(cls, v, info):
        limits = {"resumo": 300, "justificativa_urgencia": 300,
                  "orgao_responsavel": 120, "prazo_estimado": 80}
        return _clip(v.strip() if isinstance(v, str) else v, limits[info.field_name])

    @field_validator("orientacoes", mode="before")
    @classmethod
    def _normaliza_orientacoes(cls, v):
        if isinstance(v, str):
            v = [p for p in re.split(r"\n+|\d+[.)]\s+", v)]
        if isinstance(v, list):
            v = [_clip(str(x).strip(), 300) for x in v if str(x).strip()]
            return v[:6]
        return v

    @field_validator("confianca", mode="before")
    @classmethod
    def _normaliza_confianca(cls, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return v
        if 1 < v <= 100:
            v = v / 100
        return min(max(v, 0.0), 1.0)


class EnderecoInfo(BaseModel):
    logradouro: Optional[str] = None
    bairro: Optional[str] = None
    cidade: Optional[str] = None
    uf: Optional[str] = None


# ---------------------------------------------------------------- saída da API
class AnalysisResponse(BaseModel):
    protocolo: str
    criado_em: datetime
    categoria_label: str
    analise: LLMAnalysis
    telefones_emergencia: list[str] = []
    alerta_seguranca: bool = Field(
        False, description="True quando a regra de segurança elevou a urgência (palavras de emergência).")
    dados_pessoais_mascarados: bool = False
    endereco: Optional[EnderecoInfo] = None
    modelo: str
    latencia_ms: int


class FeedbackRequest(BaseModel):
    """Human-in-the-loop: decisão do cidadão sobre o resultado da IA."""

    protocolo: str = Field(..., pattern=r"^CT-[A-Z0-9-]{6,30}$")
    decisao: Literal["confirmado", "corrigido", "descartado"]
    categoria_final: Optional[Categoria] = None
    avaliacao: Optional[Literal["util", "incorreta"]] = None
    comentario: Optional[str] = Field(None, max_length=300)


class FeedbackResponse(BaseModel):
    status: Literal["registrado"]
    protocolo: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degradado"]
    api: bool = True
    ollama: bool
    modelo: str
    modelo_instalado: bool
