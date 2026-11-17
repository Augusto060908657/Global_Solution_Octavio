# CitizenTech "Cidade Para Todos" Assistente Conversacional RAG

**Global Solution FIAP 2026.2 · Generative AI & Advanced Nets · Turma 2TIAPF**

| Integrante | RM |
|---|---|
| Octavio Augusto Ramalho Brandão Pires | 555696 |

Assistente que orienta o cidadão sobre serviços públicos, acessibilidade e direitos urbanos usando **RAG**: ingestão e chunking híbrido → FAISS + BM25 (Top-K = 3, limiar calibrado) → gerador que só responde com base no contexto, cita fontes e se abstém quando não há evidência. Inclui proteção contra *prompt injection*, mascaramento de PII, API FastAPI, chat de demonstração e avaliação automatizada que gera o relatório com os números reais.

[Link do Video](https://youtu.be/aU0g1XikJJ4)
[Link do Projeto no Colab](https://colab.research.google.com/drive/13heTcMDHirc_Ce4cULGMSEVxy2NqhEpm?usp=sharing)

> **Corpus Simulado:** o corpus (`data/`) é **sintético**. O município, os prazos e as URLs `cidadeparatodos.example.gov.br` são fictícios, escritos para o protótipo. Para usar fontes reais (gov.br, Cartas de Serviços), salve cada página como `.md` em `data/` com o cabeçalho de metadados (URL real e data de consulta) e rode `python evaluate_rag.py` de novo; veja a seção 6.

---

## 1. Como rodar (Windows / macOS / Linux)

Requisitos: **Python 3.10 a 3.12** e internet na primeira execução (baixa o modelo de embeddings, ~470 MB).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate          macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env   (opcional)
```

### LLM 100% gratuito com Ollama (recomendado, sem créditos de API)

1. Instale o Ollama: https://ollama.com/download (Windows, macOS ou Linux) e abra o app (ou rode `ollama serve`).
2. Baixe o modelo uma vez (~2 GB): `ollama pull qwen2.5:3b`
3. No `.env`: `LLM_PROVIDER=ollama` (já é o padrão do `.env.example`).

Alternativas: `LLM_PROVIDER=extractive` (sem LLM, offline, cada frase vem literalmente da base) ou `openai` (pago, exige `OPENAI_API_KEY` com crédito). **Se o gerador falhar** (chave inválida, sem crédito, Ollama fora do ar), o assistente responde "serviço de geração indisponível" (não "sem evidência") e `evaluate_rag.py` **para com uma mensagem clara**, em vez de gerar um relatório inválido.

| O que | Comando |
|---|---|
| **Avaliação completa + relatório** (rode antes de gravar) | `python evaluate_rag.py` |
| **API + chat de demonstração** | `uvicorn api_service:app --port 8000` → http://localhost:8000 (Swagger em `/docs`) |
| Pergunta pelo terminal | `python rag_chain.py "Qual o prazo para reparar um buraco?"` |
| Ver o chunking | `python ingestion.py` |
| Ver o retriever | `python retrieval.py "Como peço rampa na calçada?"` |
| Testes rápidos de regressão (opcional) | `pip install pytest && python -m pytest -q` |

Ao rodar `evaluate_rag.py` são gerados: `resultados/resultados_avaliacao.json` (contexto recuperado, resposta e julgamento de cada pergunta), `resultados/tabela_end_to_end.csv`, `resultados/secao_resultados.md`, `resultados/calibracao.json` e **`RELATORIO_FINAL.md`** (relatório completo com os números da sua execução). Nada precisa ser preenchido à mão.

Se o terminal mostrar `[AVISO] ... fallback 'hashing'`, o modelo de embeddings não carregou (sem internet ou sem `sentence-transformers`): **não use esses números no relatório**; rode de novo com internet.

Exemplo de chamada à API:

```bash
curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" \
  -d '{"question":"Qual o prazo para reparar um buraco?","session_id":"demo-001"}'
```

## 2. Corpus e configuração

**Corpus** (`data/`, 11 documentos, ~42 chunks): Cartas de Serviços do protótipo (zeladoria, limpeza urbana, meio ambiente, acessibilidade, transporte, tributos, ouvidoria, privacidade, atendimento). Cada arquivo tem `titulo`, `categoria`, `url_origem`, `data_atualizacao`, `data_consulta` (05/11/2026).

| Componente | Configuração (`config.py`) |
|---|---|
| Chunking | Híbrido: `MarkdownHeaderTextSplitter` (#, ##) + `RecursiveCharacterTextSplitter`, **500 / 50**, prefixo "Título > Seção" |
| Metadados | `source`, `doc_id`, `titulo`, `secao`, `categoria`, `url_origem`, `data_atualizacao`, `data_consulta`, `chunk_index` |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dim) |
| Índice | FAISS `IndexFlatIP` (cosseno) |
| Retriever | Híbrido FAISS + BM25 com fusão RRF; **k = 3**; limiar de cosseno **calibrado** (`calibration.py`) |
| LLM | **Ollama `qwen2.5:3b`** (gratuito, local), temperatura 0, `max_tokens` 450; alternativas: `openai`, `extractive` |
| Prompt | Papel, só CONTEXTO, abstenção `Não encontrei essa informação na base oficial.`, citação `[n]`, anti-injeção, canário |

## 3. Como a avaliação evita erros metodológicos

- **Sem *data leakage*:** 5 perguntas exclusivas calibram o limiar; 15 perguntas de ouro testam recuperação/gerador; 12 perguntas finais testam o sistema completo (`eval_sets.py`).
- **Retriever:** Hit@3 e MRR comparando fixo × híbrido (300/30, 500/50, 800/80) e denso × híbrido.
- **Gerador:** correção, citação, taxa de abstenção com contexto irrelevante e números não sustentados.
- **End-to-end (4 categorias):** 5 respondíveis, 3 sem resposta/fora do escopo, 2 ambíguas, 2 adversariais (prompt injection e PII).
- **Extras de segurança:** injeção indireta (chunk envenenado), vazamento de prompt, guardrail de saída, variações de ataque × falsos positivos, multi-turno.

## 4. Mapa do barema → evidência

| Critério (peso) | Onde está |
|---|---|
| Chunking (1,0) | `ingestion.py` (docstring), Tabela A, `RELATORIO_FINAL.md` §1.2 |
| Retriever (1,0) | `retrieval.py`, `calibration.py`, Tabela A e de limiar, §1.3 |
| Generator (1,0) | `SYSTEM_PROMPT` e `LLMClient` em `rag_chain.py`, §1.4 |
| Implementação e integração (1,0) | `ingestion.py` → `retrieval.py` → `rag_chain.py` → `api_service.py` + `static/index.html` |
| Avaliação dos componentes (1,5) | `evaluate_rag.py` etapas A e B, §2.1 e §2.2 |
| Avaliação end-to-end (1,5) | etapas C e D, §2.3, `resultados/tabela_end_to_end.csv` |
| Mitigação (1,0) | `security.py`, camadas em `rag_chain.py`, §3 |
| Interpretação (2,0) | §4 do `RELATORIO_FINAL.md` (falhas, impacto, limites) |

## 5. Estrutura

```
config.py        parâmetros e identificação do grupo     text_utils.py   normalização/tokenização PT-BR
ingestion.py     chunking                                 retrieval.py    FAISS + BM25 + limiar
calibration.py   calibração do limiar (5 perguntas)      rag_chain.py    prompt, guardrails e pipeline
security.py      injeção, PII, vazamento                  api_service.py  FastAPI (POST /api/v1/chat)
eval_sets.py     conjuntos de perguntas                   evaluate_rag.py avaliação + geração de artefatos
relatorio.py     gera RELATORIO_FINAL.md                  test_smoke.py   testes rápidos
static/          chat de demonstração                     data/           corpus     ROTEIRO_VIDEO.md  roteiro de 5 min
```

## 6. Trocar o corpus por fontes reais

1. Salve de 10 a 15 páginas oficiais (gov.br, Carta de Serviços da prefeitura) em `data/*.md`, com `#`/`##` e o cabeçalho `titulo`, `categoria`, `url_origem`, `data_atualizacao`, `data_consulta`.
2. Ajuste `OFFICIAL_CHANNEL` e `SERVICE_EXAMPLES` em `config.py`.
3. Reescreva `GOLD`, `CALIBRACAO_*` e `E2E_CASES` em `eval_sets.py` para os novos temas (documento esperado + evidência-chave) e rode `python evaluate_rag.py`.

## 7. Segurança e privacidade

- Nenhuma credencial no código; `.env` está no `.gitignore`.
- CPF, CNPJ, e-mail, telefone e cartão são mascarados **antes** da recuperação, do LLM e do histórico; o log da API guarda só *hashes*.
- Histórico de sessão só em memória (3 turnos), já mascarado.

## 8. Solução de problemas

- `ModuleNotFoundError: faiss` → `pip install faiss-cpu` (no Windows prefira Python 3.11/3.12).
- Falha ao baixar o modelo → verifique internet/proxy; o sistema avisa e usa o fallback `hashing`.
- `GERADOR INDISPONÍVEL — ... AuthenticationError` → chave da OpenAI inválida/sem crédito: use `LLM_PROVIDER=ollama`.
- `GERADOR INDISPONÍVEL — ... APIConnectionError` → Ollama não está rodando (`ollama serve`) ou o modelo não foi baixado (`ollama pull qwen2.5:3b`).
- Respostas lentas na 1ª pergunta com Ollama: normal (carrega o modelo); em CPU, uma resposta leva alguns segundos.
- Mudou o corpus, os chunks ou o modelo → o limiar é recalibrado automaticamente (ou apague `resultados/calibracao.json`).
