# 🏙️ CitizenTech — Cidade Para Todos
### Integração Arquitetural, Governança em IA e Business Analytics

> **Instituição:** FIAP — Faculdade de Informática e Administração Paulista  
> **Curso:** Tecnologia em Inteligência Artificial · Turma 2TIAPY (2026.2)  
> **Integrante:** Octavio Augusto Ramalho Brandão Pires — **RM 555696**  

---

## 🔗 Entregáveis Principais (Governança em IA & Business Analytics)

* **Vídeo de Demonstração (YouTube — Não Listado):** (https://youtu.be/KlJ37WnXEHs)

---

## 1. Visão Geral do Ecossistema

O **CitizenTech - Cidade Para Todos** é um protótipo de plataforma cívica multimodal desenvolvido ao longo da Global Solution 2026.2 para apoiar o atendimento ao cidadão e a triagem de zeladoria urbana (buracos, iluminação, poda de árvores, entulho, alagamentos e acessibilidade)[cite: 1, 16]. 

A solução ataca duas barreiras centrais dos canais públicos atuais:
1. **Exclusão digital e barreiras de acesso:** oferece múltiplos canais de interação adaptados ao perfil do cidadão, texto simples com alto contraste, assistente de voz para idosos e pessoas com baixa alfabetização[cite: 16], e detecção sonora local para pessoas surdas ou com deficiência auditiva[cite: 24, 25].
2. **Triagem manual lenta e dados ruidosos:** automatiza a validação de registros, o cruzamento com alertas climáticos, a detecção de danos viários por imagem e a orientação sobre direitos urbanos fundamentada em documentos.

Este repositório consolida os **8 módulos técnicos** construídos nas disciplinas do semestre sob a ótica de **Governança em IA** (privacidade/LGPD, transparência, licenciamento, mitigação de riscos e revisão humana) e **Business Analytics** (indicadores operacionais e planejamento de capacidade).

---

## 2. Arquitetura Integrada e Fluxo de Dados

```mermaid
flowchart TB
    subgraph Canais["Canais de Entrada (Acessibilidade Multimodal)"]
        C1["Portal Web (Streamlit)<br/>Texto + Alto Contraste + ARIA"]
        C2["Assistente de Voz (Zelador)<br/>Áudio pt-BR via Gradio"]
        C3["Edge Audio (OuviAí)<br/>Microfone Smartphone (Navegador)"]
        C4["Inspeção Viária (Visão Computacional)<br/>Fotos / Sequência via Gradio"]
    end

    subgraph Processamento["Camada de IA e Automação"]
        API["Back-End FastAPI<br/>Validação Pydantic + ViaCEP"]
        LLM["Ollama + Qwen2.5 (Local)<br/>Saída JSON Estruturada"]
        RAG["Motor RAG (FAISS + BM25)<br/>Chunking Híbrido + Limiar Calibrado"]
        CV["Detector YOLO11n<br/>Buracos e Rachaduras"]
        RPA["Pipeline RPA (Batch)<br/>Open-Meteo + Pandas + SQLite"]
    end

    subgraph Governanca["Controles de Governança e LGPD"]
        G1["Mascaramento de PII (CPF, E-mail, Tel.)"]
        G2["Regras Determinísticas de Emergência<br/>(190 / 192 / 193 / 199)"]
        G3["Human-in-the-Loop & Quarentena<br/>Revisão Humana em Fronteira"]
    end

    subgraph Escala["Estudos Offline de Escala e Futuro"]
        E1["Cluster Computing (iSPD)<br/>Simulação de Picos (2 vs. 4 Workers)"]
        E2["Computação Quântica (Qiskit/NumPy)<br/>Grover, Swap Test e Roadmap PQC"]
    end

    C1 --> API --> G1 --> LLM --> G2 --> G3
    C2 --> G1
    C3 -. "100% Local (WASM / Memória Volátil)" .-> G3
    C4 --> CV -. "Integração Planejada" .-> RPA
    API <--> RAG
    RPA --> G1 --> G3
    G3 --> BA["Business Analytics<br/>Dashboards Plotly + Relatórios JSON/MD"]
    Escala -. "Planejamento de Infraestrutura e Criptografia" .-> Processamento
