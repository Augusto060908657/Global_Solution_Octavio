# Cidade Para Todos — Robô RPA + IA (CitizenTech)

**Global Solution FIAP 2026.2 · AI for Robotic Process Automation (RPA + IA)**

**Integrante:** Octavio Augusto Ramalho Brandão Pires — RM 555696

## 1. Problema

Prefeituras recebem diariamente chamados de cidadãos (buracos, semáforos quebrados, rampas de acessibilidade obstruídas, alagamentos). A triagem manual é lenta, os dados chegam sujos (datas inválidas, sem coordenadas, textos vazios) e os casos que colocam vidas em risco, sobretudo de pessoas com deficiência, idosos e crianças, ficam misturados aos casos triviais.

O robô automatiza, de ponta a ponta: **coleta do clima → leitura e validação dos chamados → priorização por regras inteligentes → alerta de emergência acessível e resumo de zeladoria por e-mail → relatório de execução**, sem nunca parar diante de erro de entrada ou de API fora do ar.

## 2. Arquitetura

```
 Open-Meteo (API) ──► coletar_clima ──┐   timeout 5s · retry exponencial · circuit breaker
        ▲ falha                       │   └─ FALLBACK: cache local → clima neutro + revisão humana
        └─ cache/clima_cache.json     ▼
 chamados_cidadao_test.csv ─► carregar ─► validar/sanitizar ─► quarentena (CSV com motivo)
                                              │ válidos (CPF/e-mail/telefone mascarados)
                                              ▼
                       priorizar (léxico NLP + categoria + clima) ─► SQLite (INSERT OR IGNORE)
                                              │ somente chamados NOVOS (idempotência)
                                              ▼
              e-mail ALERTA DE EMERGÊNCIA  +  e-mail RESUMO DE ZELADORIA  (SMTP ou .eml simulado)
                                              ▼
                 outputs/ (CSV priorizado, quarentena, resumo JSON/MD) + rpa_execution.log
```

## 3. Como executar

```bash
pip install -r requirements.txt
cp .env.example .env                 # (Windows: copy .env.example .env)
python generate_test_data.py         # gera chamados_cidadao_test.csv (dados sintéticos)
python rpa_pipeline.py               # executa o robô (ponto de entrada: main())
python test_rpa_resilience.py        # demonstração + relatório de evidências
pytest -v test_rpa_resilience.py     # 15 testes automatizados
```

Funciona sem configurar nada: o padrão é e-mail **simulado** (grava `.eml` em `outputs/outbox/`) e a API Open-Meteo **não exige chave**. Sem internet, o robô usa o cache local (`cache/clima_cache.json`).

### Como configurar as chaves

Todas as credenciais ficam no arquivo `.env` (ignorado pelo Git via `.gitignore`); não há senha no código. Para enviar e-mails reais, defina no `.env`:

```
MODO_EMAIL=smtp
SMTP_HOST=smtp.seuprovedor.com
SMTP_PORT=587
SMTP_USER=seu_usuario
SMTP_PASSWORD=sua_senha_de_app
EMAIL_TO=gestor@exemplo.com,campo@exemplo.com
```

Se o SMTP falhar ou estiver sem credenciais, a mensagem é preservada em `outputs/outbox/` e o robô continua.

### Execução automática (sem intervenção)

Agendar `python rpa_pipeline.py` — Linux/macOS: `0 7 * * * cd /caminho && python rpa_pipeline.py` (cron); Windows: Agendador de Tarefas. Graças à idempotência, reexecutar não duplica registros nem reenvia alertas.

## 4. Quadro de mapeamento dos temas

| Tema Utilizado | Onde foi aplicado no código | Qual problema resolve no Cidade Para Todos |
|---|---|---|
| **Tema 1 - Consumo de API pública (Open-Meteo)** | `rpa_pipeline.py`: `coletar_clima()`, `_interpretar_open_meteo()`, `CircuitBreaker`, `ha_alerta_climatico()`; o resultado alimenta `pontuar_chamado()` | Traz contexto ambiental real em tempo de execução: chuva forte e rajadas elevam a prioridade de alagamentos, buracos, quedas de árvore e iluminação, e disparam alerta climático para gestores. |
| **Tema 2 - Processamento, sanitização e validação com Pandas + SQL/CSV** | `carregar_chamados()`, `validar_chamados()`, `anonimizar_texto()`, `priorizar()`, `persistir()` (SQLite), `gravar_relatorios()` | Transforma chamados sujos em dados confiáveis: isola registros corrompidos em quarentena, mascara dados pessoais (LGPD), classifica por criticidade explicável e persiste sem duplicar (idempotência). |
| **Tema 3 - Automação de notificações e relatórios (SMTP/Webhook simulado)** | `montar_alerta_emergencia()`, `montar_resumo_zeladoria()`, `enviar_email()`, `gravar_relatorios()` | Entrega a informação certa à pessoa certa sem ação manual: alerta de emergência em HTML acessível (alto contraste, tabela semântica, criticidade em texto) para a equipe de campo e resumo operacional para a zeladoria. |

**Combinação:** a API (T1) muda o *peso* da decisão tomada sobre os dados limpos (T2), e o resultado vira ação concreta para pessoas (T3). Sem qualquer um dos três, o fluxo deixa de fechar.

## 5. Resiliência e segurança (resumo)

| Falha prevista | Comportamento do robô | Log |
|---|---|---|
| API fora do ar / timeout / JSON inválido | retry com espera 1s→2s→4s, circuit breaker, cache local; sem cache, clima neutro + revisão humana | `FALHA_API`, `FALLBACK_CACHE`, `FALLBACK_NEUTRO` |
| CSV ausente, vazio, sem colunas ou com encoding errado | segue com 0 chamados e gera relatório | `ARQUIVO_AUSENTE`, `SCHEMA_INVALIDO` |
| Datas/coordenadas/texto/categoria/ID inválidos, duplicados, linhas malformadas | quarentena com motivo (`outputs/quarentena_chamados.csv`) | `QUARENTENA` |
| SMTP indisponível ou sem credenciais | retry; mensagem salva em `outputs/outbox/` | `FALLBACK_OUTBOX`, `SMTP_NAO_CONFIGURADO` |
| Erro de banco | segue só com CSV | `FALHA_BANCO` |
| Erro inesperado | barreira final por registro e por etapa; `main()` retorna 1 com `CRITICAL` | `FALHA_NAO_TRATADA` |

**Revisão humana:** chamados com pontuação na fronteira entre faixas ou sem dado climático confiável recebem `revisao_humana=True` no relatório.

## 6. Evidências de validação (execução real, amostra sintética n = 84 linhas)

| Métrica | Resultado |
|---|---|
| Chamados válidos processados | **60/60 (100%)** |
| Registros corrompidos isolados | **24/24 (23 em quarentena por regra + 1 linha malformada descartada na leitura)** |
| Alertas críticos disparados | 8 |
| Acurácia da criticidade vs. rótulo manual | **43/60 = 71,7%** |
| Erros por subestimação (gravidade menor que a esperada) | 14 dos 17 erros |
| Erros já sinalizados para revisão humana | 3 de 17 |
| Fallback com API fora e cache | fonte = `cache`, 60/60 processados |
| Fallback com API fora e sem cache | fonte = `indisponivel`, 60/60 processados, 43 itens enviados à revisão humana |
| Testes automatizados | 15 aprovados |

**Limites honestos:** (i) a amostra é pequena e sintética (25 modelos de texto repetidos), então 71,7% não generaliza para chamados reais; (ii) os rótulos foram definidos por um único avaliador; (iii) a maior fonte de erro é a **subestimação** (ex.: "árvore caída sobre carro, pessoa ferida" e "fio exposto em poste caído" ficam com 70 pontos, abaixo do limiar CRÍTICA de 75, em parte pelo teto do léxico; descrições sem palavras-chave, como "buraco grande… carros desviando", recebem BAIXA). Por isso o sistema é de **apoio à decisão**: a triagem final de casos de fronteira é humana. Melhorias previstas: recalibrar pesos, limiares e teto do léxico, treinar um classificador TF-IDF com chamados reais anonimizados e ampliar o conjunto de teste.

## 7. Proteção de dados, ética e ODS

Somente dados **sintéticos** (CSV gerado por `generate_test_data.py`) e **públicos** (Open-Meteo) são usados; os dados pessoais fictícios inseridos nos testes (e-mail, telefone, CPF) são **mascarados antes** de qualquer persistência, relatório ou e-mail (LGPD, Lei 13.709/2018 — minimização), e nada é enviado a APIs externas além de coordenadas da cidade. Chaves ficam em `.env`. O sistema não decide sozinho nos casos incertos (revisão humana) e explica cada pontuação (`motivos`). **ODS 11** (Cidades e Comunidades Sustentáveis): prioriza acessibilidade e segurança urbana; **ODS 9** (Indústria, Inovação e Infraestrutura): automatiza a gestão de manutenção da infraestrutura urbana.

## 8. Estrutura do repositório

```
rpa_pipeline.py            # robô principal (main())
generate_test_data.py      # dataset sintético + cenários de falha
test_rpa_resilience.py     # pytest + demonstração + relatório de evidências
chamados_cidadao_test.csv  # gerado pelo script acima
cache/clima_cache.json     # cache inicial para fallback offline
evidencias/relatorio_validacao.json
.env.example · requirements.txt · .gitignore
```
Link Colab [Colab_do_Projeto_Externo](https://colab.research.google.com/drive/1N5PUWUXEaOVwtbW8aYv5n0zAoH7W_iHF?usp=sharing)
