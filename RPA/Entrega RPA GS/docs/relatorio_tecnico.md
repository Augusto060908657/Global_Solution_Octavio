# RELATÓRIO TÉCNICO — AI FOR ROBOTIC PROCESS AUTOMATION (RPA + IA)

**Projeto:** Cidade Para Todos — CitizenTech com RPA e IA
**Global Solution FIAP 2026.2 · Tecnólogo em Inteligência Artificial**
**Autor:** Octavio Augusto Ramalho Brandão Pires — RM 555696

---

## 1 DESCRIÇÃO DA SOLUÇÃO E ARQUITETURA DO ROBÔ

O problema tratado é a triagem lenta e inconsistente de chamados urbanos de zeladoria (buracos, semáforos quebrados, rampas de acessibilidade obstruídas, alagamentos, iluminação, quedas de árvore e calçadas danificadas). Ao chegarem em volume e com dados de baixa qualidade, os casos que colocam em risco pessoas vulneráveis, em especial pessoas com deficiência, idosos e crianças, competem por atenção com ocorrências de baixo impacto. O público beneficiado é duplo: o cidadão, que tem a ocorrência atendida conforme sua gravidade, e os gestores e equipes de campo, que recebem informação já organizada.

O robô foi implementado em Python (`rpa_pipeline.py`) e executa, a partir da função `main()`, um fluxo contínuo de seis etapas. Na etapa de entrada, o robô consulta a API pública Open-Meteo para obter precipitação, rajadas de vento e temperatura (Tema 1) e lê um arquivo CSV de chamados de cidadãos. Na etapa de processamento, a biblioteca Pandas e rotinas de validação verificam o esquema e as regras de negócio de cada registro, mascaram dados pessoais e isolam os registros inválidos em uma quarentena com o motivo da rejeição (Tema 2). Em seguida, uma regra inteligente de priorização combina três fontes de evidência: o peso da categoria da ocorrência, um léxico de processamento de linguagem natural por radicais, com termos de risco à vida, vulnerabilidade e acessibilidade, urgência e atenuantes, e o fator climático. O resultado é uma pontuação de 0 a 100 mapeada em quatro faixas (CRÍTICA, ALTA, MÉDIA e BAIXA), acompanhada da lista de motivos que a justificam, o que torna a decisão explicável. Os chamados são então persistidos em banco SQLite, e somente os novos seguem adiante.

Na etapa de saída (Tema 3), o robô gera e envia dois e-mails: um alerta de emergência, destinado às equipes de campo e aos gestores, e um resumo operacional de zeladoria. Ambos têm versão em texto simples e em HTML acessível, com idioma declarado, tabelas semânticas, alto contraste e a criticidade expressa por texto, e não apenas por cor. Ao final, são gravados o relatório de execução (JSON e Markdown), o CSV priorizado e o CSV de quarentena. A inteligência do robô é baseada em regras, sem uso de LLM, o que atende ao enunciado, preserva a explicabilidade e elimina o envio de texto de cidadãos a serviços de terceiros. Os três temas têm função real e interdependente: a API altera o peso da decisão, o tratamento de dados garante a confiabilidade do que é decidido, e a automação de notificações converte a decisão em ação para pessoas.

## 2 ANÁLISE DE RESILIÊNCIA E TRATATIVAS DE ERRO

A premissa de projeto é que o robô nunca interrompe a execução diante de falhas previsíveis. Para isso foram adotadas barreiras em camadas.

**Consumo da API.** A chamada ao Open-Meteo usa tempo limite (*timeout*) de 5 segundos e cláusula `try/except` abrangente (erros de rede, de HTTP, de JSON e de campos ausentes ou não numéricos). Em caso de falha, aplica-se *retry* com espera exponencial (1 s, 2 s, 4 s). Um *circuit breaker* conta as falhas consecutivas e, ao atingir o limite configurado (três), abre o circuito e suprime novas chamadas durante um período de resfriamento, evitando insistir em um serviço indisponível. Esgotadas as tentativas, aciona-se o caminho seguro em dois níveis: primeiro, o último dado válido gravado em cache local, registrando sua idade; na ausência de cache, o robô segue com clima neutro, desconsidera o fator climático e marca os itens sensíveis ao clima para revisão humana. O cache é gravado de forma atômica (arquivo temporário seguido de substituição).

**Entrada de dados.** A leitura do CSV trata arquivo ausente, arquivo vazio, esquema sem colunas obrigatórias, codificação incorreta (UTF-8 com alternativa latin-1) e linhas malformadas, que são descartadas e contabilizadas. Cada registro é validado individualmente (identificador ausente ou duplicado, data inválida ou futura, coordenadas ausentes, não numéricas ou fora do município, texto nulo ou curto, categoria desconhecida) e, em caso de falha, é desviado para a quarentena sem afetar os demais. Uma barreira final captura exceções inesperadas por registro.

**Persistência e idempotência.** A tabela SQLite usa o identificador do chamado como chave primária com `INSERT OR IGNORE`; reexecuções não duplicam registros nem reenviam alertas. Falhas de banco são registradas e o robô prossegue apenas com saída em CSV.

**Notificações.** O envio via SMTP usa STARTTLS ou SSL, tempo limite e *retry* com espera. Se o servidor estiver indisponível ou as credenciais ausentes, a mensagem é preservada em `outputs/outbox/` como arquivo `.eml`, para reenvio, e a execução continua.

**Logs e revisão humana.** O registro estruturado ocorre em arquivo (`rpa_execution.log`) e no console, nos níveis INFO, WARNING, ERROR e CRITICAL, com *timestamp*, etapa e status (por exemplo, `[CLIMA] status=FALLBACK_CACHE`). O contador de níveis alimenta o relatório de execução. Itens com pontuação na fronteira entre faixas, ou sem dado climático confiável, recebem a marcação `revisao_humana`, delimitando o que o robô não resolve com segurança. A função `main()` retorna código 1 e registra CRITICAL apenas diante de falha fatal não prevista.

## 3 EVIDÊNCIAS MENSURÁVEIS DE VALIDAÇÃO

A validação foi feita sobre um conjunto sintético de 84 linhas (`chamados_cidadao_test.csv`), gerado por semente fixa: 60 registros válidos, com rótulo de criticidade atribuído manualmente por modelo de texto; 23 registros corrompidos de propósito (8 com data inválida ou futura, 7 com coordenadas ausentes ou inválidas, 4 com texto nulo ou curto, 2 com identificador duplicado, 1 com categoria desconhecida e 1 sem identificador); e 1 linha com número de colunas incorreto. Foram incluídos ainda 6 registros com dados pessoais fictícios (e-mail, telefone, CPF) para testar o mascaramento. A bateria de 15 testes automatizados (`pytest`) foi integralmente aprovada, e a API foi simulada por dublês para tornar os testes determinísticos.

**Tabela 1 — Resultados da execução sobre a amostra (n = 84)**

| Indicador | Resultado |
|---|---|
| Chamados válidos processados | 60 de 60 (100%) |
| Registros corrompidos isolados (quarentena + linha malformada) | 24 de 24 (100%) |
| Alertas disparados | 8 (todos críticos; 0 climáticos com tempo seco) |
| Dados pessoais mascarados | 6 de 6 |
| Falha de API com cache disponível | fallback acionado, 60 processados, 5 avisos e 0 erros registrados |
| Falha de API sem cache | fallback neutro, 60 processados, 43 itens enviados à revisão humana, 1 ERROR registrado |
| Reexecução sobre o mesmo banco | 0 novos registros, 60 ignorados, 0 alertas reenviados |
| Falha de SMTP simulada | 2 mensagens preservadas em outbox, execução concluída |
| Acurácia da criticidade frente ao rótulo manual | 43 de 60 (71,7%) |

A distribuição prevista foi de 8 chamados CRÍTICOS, 20 ALTOS, 15 MÉDIOS e 17 BAIXOS. **Análise dos erros:** dos 17 erros de classificação, 14 foram subestimações de gravidade (o robô classificou abaixo do rótulo manual), e apenas 3 haviam sido sinalizados para revisão humana. As causas identificadas foram: (a) casos graves que ficaram com 70 pontos, logo abaixo do limiar CRÍTICA (75), como "árvore caída sobre carro, pessoa ferida", "rua alagada… moradores ilhados" e "fio exposto em poste caído", este último também afetado pelo teto de 45 pontos do léxico (6 erros em alagamento, queda de árvore e iluminação); (b) descrições sem palavras-chave reconhecidas, como "buraco grande… carros desviando", permanecem na faixa BAIXA por dependerem só do peso da categoria (6 erros); e (c) o rótulo manual considerou "médio" casos que o léxico elevou a ALTA, como o semáforo de pedestres curto para idosos (3 erros), e "alta" para calçada com cadeirante, que a regra manteve em MÉDIA (2 erros).

**Limites da solução.** A amostra é pequena e sintética, composta por 25 modelos de texto repetidos, e os rótulos foram definidos por um único avaliador, de modo que a acurácia obtida não deve ser extrapolada para chamados reais. O léxico é restrito ao vocabulário previsto e não reconhece sinônimos nem erros de grafia. Em dados reais, recomenda-se treinar um classificador supervisionado (por exemplo, TF-IDF com regressão logística) sobre chamados anonimizados e rotulados por mais de um especialista, recalibrar pesos, limiares e o teto do léxico e adotar uma regra conservadora que eleve à revisão humana os casos de categoria de alto risco sem termos reconhecidos. Até lá, o robô deve ser entendido como apoio à triagem, e não como substituto da decisão humana.

## 4 PROTEÇÃO DE DADOS (LGPD), ÉTICA E ODS

**Proteção de dados.** O protótipo utiliza exclusivamente dados sintéticos (gerados por script) e dados públicos (API Open-Meteo, que não exige autenticação nem recebe qualquer informação pessoal, apenas as coordenadas do município). Em conformidade com o princípio da minimização da Lei Geral de Proteção de Dados (Lei nº 13.709/2018), CPF, e-mail e telefone eventualmente presentes no texto livre são mascarados antes de qualquer persistência, relatório ou notificação, o que foi comprovado em teste automatizado: nenhum dado pessoal aparece nas saídas. Credenciais de SMTP e demais parâmetros são lidos de variáveis de ambiente (`python-dotenv`), e o arquivo `.env` é excluído do repositório. Nenhum dado de cidadão é enviado a serviços de IA de terceiros.

**Ética.** As decisões de priorização são explicáveis, pois cada pontuação registra os critérios que a compuseram, e transparentes quanto aos limites, já que casos incertos são encaminhados à revisão humana. O principal risco ético identificado é a subestimação da gravidade, que pode atrasar o atendimento de pessoas vulneráveis; ele é mitigado pela revisão humana, pela medição explícita desse tipo de erro e pelo uso do robô como apoio, e não como decisor único. Há também o risco de viés geográfico, caso regiões com mais registros sejam sistematicamente priorizadas, o que deve ser monitorado em uso real. A acessibilidade das notificações (alto contraste, estrutura semântica, informação não dependente de cor) busca incluir gestores e agentes com deficiência visual.

**ODS 11 — Cidades e Comunidades Sustentáveis.** O robô contribui para tornar as cidades mais seguras, inclusivas e resilientes ao dar prioridade à acessibilidade (rampas e calçadas), à segurança viária (semáforos, buracos) e à resposta a eventos climáticos (alagamentos e quedas de árvore), ao reduzir o tempo entre o registro do problema e a ação da equipe de campo. **ODS 9 — Indústria, Inovação e Infraestrutura.** A solução aplica automação e inteligência artificial de baixo custo, com ferramentas gratuitas, à manutenção da infraestrutura urbana, apoiando uma gestão preventiva e baseada em dados.

---

## REFERÊNCIAS

BRASIL. **Lei nº 13.709, de 14 de agosto de 2018.** Lei Geral de Proteção de Dados Pessoais (LGPD). Brasília, DF: Presidência da República, 2018.

ORGANIZAÇÃO DAS NAÇÕES UNIDAS. **Objetivos de Desenvolvimento Sustentável:** ODS 9 e ODS 11. Nova York: ONU, 2015.

OPEN-METEO. **Open-Meteo Weather Forecast API.** Disponível em: https://open-meteo.com/en/docs. Acesso em: 6 nov. 2026.
