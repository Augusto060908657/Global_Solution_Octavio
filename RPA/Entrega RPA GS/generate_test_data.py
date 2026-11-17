#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_test_data.py - Gera o dataset SINTÉTICO de teste (chamados_cidadao_test.csv)

Autor: Octavio Augusto Ramalho Brandão Pires - RM 555696

O arquivo contém:
  * 60 chamados VÁLIDOS (buracos, semáforos, rampas, alagamentos, iluminação,
    árvores, calçadas), cada um com `rotulo_esperado` atribuído À MÃO por
    template (CRITICA / ALTA / MEDIA / BAIXA) -> conjunto de teste rotulado.
  * 23 registros CORROMPIDOS propositalmente (rotulo_esperado = INVALIDO):
    datas inválidas/futuras, coordenadas ausentes/absurdas, textos nulos,
    categoria desconhecida, ID ausente e IDs duplicados.
  * 1 linha MALFORMADA (número errado de colunas), também esperada como erro.
  * 6 descrições com dados pessoais FALSOS (e-mail, telefone, CPF fictícios)
    para provar a anonimização LGPD.

Todos os dados são sintéticos: nenhum dado pessoal real é usado.
Execução: python generate_test_data.py [--saida arquivo.csv] [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import io
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

CAMPOS = ["id_chamado", "data_registro", "categoria", "descricao",
          "latitude", "longitude", "bairro", "rotulo_esperado"]

BAIRROS = ["Pinheiros", "Mooca", "Liberdade", "Santana", "Itaquera",
           "Vila Mariana", "Butantã", "Santo Amaro", "Penha", "Lapa"]

# (categoria, descrição, rótulo esperado) - rótulo definido por critério humano,
# NÃO calculado pelo robô. Alguns templates são difíceis de propósito.
TEMPLATES: List[Tuple[str, str, str]] = [
    ("buraco", "Buraco grande na faixa da direita, carros desviando bruscamente", "MEDIA"),
    ("buraco", "Cratera aberta na avenida, ônibus não consegue passar, perigo", "ALTA"),
    ("buraco", "Buraco pequeno na calçada em frente ao mercado", "BAIXA"),
    ("buraco", "Buraco profundo próximo a escola, crianças atravessam, risco de acidente", "CRITICA"),
    ("buraco", "Buraco no asfalto", "MEDIA"),
    ("semaforo_quebrado", "Semáforo apagado no cruzamento movimentado, quase acidente ontem", "ALTA"),
    ("semaforo_quebrado", "Semáforo apagado em frente ao hospital, ambulâncias cruzando, urgente", "CRITICA"),
    ("semaforo_quebrado", "Semáforo piscando amarelo há dois dias", "MEDIA"),
    ("semaforo_quebrado", "Tempo do semáforo de pedestres curto demais para idosos", "MEDIA"),
    ("rampa_obstruida", "Rampa de acessibilidade bloqueada por entulho, cadeirante precisa descer na rua", "ALTA"),
    ("rampa_obstruida", "Rampa obstruída por carro estacionado", "MEDIA"),
    ("rampa_obstruida", "Rampa em frente à escola obstruída, aluno cadeirante não consegue entrar", "CRITICA"),
    ("rampa_obstruida", "Rampa com pintura desgastada", "BAIXA"),
    ("alagamento", "Rua alagada, água na altura da cintura, moradores ilhados, urgente", "CRITICA"),
    ("alagamento", "Ponto de alagamento recorrente com bueiro entupido", "MEDIA"),
    ("alagamento", "Poça grande após chuva na esquina", "BAIXA"),
    ("iluminacao_publica", "Poste apagado na rua, local escuro, idosos com medo de passar", "MEDIA"),
    ("iluminacao_publica", "Fio exposto em poste caído, risco de choque", "CRITICA"),
    ("iluminacao_publica", "Lâmpada queimada em praça", "BAIXA"),
    ("queda_arvore", "Árvore caída bloqueando a rua inteira, ônibus impedidos", "ALTA"),
    ("queda_arvore", "Árvore caída sobre carro, pessoa ferida", "CRITICA"),
    ("queda_arvore", "Galho grande quase caindo sobre a calçada", "MEDIA"),
    ("calcada_danificada", "Calçada quebrada, idosa tropeçou", "MEDIA"),
    ("calcada_danificada", "Calçada com piso solto, cadeirante não consegue passar", "ALTA"),
    ("calcada_danificada", "Pequena rachadura na calçada", "BAIXA"),
]

# Dados pessoais FALSOS, usados só para testar o mascaramento
PII_FALSA = [
    " Contato: maria.teste@exemplo.com",
    " Ligar para (11) 91234-5678",
    " CPF do solicitante 123.456.789-09",
    " Falar com joao.ficticio@exemplo.org",
    " Tel 11 98765-4321",
    " Documento 987.654.321-00",
]


def _coordenadas(rng: random.Random) -> Tuple[str, str]:
    """Coordenadas aleatórias dentro do município de São Paulo."""
    return f"{rng.uniform(-23.75, -23.45):.6f}", f"{rng.uniform(-46.78, -46.40):.6f}"


def _data_valida(rng: random.Random, agora: datetime) -> str:
    dt = agora - timedelta(days=rng.randint(1, 60), hours=rng.randint(0, 23), minutes=rng.randint(0, 59))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def gerar_linhas(seed: int = 42) -> Tuple[List[Dict[str, str]], List[str]]:
    """Devolve (registros_normais, linhas_malformadas_brutas)."""
    rng = random.Random(seed)
    agora = datetime.now()
    contador = 0

    def novo_id() -> str:
        nonlocal contador
        contador += 1
        return f"CH-{contador:04d}"

    # ---------- 60 registros válidos (templates ciclados e embaralhados) ----------
    validos: List[Dict[str, str]] = []
    modelos = (TEMPLATES * 3)[:60]
    rng.shuffle(modelos)
    for categoria, descricao, rotulo in modelos:
        lat, lon = _coordenadas(rng)
        validos.append({"id_chamado": novo_id(), "data_registro": _data_valida(rng, agora),
                        "categoria": categoria, "descricao": descricao, "latitude": lat,
                        "longitude": lon, "bairro": rng.choice(BAIRROS), "rotulo_esperado": rotulo})
    # PII falsa em 6 descrições (não altera o rótulo esperado)
    for reg, pii in zip(rng.sample(validos, len(PII_FALSA)), PII_FALSA):
        reg["descricao"] += pii

    # ---------- 23 registros corrompidos ----------
    def base_ruim() -> Dict[str, str]:
        categoria, descricao, _ = rng.choice(TEMPLATES)
        lat, lon = _coordenadas(rng)
        return {"id_chamado": novo_id(), "data_registro": _data_valida(rng, agora), "categoria": categoria,
                "descricao": descricao, "latitude": lat, "longitude": lon,
                "bairro": rng.choice(BAIRROS), "rotulo_esperado": "INVALIDO"}

    corrompidos: List[Dict[str, str]] = []

    def corromper(**campos: str) -> None:
        r = base_ruim()
        r.update(campos)
        corrompidos.append(r)

    for d in ("32/13/2026 10:00", "31/02/2026", "2026-99-99 25:61:00"):   # datas impossíveis
        corromper(data_registro=d)
    for d in ("ontem", "semana passada"):                                  # datas em texto livre
        corromper(data_registro=d)
    for _ in range(2):                                                     # data vazia
        corromper(data_registro="")
    corromper(data_registro=(agora + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"))  # futura
    corromper(latitude="")                                                 # coordenadas ausentes
    corromper(latitude="")
    corromper(longitude="")
    corromper(longitude="")
    corromper(latitude="abc")                                              # não numérica
    corromper(latitude="999.0")                                            # fora do intervalo
    corromper(latitude="-22.9068", longitude="-43.1729")                   # fora de São Paulo (Rio)
    corromper(descricao="")                                                # texto nulo
    corromper(descricao="NULL")
    corromper(descricao="   ")                                             # só espaços
    corromper(descricao="ok")                                              # curto demais
    corromper(categoria="ovni_no_ceu")                                     # categoria desconhecida
    corromper(id_chamado="")                                               # ID ausente

    # IDs duplicados: cópias de registros válidos, sempre ANEXADAS DEPOIS do original
    # (a 2ª ocorrência é a inválida, como o robô as interpreta na leitura sequencial)
    duplicados: List[Dict[str, str]] = []
    for original in rng.sample(validos, 2):
        dup = dict(original)
        dup["rotulo_esperado"] = "INVALIDO"
        duplicados.append(dup)

    # ---------- 1 linha malformada (colunas a mais) ----------
    malformadas = ["CH-9999,2026-01-01 10:00:00,buraco,linha com colunas demais,-23.5,-46.6,Lapa,MEDIA,campo_extra,outro_extra"]

    todos = validos + corrompidos
    rng.shuffle(todos)
    return todos + duplicados, malformadas


def gerar_csv(caminho: str = "chamados_cidadao_test.csv", seed: int = 42) -> Dict[str, int]:
    """Grava o CSV e devolve contagens esperadas (usadas pelos testes)."""
    registros, malformadas = gerar_linhas(seed)
    buffer = io.StringIO()
    escritor = csv.DictWriter(buffer, fieldnames=CAMPOS, lineterminator="\n")
    escritor.writeheader()
    escritor.writerows(registros)
    linhas = buffer.getvalue().splitlines()
    linhas.insert(len(linhas) // 2, malformadas[0])          # malformada no meio do arquivo
    with open(caminho, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(linhas) + "\n")

    vistos, esperados_validos, esperados_invalidos = set(), 0, 0
    for r in registros:
        if r["rotulo_esperado"] != "INVALIDO" and r["id_chamado"] not in vistos:
            esperados_validos += 1
        else:
            esperados_invalidos += 1
        vistos.add(r["id_chamado"])
    return {"total_linhas_dados": len(registros) + len(malformadas),
            "validos_esperados": esperados_validos,
            "invalidos_esperados": esperados_invalidos + len(malformadas)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Gera chamados_cidadao_test.csv (dados sintéticos)")
    ap.add_argument("--saida", default="chamados_cidadao_test.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    info = gerar_csv(args.saida, args.seed)
    print(f"Arquivo '{args.saida}' gerado: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
