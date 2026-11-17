"""Roda casos de teste reais contra a API e imprime a tabela em Markdown para o README (seção 3.8-b).

Uso (com API + Ollama no ar):  python scripts/run_test_cases.py
"""
import os
import sys

import requests

API = os.getenv("API_BASE_URL", "http://localhost:8000")
CASOS = [
    ("Poste apagado há 2 semanas na Rua das Acácias, rua muito escura à noite.", "iluminacao_publica / media"),
    ("Deixaram entulho de obra na calçada da escola e as crianças andam na rua.", "limpeza_urbana_entulho / alta"),
    ("Estou sentindo um cheiro forte de gás no corredor do prédio.", "qualquer categoria / critica (regra de segurança)"),
    ("Buraco enorme na avenida, os carros desviam e já houve acidente.", "buraco_pavimentacao / alta"),
    ("A UBS do meu bairro está sem médico há três dias e minha mãe precisa de consulta.", "saude / media ou alta"),
    ("Qual é a melhor pizzaria da cidade para o fim de semana?", "outros / baixa, confiança baixa"),
]


def main() -> int:
    print("| # | Entrada | Saída esperada | Saída obtida | Ajuste realizado |")
    print("|---|---------|----------------|--------------|------------------|")
    for i, (texto, esperado) in enumerate(CASOS, 1):
        try:
            r = requests.post(f"{API}/api/v1/analyze", json={"relato": texto}, timeout=180)
            if r.ok:
                a = r.json()["analise"]
                obtido = f"{a['categoria']} / {a['urgencia']} (conf. {a['confianca']:.0%})"
            else:
                obtido = f"erro {r.status_code}: {r.json().get('detail', {}).get('codigo')}"
        except requests.RequestException as exc:
            obtido = f"falha de conexão ({type(exc).__name__})"
        print(f"| {i} | {texto} | {esperado} | {obtido} | _preencher_ |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
