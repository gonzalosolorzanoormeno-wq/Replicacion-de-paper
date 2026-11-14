import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score


PANEL_PATH = Path(
    "../outputs/panel_2025-10-01_PROVISIONAL.csv"
)

panel = pd.read_csv(PANEL_PATH)

N = len(panel)
TOP5 = math.ceil(N * 0.05)

positivos = panel[
    panel["y_kev_90d"] == 1
].copy()


# ============================================
# RANKING EPSS
# ============================================

epss_rank = panel.sort_values(
    "epss",
    ascending=False,
    kind="mergesort"
).reset_index(drop=True)

epss_rank["rank_epss"] = (
    epss_rank.index + 1
)

top_epss_ids = set(
    epss_rank.iloc[:TOP5]["cve_id"]
)

rank_epss_dict = dict(
    zip(
        epss_rank["cve_id"],
        epss_rank["rank_epss"]
    )
)


# ============================================
# RANKING CVSS
# ============================================

cvss_rank = panel.sort_values(
    "cvss_base",
    ascending=False,
    kind="mergesort"
).reset_index(drop=True)

cvss_rank["rank_cvss"] = (
    cvss_rank.index + 1
)

top_cvss_ids = set(
    cvss_rank.iloc[:TOP5]["cve_id"]
)

rank_cvss_dict = dict(
    zip(
        cvss_rank["cve_id"],
        cvss_rank["rank_cvss"]
    )
)


# ============================================
# MARCAR POSITIVOS
# ============================================

positivos["rank_epss"] = positivos[
    "cve_id"
].map(rank_epss_dict)

positivos["top5_epss"] = positivos[
    "cve_id"
].isin(top_epss_ids)

positivos["rank_cvss"] = positivos[
    "cve_id"
].map(rank_cvss_dict)

positivos["top5_cvss"] = positivos[
    "cve_id"
].isin(top_cvss_ids)


columnas = [
    "cve_id",
    "epss",
    "cvss_base",
    "rank_epss",
    "top5_epss",
    "rank_cvss",
    "top5_cvss"
]

print("==============================")
print("POSITIVOS Y TOP 5%")
print("==============================")

print(
    positivos[columnas]
    .sort_values("rank_epss")
    .to_string(index=False)
)


# ============================================
# INTERSECCIÓN
# ============================================

ambos = positivos[
    positivos["top5_epss"]
    & positivos["top5_cvss"]
]

print("\n==============================")
print("POSITIVOS CAPTURADOS POR AMBOS")
print("==============================")

print(
    ambos[columnas].to_string(
        index=False
    )
)


# ============================================
# SIMULAR ELIMINACIÓN DE CADA POSITIVO
# ============================================

print("\n==============================")
print("SIMULACIÓN: ELIMINAR 1 POSITIVO")
print("==============================")

resultados = []

for cve in positivos["cve_id"]:

    prueba = panel[
        panel["cve_id"] != cve
    ].copy()

    positivos_totales = int(
        prueba["y_kev_90d"].sum()
    )

    top_n = math.ceil(
        len(prueba) * 0.05
    )

    # EPSS
    ranking_e = prueba.sort_values(
        "epss",
        ascending=False,
        kind="mergesort"
    )

    capturados_e = int(
        ranking_e
        .iloc[:top_n]["y_kev_90d"]
        .sum()
    )

    recall_e = (
        capturados_e
        / positivos_totales
    )

    ap_e = average_precision_score(
        prueba["y_kev_90d"],
        prueba["epss"]
    )

    # CVSS
    ranking_c = prueba.sort_values(
        "cvss_base",
        ascending=False,
        kind="mergesort"
    )

    capturados_c = int(
        ranking_c
        .iloc[:top_n]["y_kev_90d"]
        .sum()
    )

    recall_c = (
        capturados_c
        / positivos_totales
    )

    ap_c = average_precision_score(
        prueba["y_kev_90d"],
        prueba["cvss_base"]
    )

    resultados.append({
        "cve_eliminado": cve,
        "EPSS_R5": recall_e,
        "CVSS_R5": recall_c,
        "EPSS_AP": ap_e,
        "CVSS_AP": ap_c
    })


df_resultados = pd.DataFrame(
    resultados
)

# Queremos localizar casos que reproduzcan
# 6/11 y 1/11.
candidatos = df_resultados[
    (df_resultados["EPSS_R5"].round(6)
        == round(6 / 11, 6))
    &
    (df_resultados["CVSS_R5"].round(6)
        == round(1 / 11, 6))
].copy()

print("\nCandidatos que producen:")
print("EPSS Recall@5% = 6/11 = 54.55%")
print("CVSS Recall@5% = 1/11 = 9.09%")
print()

if len(candidatos) == 0:
    print("Ningún candidato.")
else:
    print(
        candidatos.to_string(
            index=False
        )
    )