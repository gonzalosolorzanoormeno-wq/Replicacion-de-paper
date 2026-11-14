from pathlib import Path
import math
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

PANEL_PATH = Path(
    "../outputs/panel_2025-10-01_PROVISIONAL.csv"
)

panel = pd.read_csv(PANEL_PATH)

print("==============================")
print("PANEL")
print("==============================")

print("CVEs:", f"{len(panel):,}")
print("Positivos:", panel["y_kev_90d"].sum())

y = panel["y_kev_90d"]

# ============================================
# FUNCIÓN PARA EVALUAR UN RANKING
# ============================================

def evaluar_ranking(df, columna, nombre):
    scores = df[columna]

    ap = average_precision_score(
        df["y_kev_90d"],
        scores
    )

    auc = roc_auc_score(
        df["y_kev_90d"],
        scores
    )

    ranking = df.sort_values(
        columna,
        ascending=False,
        kind="mergesort"
    ).reset_index(drop=True)

    n_top5 = math.ceil(
        len(ranking) * 0.05
    )

    seleccion = ranking.iloc[:n_top5]

    positivos_totales = int(
        ranking["y_kev_90d"].sum()
    )

    positivos_capturados = int(
        seleccion["y_kev_90d"].sum()
    )

    recall_5 = (
        positivos_capturados
        / positivos_totales
        if positivos_totales > 0
        else 0
    )

    precision_5 = (
        positivos_capturados
        / len(seleccion)
    )

    print(f"\n===== {nombre} =====")

    print(
        "Average Precision:",
        f"{ap:.6f}"
    )

    print(
        "ROC-AUC:",
        f"{auc:.6f}"
    )

    print(
        "Top 5%:",
        f"{len(seleccion):,}",
        "CVEs"
    )

    print(
        "Positivos capturados:",
        f"{positivos_capturados}/{positivos_totales}"
    )

    print(
        "Recall@5%:",
        f"{recall_5 * 100:.2f}%"
    )

    print(
        "Precision@5%:",
        f"{precision_5 * 100:.4f}%"
    )

    return ranking


# ============================================
# EPSS
# ============================================

ranking_epss = evaluar_ranking(
    panel,
    "epss",
    "EPSS"
)


# ============================================
# CVSS BASE
# ============================================

ranking_cvss = evaluar_ranking(
    panel,
    "cvss_base",
    "CVSS BASE"
)


# ============================================
# POSITIVOS
# ============================================

positivos = panel[
    panel["y_kev_90d"] == 1
].copy()

positivos = positivos[
    [
        "cve_id",
        "epss",
        "percentile",
        "cvss_base",
        "cvss_exploitability",
        "published"
    ]
]

positivos = positivos.sort_values(
    "epss",
    ascending=False
)

print("\n==============================")
print("12 POSITIVOS DE NUESTRO PANEL")
print("==============================")

print(
    positivos.to_string(
        index=False
    )
)

positivos.to_csv(
    "../outputs/positivos_90d_PROVISIONAL.csv",
    index=False
)