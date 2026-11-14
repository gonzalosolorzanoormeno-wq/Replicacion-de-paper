from pathlib import Path
import math

import pandas as pd
from sklearn.metrics import average_precision_score


PANEL_PATH = Path(
    "../outputs/panel_2025-10-01_PROVISIONAL.csv"
)

SALIDA = Path(
    "../outputs/fusion_transparente_octubre2025.csv"
)

TARGET = "y_kev_90d"
BUDGET = 0.05


def recall_5(df, score_col):

    k = math.ceil(
        BUDGET * len(df)
    )

    ranking = (
        df.sort_values(
            score_col,
            ascending=False,
            kind="mergesort"
        )
        .head(k)
    )

    total_positivos = int(
        df[TARGET].sum()
    )

    capturados = int(
        ranking[TARGET].sum()
    )

    recall = (
        capturados / total_positivos
        if total_positivos > 0
        else 0
    )

    return recall, capturados, k


def weighted_recall_5(
    df,
    score_col
):

    k = math.ceil(
        BUDGET * len(df)
    )

    ranking = (
        df.sort_values(
            score_col,
            ascending=False,
            kind="mergesort"
        )
        .head(k)
    )

    # Solo los positivos contribuyen al peso.
    # El paper usa CVSS base como ponderación
    # de severidad de los futuros KEV.
    total_weight = (
        df.loc[
            df[TARGET] == 1,
            "cvss_base"
        ].sum()
    )

    captured_weight = (
        ranking.loc[
            ranking[TARGET] == 1,
            "cvss_base"
        ].sum()
    )

    return (
        captured_weight / total_weight
        if total_weight > 0
        else 0
    )


def main():

    df = pd.read_csv(
        PANEL_PATH
    )

    print("=" * 80)
    print(
        "FUSIÓN TRANSPARENTE — "
        "VALIDACIÓN OCTUBRE 2025"
    )
    print("=" * 80)

    print(
        f"\nPanel: {len(df):,}"
    )

    print(
        f"Positivos 90d: "
        f"{int(df[TARGET].sum())}"
    )

    # ---------------------------------
    # Percentile ranks dentro del panel
    # ---------------------------------

    df["rank_epss"] = (
        df["epss"]
        .rank(
            method="average",
            pct=True
        )
    )

    df["rank_cvss"] = (
        df["cvss_base"]
        .rank(
            method="average",
            pct=True
        )
    )

    resultados = []

    # α = 0, 0.05, ..., 1
    alphas = [
        i / 20
        for i in range(21)
    ]

    print(
        "\n"
        "alpha | Weighted R@5% | "
        "R@5% | AP"
    )

    print("-" * 80)

    for alpha in alphas:

        df["fusion"] = (
            alpha
            * df["rank_epss"]
            +
            (1 - alpha)
            * df["rank_cvss"]
        )

        recall, hits, k = recall_5(
            df,
            "fusion"
        )

        weighted_recall = (
            weighted_recall_5(
                df,
                "fusion"
            )
        )

        ap = average_precision_score(
            df[TARGET],
            df["fusion"]
        )

        resultados.append(
            {
                "alpha": alpha,
                "weighted_r5": (
                    weighted_recall
                ),
                "r5": recall,
                "ap": ap,
                "hits": hits,
                "k": k,
            }
        )

        print(
            f"{alpha:4.2f}  | "
            f"{weighted_recall:8.3%}      | "
            f"{recall:6.2%} | "
            f"{ap:.6f}"
        )

    resultados_df = pd.DataFrame(
        resultados
    )

    resultados_df.to_csv(
        SALIDA,
        index=False
    )

    # ---------------------------------
    # Jerarquía de selección del paper
    # ---------------------------------

    max_weighted = (
        resultados_df[
            "weighted_r5"
        ].max()
    )

    candidatos = (
        resultados_df[
            resultados_df[
                "weighted_r5"
            ].round(12)
            == round(
                max_weighted,
                12
            )
        ]
    )

    max_recall = (
        candidatos["r5"].max()
    )

    candidatos = (
        candidatos[
            candidatos["r5"]
            .round(12)
            == round(
                max_recall,
                12
            )
        ]
    )

    seleccionado = (
        candidatos
        .sort_values(
            "ap",
            ascending=False
        )
        .iloc[0]
    )

    print("\n" + "=" * 80)

    print(
        "MEJOR REGIÓN SEGÚN "
        "LA JERARQUÍA DEL PAPER"
    )

    print("=" * 80)

    print(
        "\nCandidatos empatados "
        "en Weighted R@5% "
        "y Recall@5%:"
    )

    print(
        candidatos[
            [
                "alpha",
                "weighted_r5",
                "r5",
                "ap",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nALPHA SELECCIONADO:"
    )

    print(
        f"α = "
        f"{seleccionado['alpha']:.2f}"
    )

    print(
        f"Weighted R@5% = "
        f"{seleccionado['weighted_r5']:.3%}"
    )

    print(f"Recall@5% = "f"{seleccionado['r5']:.2%}")
    print(f"AP = "f"{seleccionado['ap']:.6f}")
    print("\nArchivo generado:")
    print(SALIDA)
if __name__ == "__main__":
    main()