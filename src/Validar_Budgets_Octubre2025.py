from pathlib import Path
import math

import pandas as pd


PANEL_PATH = Path(
    "../outputs/panel_2025-10-01_PROVISIONAL.csv"
)

BUDGETS = [
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
]


def evaluar_budget(
    df,
    score_col,
    target_col,
    budget
):
    n = len(df)

    k = math.ceil(
        budget * n
    )

    # Orden estable, como en las validaciones anteriores
    ranking = (
        df.sort_values(
            score_col,
            ascending=False,
            kind="mergesort"
        )
        .reset_index(drop=True)
    )

    seleccionados = ranking.head(k)

    positivos_totales = int(
        df[target_col].sum()
    )

    hits = int(
        seleccionados[target_col].sum()
    )

    recall = (
        hits / positivos_totales
        if positivos_totales > 0
        else 0
    )

    precision = (
        hits / k
        if k > 0
        else 0
    )

    # -------------------------
    # Empate en la frontera
    # -------------------------

    score_corte = ranking.iloc[k - 1][
        score_col
    ]

    mejores = ranking[
        ranking[score_col] > score_corte
    ]

    empatados = ranking[
        ranking[score_col] == score_corte
    ]

    tie_size = len(
        empatados
    )

    slots = (
        k
        - len(mejores)
    )

    return {
        "ranking": score_col,
        "budget": budget,
        "k": k,
        "hits": hits,
        "positivos": positivos_totales,
        "recall": recall,
        "precision": precision,
        "score_corte": score_corte,
        "tie_size": tie_size,
        "slots": slots,
    }


def main():

    panel = pd.read_csv(
        PANEL_PATH
    )

    target = "y_kev_90d"

    print("=" * 82)
    print(
        "VALIDACIÓN BUDGET-AWARE — "
        "OCTUBRE 2025"
    )
    print("=" * 82)

    print(
        f"\nCVEs elegibles: "
        f"{len(panel):,}"
    )

    print(
        f"Positivos 90d: "
        f"{int(panel[target].sum())}"
    )

    resultados = []

    for score in [
        "epss",
        "cvss_base"
    ]:

        print("\n" + "=" * 82)
        print(
            f"RANKING: {score.upper()}"
        )
        print("=" * 82)

        for budget in BUDGETS:

            r = evaluar_budget(
                panel,
                score,
                target,
                budget
            )

            resultados.append(r)

            print(
                f"\nBudget: "
                f"{budget * 100:.1f}%"
            )

            print(
                f"k: "
                f"{r['k']:,}"
            )

            print(
                f"Positivos capturados: "
                f"{r['hits']}/"
                f"{r['positivos']}"
            )

            print(
                f"Recall: "
                f"{r['recall']:.2%}"
            )

            print(
                f"Precision: "
                f"{r['precision']:.5%}"
            )

            print(
                f"Score de corte: "
                f"{r['score_corte']}"
            )

            print(
                f"Tamaño empate frontera: "
                f"{r['tie_size']:,}"
            )

            print(
                f"Slots usados dentro "
                f"del empate: "
                f"{r['slots']:,}"
            )

    # -----------------------------------
    # Tabla final
    # -----------------------------------

    resultados_df = pd.DataFrame(
        resultados
    )

    resultados_df[
        "budget_pct"
    ] = (
        resultados_df["budget"]
        * 100
    )

    resultados_df[
        "recall_pct"
    ] = (
        resultados_df["recall"]
        * 100
    )

    resultados_df[
        "precision_pct"
    ] = (
        resultados_df["precision"]
        * 100
    )

    print("\n" + "=" * 82)
    print("RESUMEN")
    print("=" * 82)

    print(
        resultados_df[
            [
                "ranking",
                "budget_pct",
                "k",
                "hits",
                "recall_pct",
                "precision_pct",
                "score_corte",
                "tie_size",
                "slots",
            ]
        ].to_string(
            index=False
        )
    )

    salida = Path(
        "../outputs/"
        "budgets_octubre2025.csv"
    )

    resultados_df.to_csv(
        salida,
        index=False
    )

    print(
        "\nArchivo generado:"
    )

    print(
        salida
    )


if __name__ == "__main__":
    main()