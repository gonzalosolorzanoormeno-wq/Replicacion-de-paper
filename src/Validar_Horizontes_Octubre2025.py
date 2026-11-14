import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from pathlib import Path


PANEL_PATH = Path("../outputs/panel_2025-10-01_PROVISIONAL.csv")
KEV_PATH = Path("../data/kev/known_exploited_vulnerabilities.csv")

FECHA_T = pd.Timestamp("2025-10-01")

HORIZONTES = [30, 90, 180]


def recall_at_5(df, score, target):
    k = int(np.ceil(0.05 * len(df)))

    ranking = (
        df.sort_values(
            score,
            ascending=False,
            kind="mergesort"
        )
        .head(k)
    )

    positivos = int(df[target].sum())

    capturados = int(
        ranking[target].sum()
    )

    recall = (
        capturados / positivos
        if positivos > 0
        else 0
    )

    return recall, capturados, k


def main():

    panel = pd.read_csv(PANEL_PATH)

    kev = pd.read_csv(KEV_PATH)

    kev = kev.rename(
        columns={"cveID": "cve_id"}
    )

    kev["dateAdded"] = pd.to_datetime(
        kev["dateAdded"],
        errors="coerce"
    )

    print("=" * 70)
    print("VALIDACIÓN DE HORIZONTES — OCTUBRE 2025")
    print("=" * 70)

    print(f"\nPanel: {len(panel):,} CVEs")

    resultados = []

    for dias in HORIZONTES:

        fecha_fin = (
            FECHA_T
            + pd.Timedelta(days=dias)
        )

        kev_futuros = kev[
            (kev["dateAdded"] > FECHA_T)
            & (kev["dateAdded"] <= fecha_fin)
        ]

        positivos_ids = set(
            kev_futuros["cve_id"]
        )

        target = f"y_kev_{dias}d"

        panel[target] = (
            panel["cve_id"]
            .isin(positivos_ids)
            .astype(int)
        )

        y = panel[target]

        positivos = int(y.sum())

        epss_ap = average_precision_score(
            y,
            panel["epss"]
        )

        cvss_ap = average_precision_score(
            y,
            panel["cvss_base"]
        )

        epss_r5, epss_hits, k = recall_at_5(
            panel,
            "epss",
            target
        )

        cvss_r5, cvss_hits, _ = recall_at_5(
            panel,
            "cvss_base",
            target
        )

        resultados.append({
            "horizonte": dias,
            "positivos": positivos,
            "epss_ap": epss_ap,
            "cvss_ap": cvss_ap,
            "epss_r5": epss_r5,
            "cvss_r5": cvss_r5,
            "epss_hits": epss_hits,
            "cvss_hits": cvss_hits,
            "k": k
        })

    print("\nRESULTADOS")
    print("-" * 70)

    for r in resultados:

        print(
            f"\nHorizonte: {r['horizonte']} días"
        )

        print(
            f"Positivos: {r['positivos']}"
        )

        print(
            f"EPSS AP:   {r['epss_ap']:.6f}"
        )

        print(
            f"CVSS AP:   {r['cvss_ap']:.6f}"
        )

        print(
            f"EPSS R@5%: {r['epss_r5']:.2%} "
            f"({r['epss_hits']}/{r['positivos']})"
        )

        print(
            f"CVSS R@5%: {r['cvss_r5']:.2%} "
            f"({r['cvss_hits']}/{r['positivos']})"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()