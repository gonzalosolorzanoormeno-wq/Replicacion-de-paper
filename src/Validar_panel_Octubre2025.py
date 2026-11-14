import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PANEL = "../outputs/panel_2025-10-01_PROVISIONAL.csv"

# Valores reportados por Wang y Xu para 2025-10-01
PAPER_N = 158_076
PAPER_POSITIVES = 11
PAPER_EPSS_AP = 0.022486
PAPER_CVSS_AP = 0.000192
PAPER_EPSS_R5 = 0.5455
PAPER_CVSS_R5 = 0.0909


def main():
    df = pd.read_csv(PANEL)

    print("=" * 60)
    print("VALIDACIÓN DE REPLICACIÓN — PANEL 2025-10-01")
    print("=" * 60)

    print("\nPanel provisional:")
    print(f"N: {len(df):,}")
    print(f"Positivos: {int(df['y_kev_90d'].sum())}")

    # Año codificado en el identificador CVE
    cve_year = (
        df["cve_id"]
        .str.extract(r"CVE-(\d{4})")[0]
        .astype(int)
    )

    old_mask = cve_year < 2017

    print("\nDiagnóstico del corte 2017:")
    print(f"CVE con año de ID < 2017: {old_mask.sum():,}")
    print(
        "Positivos entre CVE con año de ID < 2017:",
        int(df.loc[old_mask, "y_kev_90d"].sum()),
    )

    # Panel que reproduce los conteos publicados
    x = df.loc[cve_year >= 2017].copy()

    y = x["y_kev_90d"].astype(int)

    n = len(x)
    positives = int(y.sum())
    k = int(np.ceil(0.05 * n))

    epss_ap = average_precision_score(y, x["epss"])
    cvss_ap = average_precision_score(y, x["cvss_base"])

    epss_top = (
        x.sort_values(
            "epss",
            ascending=False,
            kind="mergesort",
        )
        .head(k)
    )

    cvss_top = (
        x.sort_values(
            "cvss_base",
            ascending=False,
            kind="mergesort",
        )
        .head(k)
    )

    epss_hits = int(epss_top["y_kev_90d"].sum())
    cvss_hits = int(cvss_top["y_kev_90d"].sum())

    epss_r5 = epss_hits / positives
    cvss_r5 = cvss_hits / positives

    print("\nPanel después de aplicar CVE year >= 2017:")
    print(f"N: {n:,}")
    print(f"Positivos: {positives}")
    print(f"k @5%: {k:,}")

    print("\nResultados reproducidos:")
    print(f"EPSS AP:       {epss_ap:.9f}")
    print(f"CVSS AP:       {cvss_ap:.9f}")
    print(f"EPSS R@5%:     {epss_r5:.4%} ({epss_hits}/{positives})")
    print(f"CVSS R@5%:     {cvss_r5:.4%} ({cvss_hits}/{positives})")

    print("\nComparación con el paper:")
    print(f"N:             {n:,} vs {PAPER_N:,}")
    print(f"Positivos:     {positives} vs {PAPER_POSITIVES}")
    print(f"EPSS AP:       {epss_ap:.6f} vs {PAPER_EPSS_AP:.6f}")
    print(f"CVSS AP:       {cvss_ap:.6f} vs {PAPER_CVSS_AP:.6f}")
    print(f"EPSS R@5%:     {epss_r5:.2%} vs {PAPER_EPSS_R5:.2%}")
    print(f"CVSS R@5%:     {cvss_r5:.2%} vs {PAPER_CVSS_R5:.2%}")

    exact_counts = (
        n == PAPER_N
        and positives == PAPER_POSITIVES
    )

    close_metrics = (
        np.isclose(epss_ap, PAPER_EPSS_AP, atol=5e-7)
        and np.isclose(cvss_ap, PAPER_CVSS_AP, atol=5e-7)
        and np.isclose(epss_r5, PAPER_EPSS_R5, atol=5e-5)
        and np.isclose(cvss_r5, PAPER_CVSS_R5, atol=5e-5)
    )

    print("\n" + "=" * 60)

    if exact_counts and close_metrics:
        print("RESULTADO: REPLICACIÓN DEL PANEL OCTUBRE COINCIDE")
    else:
        print("RESULTADO: TODAVÍA EXISTEN DIFERENCIAS")

    print("=" * 60)


if __name__ == "__main__":
    main()