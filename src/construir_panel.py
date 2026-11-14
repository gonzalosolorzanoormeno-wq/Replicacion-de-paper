from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PAPER_90D = {
    "2025-04-01": {
        "n": 152_422,
        "positivos": 15,
        "epss_ap": 0.001150,
        "cvss_ap": 0.000139,
        "epss_r5": 0.4000,
        "cvss_r5": 0.0000,
    },
    "2025-07-01": {
        "n": 158_281,
        "positivos": 11,
        "epss_ap": 0.011416,
        "cvss_ap": 0.000128,
        "epss_r5": 0.6364,
        "cvss_r5": 0.0000,
    },
    "2025-10-01": {
        "n": 158_076,
        "positivos": 11,
        "epss_ap": 0.022486,
        "cvss_ap": 0.000192,
        "epss_r5": 0.5455,
        "cvss_r5": 0.0909,
    },
    "2026-01-01": {
        "n": 165_222,
        "positivos": 19,
        "epss_ap": 0.002860,
        "cvss_ap": 0.000994,
        "epss_r5": 0.4737,
        "cvss_r5": 0.1579,
    },
    "2026-03-25": {
        "n": 171_188,
        "positivos": 12,
        "epss_ap": 0.000783,
        "cvss_ap": 0.000206,
        "epss_r5": 0.5833,
        "cvss_r5": 0.0833,
    },
}


def evaluar_ranking(df: pd.DataFrame, score: str, target: str):
    y = df[target].astype(int)
    scores = df[score].astype(float)

    ap = average_precision_score(y, scores)
    auc = roc_auc_score(y, scores)

    k = math.ceil(0.05 * len(df))
    ranking = df.sort_values(
        score,
        ascending=False,
        kind="mergesort",
    )
    top = ranking.head(k)

    positivos = int(y.sum())
    hits = int(top[target].sum())
    recall = hits / positivos if positivos else 0.0
    precision = hits / k if k else 0.0

    return {
        "ap": ap,
        "auc": auc,
        "k": k,
        "hits": hits,
        "recall": recall,
        "precision": precision,
    }


def cargar_epss(fecha: str) -> pd.DataFrame:
    ruta = Path(f"../data/epss/epss_scores-{fecha}.csv.gz")

    if not ruta.exists():
        raise FileNotFoundError(
            f"Falta el histórico EPSS: {ruta}\n"
            "Descarga el archivo correspondiente de FIRST y colócalo allí."
        )

    epss = pd.read_csv(
        ruta,
        compression="gzip",
        skiprows=1,
    )
    epss = epss.rename(columns={"cve": "cve_id"})
    epss["cve_id"] = epss["cve_id"].astype(str).str.strip()

    return epss[["cve_id", "epss", "percentile"]].drop_duplicates("cve_id")


def cargar_kev() -> pd.DataFrame:
    ruta = Path("../data/kev/known_exploited_vulnerabilities.csv")

    if not ruta.exists():
        raise FileNotFoundError(f"Falta el catálogo KEV: {ruta}")

    kev = pd.read_csv(ruta)
    kev = kev.rename(columns={"cveID": "cve_id"})
    kev["cve_id"] = kev["cve_id"].astype(str).str.strip()
    kev["dateAdded"] = pd.to_datetime(
        kev["dateAdded"],
        errors="coerce",
    )
    return kev


def main():
    parser = argparse.ArgumentParser(
        description="Construye y evalúa un panel temporal de Wang & Xu."
    )
    parser.add_argument(
        "--fecha",
        required=True,
        choices=list(PAPER_90D),
    )
    parser.add_argument(
        "--horizonte",
        type=int,
        default=90,
        choices=[30, 90, 180],
    )
    args = parser.parse_args()

    fecha = args.fecha
    horizonte = args.horizonte

    fecha_t = pd.Timestamp(fecha)
    fecha_fin = fecha_t + pd.Timedelta(days=horizonte)

    nvd_path = Path(f"../outputs/nvd_snapshot_{fecha}.csv")
    if not nvd_path.exists():
        raise FileNotFoundError(
            f"Falta {nvd_path}. Ejecuta primero:\n"
            f"python generar_snapshot_nvd.py --fecha {fecha}"
        )

    nvd = pd.read_csv(nvd_path)
    nvd["published"] = pd.to_datetime(
        nvd["published"],
        errors="coerce",
    )

    # Mantener la regla empíricamente validada en octubre:
    # fecha de publicación >= 2017 y año del identificador CVE >= 2017.
    cve_year = pd.to_numeric(
        nvd["cve_id"].str.extract(r"^CVE-(\d{4})-")[0],
        errors="coerce",
    )

    nvd = nvd[
        (nvd["published"] >= pd.Timestamp("2017-01-01"))
        & (nvd["published"] <= fecha_t + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
        & (cve_year >= 2017)
    ].copy()

    nvd = nvd.drop_duplicates(subset="cve_id", keep="first")

    epss = cargar_epss(fecha)
    kev = cargar_kev()

    panel = nvd.merge(
        epss,
        on="cve_id",
        how="inner",
    )

    kev_previos = set(
        kev.loc[kev["dateAdded"] <= fecha_t, "cve_id"]
    )
    panel = panel[
        ~panel["cve_id"].isin(kev_previos)
    ].copy()

    futuros = set(
        kev.loc[
            (kev["dateAdded"] > fecha_t)
            & (kev["dateAdded"] <= fecha_fin),
            "cve_id",
        ]
    )

    target = f"y_kev_{horizonte}d"
    panel[target] = panel["cve_id"].isin(futuros).astype(int)

    salida = Path(
        f"../outputs/panel_{fecha}_{horizonte}d.csv"
    )
    panel.to_csv(salida, index=False)

    epss_result = evaluar_ranking(
        panel,
        "epss",
        target,
    )
    cvss_result = evaluar_ranking(
        panel,
        "cvss_base",
        target,
    )

    n = len(panel)
    positivos = int(panel[target].sum())

    print("=" * 72)
    print(f"PANEL {fecha} | HORIZONTE {horizonte} DÍAS")
    print("=" * 72)
    print(f"N: {n:,}")
    print(f"Positivos: {positivos}")
    print(f"k@5%: {epss_result['k']:,}")

    print("\nEPSS")
    print(f"AP:       {epss_result['ap']:.9f}")
    print(f"ROC-AUC:  {epss_result['auc']:.9f}")
    print(
        f"Recall@5%: {epss_result['recall']:.2%} "
        f"({epss_result['hits']}/{positivos})"
    )

    print("\nCVSS BASE")
    print(f"AP:       {cvss_result['ap']:.9f}")
    print(f"ROC-AUC:  {cvss_result['auc']:.9f}")
    print(
        f"Recall@5%: {cvss_result['recall']:.2%} "
        f"({cvss_result['hits']}/{positivos})"
    )

    if horizonte == 90:
        p = PAPER_90D[fecha]

        print("\nCOMPARACIÓN CON WANG & XU")
        print(
            f"N:          {n:,} vs {p['n']:,} "
            f"(dif {n - p['n']:+,})"
        )
        print(
            f"Positivos:  {positivos} vs {p['positivos']} "
            f"(dif {positivos - p['positivos']:+})"
        )
        print(
            f"EPSS AP:    {epss_result['ap']:.6f} vs {p['epss_ap']:.6f}"
        )
        print(
            f"CVSS AP:    {cvss_result['ap']:.6f} vs {p['cvss_ap']:.6f}"
        )
        print(
            f"EPSS R@5%:  {epss_result['recall']:.2%} vs {p['epss_r5']:.2%}"
        )
        print(
            f"CVSS R@5%:  {cvss_result['recall']:.2%} vs {p['cvss_r5']:.2%}"
        )

        exact_counts = (
            n == p["n"]
            and positivos == p["positivos"]
        )

        if exact_counts:
            print("\nCONTEOS: COINCIDEN")
        else:
            print("\nCONTEOS: TODAVÍA HAY DIFERENCIAS")

    print(f"\nPanel guardado en: {salida}")


if __name__ == "__main__":
    main()
