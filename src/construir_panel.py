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


def evaluar_ranking(
    df: pd.DataFrame,
    score: str,
    target: str,
):
    y = df[target].astype(int)
    scores = df[score].astype(float)

    ap = average_precision_score(
        y,
        scores,
    )

    auc = roc_auc_score(
        y,
        scores,
    )

    k = math.ceil(
        0.05 * len(df)
    )

    ranking = df.sort_values(
        score,
        ascending=False,
        kind="mergesort",
    )

    top = ranking.head(k)

    positivos = int(
        y.sum()
    )

    hits = int(
        top[target].sum()
    )

    recall = (
        hits / positivos
        if positivos
        else 0.0
    )

    precision = (
        hits / k
        if k
        else 0.0
    )

    return {
        "ap": ap,
        "auc": auc,
        "k": k,
        "hits": hits,
        "recall": recall,
        "precision": precision,
    }


def cargar_epss(
    fecha: str,
) -> pd.DataFrame:

    ruta = Path(
        f"../data/epss/"
        f"epss_scores-{fecha}.csv.gz"
    )

    if not ruta.exists():

        raise FileNotFoundError(
            f"Falta el histórico EPSS: "
            f"{ruta}\n"
            f"Descarga el archivo "
            f"correspondiente y colócalo "
            f"en data/epss."
        )

    epss = pd.read_csv(
        ruta,
        compression="gzip",
        skiprows=1,
    )

    epss = epss.rename(
        columns={
            "cve": "cve_id"
        }
    )

    epss["cve_id"] = (
        epss["cve_id"]
        .astype(str)
        .str.strip()
    )

    return (
        epss[
            [
                "cve_id",
                "epss",
                "percentile",
            ]
        ]
        .drop_duplicates(
            "cve_id"
        )
    )


def cargar_kev() -> pd.DataFrame:

    ruta = Path(
        "../data/kev/"
        "known_exploited_vulnerabilities.csv"
    )

    if not ruta.exists():

        raise FileNotFoundError(
            f"Falta el catálogo KEV: "
            f"{ruta}"
        )

    kev = pd.read_csv(
        ruta
    )

    kev = kev.rename(
        columns={
            "cveID": "cve_id"
        }
    )

    kev["cve_id"] = (
        kev["cve_id"]
        .astype(str)
        .str.strip()
    )

    kev["dateAdded"] = (
        pd.to_datetime(
            kev["dateAdded"],
            errors="coerce",
        )
    )

    return kev


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Construye y evalúa "
            "un panel temporal "
            "de Wang & Xu."
        )
    )

    parser.add_argument(
        "--fecha",
        required=True,
        choices=list(
            PAPER_90D
        ),
    )

    parser.add_argument(
        "--horizonte",
        type=int,
        default=90,
        choices=[
            30,
            90,
            180,
        ],
    )

    args = parser.parse_args()

    fecha = args.fecha
    horizonte = args.horizonte

    fecha_t = pd.Timestamp(
        fecha
    )

    fecha_fin = (
        fecha_t
        + pd.Timedelta(
            days=horizonte
        )
    )

    # ==========================================
    # 1. CARGAR NVD
    # ==========================================

    nvd_path = Path(
        f"../outputs/"
        f"nvd_snapshot_{fecha}.csv"
    )

    if not nvd_path.exists():

        raise FileNotFoundError(
            f"Falta {nvd_path}"
        )

    nvd = pd.read_csv(
        nvd_path
    )

    # Los snapshots históricos pueden traer
    # fechas con Z / UTC.
    # Se convierten a UTC y luego se quita
    # la zona horaria para compararlas con T.
    nvd["published"] = (
        pd.to_datetime(
            nvd["published"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
    )

    print("=" * 72)
    print(
        f"PANEL {fecha} | "
        f"HORIZONTE "
        f"{horizonte} DÍAS"
    )
    print("=" * 72)

    print(
        f"\nNVD cargado: "
        f"{len(nvd):,}"
    )

    # ==========================================
    # 2. FILTRO TEMPORAL Y CVE YEAR
    # ==========================================

    cve_year = pd.to_numeric(
        nvd["cve_id"]
        .str.extract(
            r"^CVE-(\d{4})-"
        )[0],
        errors="coerce",
    )

    fecha_inicio = pd.Timestamp(
        "2017-01-01"
    )

    fecha_fin_dia = (
        fecha_t
        + pd.Timedelta(
            days=1
        )
        - pd.Timedelta(
            microseconds=1
        )
    )

    nvd = nvd[
        (
            nvd["published"]
            >= fecha_inicio
        )
        &
        (
            nvd["published"]
            <= fecha_fin_dia
        )
        &
        (
            cve_year >= 2017
        )
    ].copy()

    nvd = nvd.drop_duplicates(
        subset="cve_id",
        keep="first",
    )

    print(
        "NVD después de filtros "
        f"2017-T: {len(nvd):,}"
    )

    # ==========================================
    # 3. CARGAR EPSS
    # ==========================================

    epss = cargar_epss(
        fecha
    )

    print(
        f"EPSS cargado: "
        f"{len(epss):,}"
    )

    # ==========================================
    # 4. COMMON SUPPORT NVD + EPSS
    # ==========================================

    panel = nvd.merge(
        epss,
        on="cve_id",
        how="inner",
    )

    print(
        "NVD + EPSS: "
        f"{len(panel):,}"
    )

    # ==========================================
    # 5. CARGAR KEV
    # ==========================================

    kev = cargar_kev()

    # CVEs que ya estaban en KEV en T
    kev_previos = set(
        kev.loc[
            kev["dateAdded"]
            <= fecha_t,
            "cve_id",
        ]
    )

    panel = panel[
        ~panel[
            "cve_id"
        ].isin(
            kev_previos
        )
    ].copy()

    print(
        "Después de excluir "
        "KEV previos: "
        f"{len(panel):,}"
    )

    # ==========================================
    # 6. TARGET FUTURO
    # ==========================================

    futuros = set(
        kev.loc[
            (
                kev["dateAdded"]
                > fecha_t
            )
            &
            (
                kev["dateAdded"]
                <= fecha_fin
            ),
            "cve_id",
        ]
    )

    target = (
        f"y_kev_"
        f"{horizonte}d"
    )

    panel[target] = (
        panel["cve_id"]
        .isin(
            futuros
        )
        .astype(int)
    )

    # ==========================================
    # 7. GUARDAR PANEL
    # ==========================================

    salida = Path(
        f"../outputs/"
        f"panel_{fecha}_"
        f"{horizonte}d.csv"
    )

    panel.to_csv(
        salida,
        index=False,
    )

    # ==========================================
    # 8. EVALUAR EPSS Y CVSS
    # ==========================================

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

    n = len(
        panel
    )

    positivos = int(
        panel[target].sum()
    )

    print()
    print("=" * 72)
    print("RESULTADOS")
    print("=" * 72)

    print(
        f"N: "
        f"{n:,}"
    )

    print(
        f"Positivos: "
        f"{positivos}"
    )

    print(
        f"k@5%: "
        f"{epss_result['k']:,}"
    )

    print("\nEPSS")

    print(
        f"AP:       "
        f"{epss_result['ap']:.9f}"
    )

    print(
        f"ROC-AUC:  "
        f"{epss_result['auc']:.9f}"
    )

    print(
        f"Recall@5%: "
        f"{epss_result['recall']:.2%} "
        f"("
        f"{epss_result['hits']}/"
        f"{positivos}"
        f")"
    )

    print("\nCVSS BASE")

    print(
        f"AP:       "
        f"{cvss_result['ap']:.9f}"
    )

    print(
        f"ROC-AUC:  "
        f"{cvss_result['auc']:.9f}"
    )

    print(
        f"Recall@5%: "
        f"{cvss_result['recall']:.2%} "
        f"("
        f"{cvss_result['hits']}/"
        f"{positivos}"
        f")"
    )

    # ==========================================
    # 9. COMPARAR CON PAPER
    # ==========================================

    if horizonte == 90:

        p = PAPER_90D[
            fecha
        ]

        print()
        print("=" * 72)
        print(
            "COMPARACIÓN "
            "CON WANG & XU"
        )
        print("=" * 72)

        print(
            f"N:          "
            f"{n:,} "
            f"vs "
            f"{p['n']:,} "
            f"(dif "
            f"{n - p['n']:+,}"
            f")"
        )

        print(
            f"Positivos:  "
            f"{positivos} "
            f"vs "
            f"{p['positivos']} "
            f"(dif "
            f"{positivos - p['positivos']:+}"
            f")"
        )

        print(
            f"EPSS AP:    "
            f"{epss_result['ap']:.6f} "
            f"vs "
            f"{p['epss_ap']:.6f}"
        )

        print(
            f"CVSS AP:    "
            f"{cvss_result['ap']:.6f} "
            f"vs "
            f"{p['cvss_ap']:.6f}"
        )

        print(
            f"EPSS R@5%:  "
            f"{epss_result['recall']:.2%} "
            f"vs "
            f"{p['epss_r5']:.2%}"
        )

        print(
            f"CVSS R@5%:  "
            f"{cvss_result['recall']:.2%} "
            f"vs "
            f"{p['cvss_r5']:.2%}"
        )

        exact_counts = (
            n == p["n"]
            and
            positivos
            == p["positivos"]
        )

        close_metrics = (
            math.isclose(
                epss_result["ap"],
                p["epss_ap"],
                abs_tol=5e-7,
            )
            and
            math.isclose(
                cvss_result["ap"],
                p["cvss_ap"],
                abs_tol=5e-7,
            )
            and
            math.isclose(
                epss_result["recall"],
                p["epss_r5"],
                abs_tol=5e-4,
            )
            and
            math.isclose(
                cvss_result["recall"],
                p["cvss_r5"],
                abs_tol=5e-4,
            )
        )

        print()

        if (
            exact_counts
            and close_metrics
        ):
            print(
                "RESULTADO: "
                "REPLICACIÓN COINCIDE"
            )

        elif exact_counts:
            print(
                "CONTEOS: COINCIDEN"
            )

            print(
                "MÉTRICAS: "
                "REVISAR DIFERENCIAS"
            )

        else:
            print(
                "CONTEOS: "
                "TODAVÍA HAY "
                "DIFERENCIAS"
            )

    print()
    print(
        f"Panel guardado en: "
        f"{salida}"
    )


if __name__ == "__main__":
    main()