import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


# ============================================================
# UTILIDAD COMÚN PARA CONSTRUIR LA FILA
# ============================================================

def construir_fila(
    cve_id,
    published,
    last_modified,
    vuln_status,
    cvss,
    exploitability_score,
    impact_score,
):
    version = str(
        cvss.get(
            "version",
            ""
        )
    )

    if version not in (
        "3.0",
        "3.1",
    ):
        return None

    return {
        "cve_id": cve_id,
        "published": published,
        "last_modified": last_modified,
        "vuln_status": vuln_status,

        "cvss_version":
            version,

        "cvss_base":
            cvss.get(
                "baseScore"
            ),

        "base_severity":
            cvss.get(
                "baseSeverity"
            ),

        "cvss_exploitability":
            exploitability_score,

        "cvss_impact":
            impact_score,

        "attack_vector":
            cvss.get(
                "attackVector"
            ),

        "attack_complexity":
            cvss.get(
                "attackComplexity"
            ),

        "privileges_required":
            cvss.get(
                "privilegesRequired"
            ),

        "user_interaction":
            cvss.get(
                "userInteraction"
            ),

        "scope":
            cvss.get(
                "scope"
            ),

        "confidentiality_impact":
            cvss.get(
                "confidentialityImpact"
            ),

        "integrity_impact":
            cvss.get(
                "integrityImpact"
            ),

        "availability_impact":
            cvss.get(
                "availabilityImpact"
            ),

        "vector_string":
            cvss.get(
                "vectorString"
            ),
    }


# ============================================================
# FORMATO NVD ANTIGUO
# Abril / Julio 2025
#
# publishedDate
# cve.CVE_data_meta.ID
# impact.baseMetricV3.cvssV3
# ============================================================

def procesar_formato_antiguo(
    data,
    fecha_t,
):
    published = data.get(
        "publishedDate"
    )

    if not published:
        return None

    if published[:10] > fecha_t:
        return None

    cve = data.get(
        "cve",
        {}
    )

    meta = cve.get(
        "CVE_data_meta",
        {}
    )

    cve_id = meta.get(
        "ID"
    )

    if not cve_id:
        return None

    impact = data.get(
        "impact",
        {}
    )

    base_metric_v3 = impact.get(
        "baseMetricV3"
    )

    if not base_metric_v3:
        return None

    cvss = base_metric_v3.get(
        "cvssV3",
        {}
    )

    if not cvss:
        return None

    return construir_fila(
        cve_id=cve_id,
        published=published,
        last_modified=data.get(
            "lastModifiedDate"
        ),
        vuln_status=data.get(
            "vulnStatus"
        ),
        cvss=cvss,
        exploitability_score=(
            base_metric_v3.get(
                "exploitabilityScore"
            )
        ),
        impact_score=(
            base_metric_v3.get(
                "impactScore"
            )
        ),
    )


# ============================================================
# FORMATO NVD NUEVO
# Octubre 2025 en adelante
#
# id
# published
# metrics.cvssMetricV31
# metrics.cvssMetricV30
#
# IMPORTANTE:
# Wang & Xu usan únicamente CVSS asignado por NVD.
# Por eso exigimos source == nvd@nist.gov.
# ============================================================

def buscar_cvss_nvd_nuevo(
    data,
):
    metrics = data.get(
        "metrics",
        {}
    )

    # El paper trabaja con CVSS v3.
    # Preferimos 3.1 y luego 3.0.
    claves = [
        "cvssMetricV31",
        "cvssMetricV30",
    ]

    for clave in claves:

        candidatos = metrics.get(
            clave,
            []
        )

        for metrica in candidatos:

            if (
                metrica.get(
                    "source"
                )
                != "nvd@nist.gov"
            ):
                continue

            cvss = metrica.get(
                "cvssData",
                {}
            )

            if not cvss:
                continue

            version = str(
                cvss.get(
                    "version",
                    ""
                )
            )

            if version not in (
                "3.0",
                "3.1",
            ):
                continue

            return (
                cvss,
                metrica.get(
                    "exploitabilityScore"
                ),
                metrica.get(
                    "impactScore"
                ),
            )

    return None


def procesar_formato_nuevo(
    data,
    fecha_t,
):
    published = data.get(
        "published"
    )

    if not published:
        return None

    if published[:10] > fecha_t:
        return None

    cve_id = data.get(
        "id"
    )

    if not cve_id:
        return None

    resultado_cvss = (
        buscar_cvss_nvd_nuevo(
            data
        )
    )

    if resultado_cvss is None:
        return None

    (
        cvss,
        exploitability_score,
        impact_score,
    ) = resultado_cvss

    return construir_fila(
        cve_id=cve_id,
        published=published,
        last_modified=data.get(
            "lastModified"
        ),
        vuln_status=data.get(
            "vulnStatus"
        ),
        cvss=cvss,
        exploitability_score=(
            exploitability_score
        ),
        impact_score=(
            impact_score
        ),
    )


# ============================================================
# DETECTOR AUTOMÁTICO DE FORMATO
# ============================================================

def procesar_archivo(
    archivo,
    fecha_t,
):
    try:
        with open(
            archivo,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        # Formato NVD nuevo
        if (
            "id" in data
            and
            "metrics" in data
        ):
            return (
                procesar_formato_nuevo(
                    data,
                    fecha_t,
                )
            )

        # Formato NVD antiguo
        if (
            "cve" in data
            and
            "impact" in data
        ):
            return (
                procesar_formato_antiguo(
                    data,
                    fecha_t,
                )
            )

        return None

    except Exception:
        return None


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extrae CVSS v3 NVD "
            "desde snapshots históricos "
            "de olbat/nvdcve, soportando "
            "formato NVD antiguo y nuevo."
        )
    )

    parser.add_argument(
        "--nvd-dir",
        required=True,
    )

    parser.add_argument(
        "--fecha",
        required=True,
    )

    parser.add_argument(
        "--salida",
        required=True,
    )

    args = parser.parse_args()

    nvd_dir = Path(
        args.nvd_dir
    )

    salida = Path(
        args.salida
    )

    fecha_t = args.fecha

    year_fin = int(
        fecha_t[:4]
    )

    print("=" * 72)
    print(
        "EXTRACCIÓN NVD HISTÓRICA"
    )
    print("=" * 72)

    print(
        f"Fecha T: {fecha_t}"
    )

    print(
        f"NVD: {nvd_dir}"
    )

    archivos = []

    for year in range(
        2017,
        year_fin + 1,
    ):

        encontrados = list(
            nvd_dir.glob(
                f"CVE-{year}-*.json"
            )
        )

        archivos.extend(
            encontrados
        )

        print(
            f"{year}: "
            f"{len(encontrados):,}"
        )

    print(
        "\nTotal a procesar: "
        f"{len(archivos):,}"
    )

    filas = []

    def worker(
        archivo
    ):
        return procesar_archivo(
            archivo,
            fecha_t,
        )

    with ThreadPoolExecutor(
        max_workers=12
    ) as executor:

        resultados = executor.map(
            worker,
            archivos,
        )

        for i, resultado in enumerate(
            resultados,
            start=1,
        ):

            if resultado is not None:
                filas.append(
                    resultado
                )

            if i % 5000 == 0:

                print(
                    "Procesados: "
                    f"{i:,}/"
                    f"{len(archivos):,}"
                    " | CVSS NVD: "
                    f"{len(filas):,}"
                )

    # Evitar volver a generar un CSV vacío
    # silenciosamente.
    if len(filas) == 0:

        raise RuntimeError(
            "\nERROR: no se encontró "
            "ningún CVSS v3 asignado "
            "por NVD.\n"
            "No se generará un CSV vacío."
        )

    df = pd.DataFrame(
        filas
    )

    df = df.drop_duplicates(
        subset="cve_id",
        keep="first",
    )

    salida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        salida,
        index=False,
    )

    sin_cvss = (
        len(archivos)
        - len(df)
    )

    missing_pct = (
        sin_cvss
        / len(archivos)
        * 100
        if archivos
        else 0
    )

    print()
    print("=" * 72)
    print(
        "RESULTADO"
    )
    print("=" * 72)

    print(
        "NVD records examinados: "
        f"{len(archivos):,}"
    )

    print(
        "CVEs con CVSS v3 NVD: "
        f"{len(df):,}"
    )

    print(
        "Sin CVSS v3: "
        f"{sin_cvss:,}"
    )

    print(
        "Missing CVSS v3: "
        f"{missing_pct:.2f}%"
    )

    print(
        "\nArchivo generado: "
        f"{salida}"
    )


if __name__ == "__main__":
    main()