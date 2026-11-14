from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


COMMITS_NVD = {
    "2025-04-01": "e8af36960333",
    "2025-07-01": "436a9fad9aef",
    "2025-10-01": "3d1d5e8d036d",
    "2026-01-01": "2c076a9033a6",
    "2026-03-25": "d4c05f32b5b2",
}


def ejecutar_git(repo: Path, *args: str) -> str:
    cmd = ["git", "-C", str(repo), *args]
    return subprocess.check_output(
        cmd,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def detectar_repo_nvd() -> Path:
    candidatos = [
        Path("../data/nvd/nvdcve-repo/nvdcve-repo"),
        Path("../data/nvd/nvdcve-repo"),
    ]

    for candidato in candidatos:
        if (candidato / ".git").exists():
            return candidato

    raise FileNotFoundError(
        "No se encontró el repositorio git de NVD. "
        "Usa --repo-nvd para indicar la ruta."
    )


def procesar_archivo(archivo: Path, fecha_t: str):
    try:
        with archivo.open("r", encoding="utf-8") as f:
            data = json.load(f)

        cve_id = data.get("id")
        published = data.get("published")

        if not cve_id or not published:
            return None

        # Mantener la misma regla con la que octubre reprodujo el paper.
        try:
            cve_year = int(cve_id.split("-")[1])
        except (IndexError, ValueError):
            return None

        if cve_year < 2017:
            return None

        # El commit histórico ya evita información posterior,
        # pero conservamos el filtro por fecha de decisión.
        if published[:10] > fecha_t:
            return None

        metrics = data.get("metrics", {})
        metrica_nvd = None

        # El paper utiliza CVSS v3.1/v3.0 atribuido a NVD.
        for clave in ("cvssMetricV31", "cvssMetricV30"):
            for metrica in metrics.get(clave, []):
                source = str(metrica.get("source", "")).lower()
                if source == "nvd@nist.gov":
                    metrica_nvd = metrica
                    break
            if metrica_nvd is not None:
                break

        if metrica_nvd is None:
            return None

        cvss = metrica_nvd.get("cvssData", {})

        return {
            "cve_id": cve_id,
            "published": published,
            "last_modified": data.get("lastModified"),
            "vuln_status": data.get("vulnStatus"),
            "cvss_version": cvss.get("version"),
            "cvss_base": cvss.get("baseScore"),
            "base_severity": cvss.get("baseSeverity"),
            "cvss_exploitability": metrica_nvd.get("exploitabilityScore"),
            "cvss_impact": metrica_nvd.get("impactScore"),
            "attack_vector": cvss.get("attackVector"),
            "attack_complexity": cvss.get("attackComplexity"),
            "privileges_required": cvss.get("privilegesRequired"),
            "user_interaction": cvss.get("userInteraction"),
            "scope": cvss.get("scope"),
            "confidentiality_impact": cvss.get("confidentialityImpact"),
            "integrity_impact": cvss.get("integrityImpact"),
            "availability_impact": cvss.get("availabilityImpact"),
            "vector_string": cvss.get("vectorString"),
        }

    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Genera el snapshot NVD point-in-time usado por Wang & Xu."
    )
    parser.add_argument(
        "--fecha",
        required=True,
        choices=list(COMMITS_NVD),
        help="Fecha de decisión T.",
    )
    parser.add_argument(
        "--repo-nvd",
        type=Path,
        default=None,
        help="Ruta al repositorio git nvdcve-repo.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
    )
    args = parser.parse_args()

    fecha_t = args.fecha
    commit = COMMITS_NVD[fecha_t]
    repo = args.repo_nvd or detectar_repo_nvd()
    repo = repo.resolve()

    if not (repo / ".git").exists():
        raise FileNotFoundError(f"No es un repositorio git válido: {repo}")

    estado = ejecutar_git(repo, "status", "--porcelain")
    if estado:
        raise RuntimeError(
            "El repositorio NVD tiene cambios locales. "
            "Guárdalos o descártalos antes de continuar."
        )

    head_original = ejecutar_git(repo, "rev-parse", "HEAD")
    rama_original = ejecutar_git(repo, "symbolic-ref", "--short", "-q", "HEAD")

    salida = Path(f"../outputs/nvd_snapshot_{fecha_t}.csv")
    salida.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SNAPSHOT NVD HISTÓRICO")
    print("=" * 72)
    print(f"Fecha T: {fecha_t}")
    print(f"Commit del paper: {commit}")
    print(f"Repositorio: {repo}")

    try:
        ejecutar_git(repo, "checkout", "--quiet", commit)

        nvd_dir = repo / "nvdcve"
        if not nvd_dir.exists():
            raise FileNotFoundError(
                f"No se encontró la carpeta nvdcve dentro de {repo}"
            )

        year_max = pd.Timestamp(fecha_t).year
        archivos = []

        for year in range(2017, year_max + 1):
            encontrados = list(nvd_dir.glob(f"CVE-{year}-*.json"))
            archivos.extend(encontrados)
            print(f"{year}: {len(encontrados):,} archivos")

        print(f"Total a procesar: {len(archivos):,}")

        filas = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            resultados = executor.map(
                lambda p: procesar_archivo(p, fecha_t),
                archivos,
            )

            for i, fila in enumerate(resultados, start=1):
                if fila is not None:
                    filas.append(fila)

                if i % 10000 == 0:
                    print(
                        f"Procesados {i:,}/{len(archivos):,} | "
                        f"CVSS NVD válidos: {len(filas):,}"
                    )

        df = pd.DataFrame(filas)
        df = df.drop_duplicates(subset="cve_id", keep="first")
        df.to_csv(salida, index=False)

        print("\nRESULTADO")
        print(f"CVEs con CVSS v3 de NVD: {len(df):,}")
        print(f"Archivo: {salida}")

    finally:
        destino = rama_original if rama_original else head_original
        ejecutar_git(repo, "checkout", "--quiet", destino)
        print(f"Repositorio NVD restaurado a: {destino}")


if __name__ == "__main__":
    main()
