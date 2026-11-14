import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

NVD_DIR = Path("../data/nvd/nvdcve-repo/nvdcve-repo/nvdcve")
SALIDA = Path("../outputs/nvd_snapshot_2025-10-01.csv")

FECHA_T = "2025-10-01"


def procesar_archivo(archivo):
    try:
        with open(archivo, "r", encoding="utf-8") as f:
            data = json.load(f)

        published = data.get("published")

        if not published:
            return None

        fecha_publicacion = published[:10]

        # Excluir CVEs publicados después de T
        if fecha_publicacion > FECHA_T:
            return None

        metrics = data.get("metrics", {})

        metrica_nvd = None

        # Primero CVSS 3.1, luego 3.0
        for clave in ["cvssMetricV31", "cvssMetricV30"]:
            for metrica in metrics.get(clave, []):
                if str(metrica.get("source", "")).lower() == "nvd@nist.gov":
                    metrica_nvd = metrica
                    break

            if metrica_nvd:
                break

        if metrica_nvd is None:
            return None

        cvss = metrica_nvd.get("cvssData", {})

        return {
            "cve_id": data.get("id"),
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


print("Buscando archivos 2017-2025...")

archivos = []

for year in range(2017, 2026):
    encontrados = list(NVD_DIR.glob(f"CVE-{year}-*.json"))
    archivos.extend(encontrados)
    print(f"{year}: {len(encontrados):,}")

print(f"\nTotal a procesar: {len(archivos):,}")
print("Comenzando procesamiento paralelo...\n")

filas = []

with ThreadPoolExecutor(max_workers=12) as executor:
    resultados = executor.map(procesar_archivo, archivos)

    for i, resultado in enumerate(resultados, start=1):

        if resultado is not None:
            filas.append(resultado)

        if i % 5000 == 0:
            print(
                f"Procesados: {i:,}/{len(archivos):,} | "
                f"CVSS NVD encontrados: {len(filas):,}"
            )


df = pd.DataFrame(filas)

SALIDA.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(SALIDA, index=False)

print("\n==============================")
print("RESULTADOS")
print("==============================")

print(f"Archivos considerados: {len(archivos):,}")
print(f"CVEs con CVSS v3 de NVD: {len(df):,}")

print("\nArchivo generado:")
print(SALIDA)

print("\nPrimeras filas:")
print(df.head())