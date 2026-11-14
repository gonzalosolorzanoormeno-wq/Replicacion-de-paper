import json
from pathlib import Path
nvd_dir = Path("../data/nvd/nvdcve-repo/nvdcve-repo/nvdcve")
encontrado = False
for archivo in nvd_dir.glob("CVE-2025-*.json"):
    with open(archivo, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})

    # Revisar CVSS 3.1 y 3.0
    candidatos = (
        metrics.get("cvssMetricV31", [])
        + metrics.get("cvssMetricV30", [])
    )

    for metrica in candidatos:
        source = metrica.get("source", "")
        tipo = metrica.get("type", "")

        if "nvd@nist.gov" in source.lower():
            print("=== ENCONTRADO ===")
            print("CVE:", data.get("id"))
            print("Publicado:", data.get("published"))
            print("Source:", source)
            print("Type:", tipo)

            cvss = metrica.get("cvssData", {})

            print("Versión:", cvss.get("version"))
            print("Base score:", cvss.get("baseScore"))
            print("Base severity:", cvss.get("baseSeverity"))
            print("Exploitability:", metrica.get("exploitabilityScore"))
            print("Impact:", metrica.get("impactScore"))
            print("Vector:", cvss.get("vectorString"))

            encontrado = True
            break

    if encontrado:
        break

if not encontrado:
    print("No se encontró una métrica NVD en los CVE-2025 revisados.")