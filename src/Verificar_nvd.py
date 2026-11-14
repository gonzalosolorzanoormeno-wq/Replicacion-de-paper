import json
from pathlib import Path
from pprint import pprint
nvd_dir = Path("../data/nvd/nvdcve-repo/nvdcve-repo/nvdcve")
archivo = nvd_dir / "CVE-2025-0001.json"
with open(archivo, "r", encoding="utf-8") as f:
    data = json.load(f)

print("ID:", data.get("id"))
print("Publicado:", data.get("published"))
print("\n=== CVSS V3.1 ===")
pprint(data.get("metrics", {}).get("cvssMetricV31", []))