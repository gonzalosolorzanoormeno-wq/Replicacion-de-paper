import pandas as pd
from pathlib import Path

EPSS_PATH = Path("../data/epss/epss_scores-2025-10-01.csv.gz")
KEV_PATH = Path("../data/kev/known_exploited_vulnerabilities.csv")
OUTPUT_DIR = Path("../outputs")

FECHA_T = pd.Timestamp("2025-10-01")
FIN_90D = FECHA_T + pd.Timedelta(days=90)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# EPSS
# =========================

epss = pd.read_csv(
    EPSS_PATH,
    compression="gzip",
    skiprows=1
)

epss = epss.rename(columns={"cve": "cve_id"})

epss["cve_id"] = epss["cve_id"].astype(str).str.strip()

print("===== EPSS =====")
print("Registros:", len(epss))
print("Duplicados CVE:", epss["cve_id"].duplicated().sum())
print("EPSS nulos:", epss["epss"].isna().sum())

epss.to_csv(
    OUTPUT_DIR / "epss_2025-10-01_limpio.csv",
    index=False
)

# =========================
# KEV
# =========================

kev = pd.read_csv(KEV_PATH)

kev = kev.rename(columns={"cveID": "cve_id"})
kev["cve_id"] = kev["cve_id"].astype(str).str.strip()

kev["dateAdded"] = pd.to_datetime(
    kev["dateAdded"],
    errors="coerce"
)

print("\n===== KEV =====")
print("Registros actuales:", len(kev))
print("Duplicados CVE:", kev["cve_id"].duplicated().sum())
print("dateAdded nulos:", kev["dateAdded"].isna().sum())

# CVEs que ya estaban en KEV en la fecha T
kev_hasta_t = kev[
    kev["dateAdded"] <= FECHA_T
].copy()

# CVEs añadidos DESPUÉS de T y hasta T + 90 días
kev_90d = kev[
    (kev["dateAdded"] > FECHA_T)
    & (kev["dateAdded"] <= FIN_90D)
].copy()

print("\n===== RECONSTRUCCIÓN TEMPORAL =====")
print("Fecha T:", FECHA_T.date())
print("Fin horizonte 90 días:", FIN_90D.date())

print(
    "CVEs que ya estaban en KEV en T:",
    len(kev_hasta_t)
)
print(
    "CVEs incorporados a KEV durante los 90 días posteriores:",
    len(kev_90d)
)
print("\nPrimeros KEV futuros:")
print(
    kev_90d[
        ["cve_id", "dateAdded", "vendorProject", "product"]
    ].head(20)
)
kev_hasta_t.to_csv(
    OUTPUT_DIR / "kev_hasta_2025-10-01.csv",
    index=False
)
kev_90d.to_csv(
    OUTPUT_DIR / "kev_futuros_90d_2025-10-01.csv",
    index=False
)
print("\nArchivos generados:")
print("- epss_2025-10-01_limpio.csv")
print("- kev_hasta_2025-10-01.csv")
print("- kev_futuros_90d_2025-10-01.csv")