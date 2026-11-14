from pathlib import Path
import pandas as pd

# =========================
# RUTAS
# =========================

NVD_PATH = Path("../outputs/nvd_snapshot_2025-10-01.csv")
EPSS_PATH = Path("../outputs/epss_2025-10-01_limpio.csv")
KEV_HASTA_T_PATH = Path("../outputs/kev_hasta_2025-10-01.csv")
KEV_90D_PATH = Path("../outputs/kev_futuros_90d_2025-10-01.csv")

SALIDA = Path("../outputs/panel_2025-10-01_PROVISIONAL.csv")

FECHA_INICIO = pd.Timestamp("2017-01-01")
FECHA_T = pd.Timestamp("2025-10-01")


# =========================
# 1. CARGAR NVD
# =========================

nvd = pd.read_csv(NVD_PATH)

nvd["published"] = pd.to_datetime(
    nvd["published"],
    errors="coerce"
)

print("===== NVD =====")
print("Registros cargados:", len(nvd))

# El paper describe CVEs desde 2017 hasta T.
# Para reproducir exactamente los conteos publicados,
# también restringimos el año del identificador CVE a >= 2017.
nvd["cve_year"] = pd.to_numeric(
    nvd["cve_id"]
    .str.extract(r"^CVE-(\d{4})-")[0],
    errors="coerce"
)

nvd = nvd[
    (nvd["published"] >= FECHA_INICIO)
    & (nvd["published"] <= FECHA_T)
    & (nvd["cve_year"] >= 2017)
].copy()

print(
    "NVD elegibles por fecha y CVE year >= 2017:",
    len(nvd)
)
print(
    "Duplicados CVE en NVD:",
    nvd["cve_id"].duplicated().sum()
)

nvd = nvd.drop_duplicates(
    subset="cve_id",
    keep="first"
)


# =========================
# 2. CARGAR EPSS
# =========================

epss = pd.read_csv(EPSS_PATH)

print("\n===== EPSS =====")
print("Registros EPSS:", len(epss))


# =========================
# 3. COMMON SUPPORT
# NVD + EPSS
# =========================

panel = nvd.merge(
    epss[["cve_id", "epss", "percentile"]],
    on="cve_id",
    how="inner"
)

print("\n===== NVD + EPSS =====")
print("CVEs con NVD CVSS + EPSS:", len(panel))


# =========================
# 4. EXCLUIR CVEs YA EN KEV
# =========================

kev_hasta_t = pd.read_csv(
    KEV_HASTA_T_PATH
)

kev_hasta_t_ids = set(
    kev_hasta_t["cve_id"]
)

panel = panel[
    ~panel["cve_id"].isin(kev_hasta_t_ids)
].copy()

print("\n===== DESPUÉS DE EXCLUIR KEV EXISTENTES =====")
print("CVEs elegibles provisionales:", len(panel))


# =========================
# 5. CREAR TARGET 90 DÍAS
# =========================

kev_90d = pd.read_csv(
    KEV_90D_PATH
)

kev_90d_ids = set(
    kev_90d["cve_id"]
)

panel["y_kev_90d"] = (
    panel["cve_id"]
    .isin(kev_90d_ids)
    .astype(int)
)

positivos = int(
    panel["y_kev_90d"].sum()
)

negativos = len(panel) - positivos

prevalencia = (
    positivos / len(panel) * 100
    if len(panel) > 0
    else 0
)


# =========================
# RESULTADOS
# =========================

print("\n==============================")
print("PANEL PROVISIONAL")
print("==============================")

print(
    "CVEs elegibles:",
    f"{len(panel):,}"
)

print(
    "Positivos KEV a 90 días:",
    positivos
)

print(
    "Negativos:",
    f"{negativos:,}"
)

print(
    "Prevalencia:",
    f"{prevalencia:.5f}%"
)


# =========================
# COMPARACIÓN CON PAPER
# =========================

PAPER_ELEGIBLES = 158076
PAPER_POSITIVOS = 11

print("\n==============================")
print("COMPARACIÓN CON WANG & XU")
print("==============================")

print(
    "Paper elegibles:",
    f"{PAPER_ELEGIBLES:,}"
)

print(
    "Nuestro panel:",
    f"{len(panel):,}"
)

print(
    "Diferencia:",
    f"{len(panel) - PAPER_ELEGIBLES:+,}"
)

print()

print(
    "Paper positivos:",
    PAPER_POSITIVOS
)

print(
    "Nuestros positivos:",
    positivos
)

print(
    "Diferencia:",
    f"{positivos - PAPER_POSITIVOS:+}"
)


# =========================
# GUARDAR
# =========================

panel.to_csv(
    SALIDA,
    index=False
)

print("\nArchivo generado:")
print(SALIDA)