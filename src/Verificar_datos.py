import pandas as pd

epss_path = "../data/epss/epss_scores-2025-10-01.csv.gz"
kev_path = "../data/kev/known_exploited_vulnerabilities.csv"
epss = pd.read_csv(epss_path, compression="gzip", skiprows=1)
kev = pd.read_csv(kev_path)
print("EPSS:")
print(epss.head())
print()
print("Columnas EPSS:", epss.columns.tolist())
print("Cantidad EPSS:", len(epss))
print("\nKEV:")
print(kev.head())
print()
print("Columnas KEV:", kev.columns.tolist())
print("Cantidad KEV:", len(kev))