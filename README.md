# Replicación de Wang & Xu (2026)

Replicación independiente del artículo:

**Wang, Y. & Xu, Y. (2026). _Do Machine-Learning Fusion Models Improve Vulnerability Prioritization? A Time-Aligned Budget-Aware Evaluation of CVSS, EPSS, and CISA KEV_. IEEE Access.**

DOI: `10.1109/ACCESS.2026.3745734`

## Objetivo

El objetivo de este repositorio es replicar la metodología y los principales resultados experimentales del artículo de Wang y Xu, que compara estrategias de priorización de vulnerabilidades basadas en:

- CVSS
- EPSS
- CISA KEV
- Regresión logística
- XGBoost
- Modelos de fusión CVSS + EPSS

La evaluación se realiza mediante paneles temporales, utilizando únicamente información disponible en cada fecha de decisión.

---

## Diseño temporal

Se reconstruyeron las cinco fechas utilizadas en el artículo:

| Fecha | Rol | CVEs elegibles | Positivos KEV a 90 días |
|---|---|---:|---:|
| 2025-04-01 | Desarrollo 1 | 152,422 | 15 |
| 2025-07-01 | Desarrollo 2 | 158,281 | 11 |
| 2025-10-01 | Validación | 158,076 | 11 |
| 2026-01-01 | Held-out 1 | 165,222 | 19 |
| 2026-03-25 | Held-out 2 | 171,188 | 12 |

Los cinco paneles fueron reconstruidos de forma independiente y sus tamaños y cantidades de positivos coinciden exactamente con los reportados por Wang y Xu.

---

## Fuentes de datos

### EPSS

Se utilizaron los archivos históricos diarios de EPSS correspondientes a cada fecha de decisión.

### CISA KEV

El catálogo Known Exploited Vulnerabilities de CISA se utiliza para:

1. excluir vulnerabilidades que ya pertenecían a KEV en la fecha de decisión;
2. identificar incorporaciones futuras dentro de la ventana de 90 días.

### NVD histórico

Para evitar utilizar información CVSS publicada después de cada fecha de decisión, se reconstruyó el estado histórico de NVD mediante el repositorio:

`olbat/nvdcve`

Los commits utilizados fueron:

| Fecha | Commit |
|---|---|
| 2025-04-01 | `e8af36960333a78e61e91796c644437f8ccb5ff1` |
| 2025-07-01 | `436a9fad9aefe3bc615cdc661d66209473720fb0` |
| 2025-10-01 | `3d1d5e8d036dbfe52d5b4efb837b3e8669e84571` |
| 2026-01-01 | `2c076a9033a69a3a058e8397538d421a8d7acc32` |
| 2026-03-25 | `d4c05f32b5b29bd8b033b5e227ee49d194044712` |

Solo se conservaron métricas CVSS v3.0/v3.1 atribuibles a NVD.

---

## Verificación del NVD histórico

La reconstrucción produjo los mismos conteos de disponibilidad CVSS reportados en el artículo:

| Fecha | Registros NVD | CVSS v3 NVD | Sin CVSS v3 | Missing |
|---|---:|---:|---:|---:|
| 2025-04-01 | 194,946 | 153,407 | 41,539 | 21.31% |
| 2025-07-01 | 207,134 | 159,310 | 47,824 | 23.09% |
| 2025-10-01 | 219,164 | 159,105 | 60,059 | 27.40% |
| 2026-01-01 | 232,283 | 166,273 | 66,010 | 28.42% |
| 2026-03-25 | 246,836 | 172,283 | 74,553 | 30.20% |

---

## Replicación de baselines

Los resultados de EPSS y CVSS base fueron reproducidos prácticamente de forma exacta.

| Fecha | EPSS AP | Paper | EPSS R@5% | Paper | CVSS AP | Paper | CVSS R@5% | Paper |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-04-01 | 0.001150 | 0.001150 | 40.00% | 40.00% | 0.000139 | 0.000139 | 0.00% | 0.00% |
| 2025-07-01 | 0.011416 | 0.011416 | 63.64% | 63.64% | 0.000128 | 0.000128 | 0.00% | 0.00% |
| 2025-10-01 | 0.022486 | 0.022486 | 54.55% | 54.55% | 0.000192 | 0.000192 | 9.09% | 9.09% |
| 2026-01-01 | 0.002860 | 0.002860 | 47.37% | 47.37% | 0.000994 | 0.000994 | 15.79% | 15.79% |
| 2026-03-25 | 0.000783 | 0.000783 | 58.33% | 58.33% | 0.000206 | 0.000206 | 8.33% | 8.33% |

---

## Modelos de Machine Learning

Los modelos utilizan las siguientes variables CVSS:

### Numéricas

- Base Score
- Exploitability Score
- Impact Score

### Categóricas

- Base Severity
- Attack Vector
- Attack Complexity
- Privileges Required
- User Interaction
- Scope
- Confidentiality Impact
- Integrity Impact
- Availability Impact

Para los modelos de fusión se añade EPSS.

En Logistic Fusion se utiliza:

`logit(EPSS)`

con EPSS limitado al intervalo `[1e-6, 1 - 1e-6]`.

---

## División experimental

La división temporal utilizada es la misma del artículo:

### Desarrollo

- 2025-04-01
- 2025-07-01

### Validación

- 2025-10-01

### Refit final

Después de seleccionar los modelos, se combinan:

- abril
- julio
- octubre

Total de entrenamiento final:

- `N = 468,779`
- positivos = `37`

### Held-out

Los modelos se evalúan finalmente sobre fechas no utilizadas para su selección:

- 2026-01-01
- 2026-03-25

---

## Validación de XGBoost

La búsqueda de hiperparámetros fue replicada utilizando las ocho configuraciones especificadas por Wang y Xu.

La replicación independiente produjo pequeñas diferencias respecto a los AP de validación publicados.

### Configuraciones seleccionadas por nuestra validación

**XGBoost CVSS**

- max_depth = 3
- n_estimators = 300
- learning_rate = 0.03
- min_child_weight = 1

**XGBoost Fusion**

- max_depth = 2
- n_estimators = 600
- learning_rate = 0.03
- min_child_weight = 1

Las configuraciones seleccionadas por el artículo fueron diferentes, por lo que se realizaron dos evaluaciones finales.

---

## Evaluación confirmatoria

En esta evaluación se utilizaron las configuraciones XGBoost seleccionadas y publicadas por Wang y Xu.

### Enero de 2026

| Modelo | AP réplica | AP paper | R@5% réplica | R@5% paper |
|---|---:|---:|---:|---:|
| EPSS | 0.002860 | 0.002860 | 47.37% | 47.37% |
| Logistic Fusion | 0.001853 | 0.001624 | 47.37% | 47.37% |
| XGBoost Fusion | 0.000898 | 0.001280 | 42.11% | 47.37% |

### Marzo de 2026

| Modelo | AP réplica | AP paper | R@5% réplica | R@5% paper |
|---|---:|---:|---:|---:|
| EPSS | 0.000783 | 0.000783 | 58.33% | 58.33% |
| Logistic Fusion | 0.000163 | 0.000167 | 16.67% | 16.67% |
| XGBoost Fusion | 0.000754 | 0.001007 | 41.67% | 41.67% |

Los modelos XGBoost presentan diferencias numéricas, aunque conservan el comportamiento general observado en el artículo.

---

## Evaluación independiente

También se realizó una segunda evaluación empleando únicamente los hiperparámetros seleccionados por nuestra propia validación de octubre.

### Enero de 2026

**EPSS**

- AP = `0.002860`
- Recall@5% = `47.37%` (9/19)

**XGBoost Fusion**

- AP = `0.002049`
- Recall@5% = `47.37%` (9/19)

EPSS obtiene mayor AP y empata en Recall@5%.

### Marzo de 2026

**EPSS**

- AP = `0.000783`
- Recall@5% = `58.33%` (7/12)

**XGBoost Fusion**

- AP = `0.000611`
- Recall@5% = `33.33%` (4/12)

EPSS supera a XGBoost Fusion en ambas métricas.

---

## Resultado principal de la replicación

La reconstrucción independiente confirma la conclusión principal de Wang y Xu:

> La incorporación de CVSS mediante modelos de fusión de Machine Learning no produce una mejora consistente sobre EPSS para priorizar vulnerabilidades que posteriormente son incorporadas al catálogo CISA KEV.

Los resultados muestran que:

- EPSS reproduce exactamente los resultados directos del artículo.
- CVSS base reproduce exactamente los principales resultados reportados.
- la fusión transparente selecciona `alpha = 1.0`, por lo que termina siendo equivalente a EPSS;
- Logistic Fusion no mejora consistentemente a EPSS;
- XGBoost Fusion presenta sensibilidad a la configuración y al entrenamiento;
- las diferencias en modelos ML no modifican la conclusión principal del artículo.

---

## Diferencias de reproducibilidad

No todos los resultados de Machine Learning pudieron reproducirse de forma numéricamente idéntica.

Las principales diferencias se observaron en:

- selección de configuraciones XGBoost durante validación;
- AP de XGBoost Fusion;
- algunos cortes Recall@5%;
- Logistic CVSS.

Estas diferencias se documentan explícitamente y no fueron corregidas mediante ajuste posterior sobre los conjuntos held-out.

Esto evita introducir información futura en el proceso de selección y mantiene la independencia de la replicación.

---

## Entorno

El artículo reporta:

- Python 3.12.13
- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0
- XGBoost 3.1.3
- SciPy 1.17.0
- Matplotlib 3.10.8

La réplica se ejecutó en Windows utilizando Python 3.12.10 y las mismas versiones principales de las librerías reportadas por el artículo.

El archivo `requirements.txt` contiene las dependencias utilizadas.

---

## Estructura

```text
replica_wang_xu/
│
├── data/
│   ├── epss/
│   ├── kev/
│   └── nvd/
│
├── outputs/
│   ├── validacion_ml_octubre2025.csv
│   ├── evaluacion_heldout_ml.csv
│   ├── evaluacion_heldout_ml_independiente.csv
│   ├── budgets_octubre2025.csv
│   └── fusion_transparente_octubre2025.csv
│
├── src/
│   ├── Extraer_nvd_historico.py
│   ├── construir_panel.py
│   ├── Entrenar_Validar_ML.py
│   ├── Evaluar_Heldout_ML.py
│   ├── Evaluar_Heldout_ML_Independiente.py
│   ├── Validar_Fusion_Transparente_Octubre2025.py
│   └── Validar_Budgets_Octubre2025.py
│
├── requirements.txt
├── .gitignore
└── README.md







