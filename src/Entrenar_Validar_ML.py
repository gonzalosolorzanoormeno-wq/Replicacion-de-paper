from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from xgboost import XGBClassifier


# ============================================================
# ARCHIVOS DEFINITIVOS
# ============================================================

PANEL_ABRIL = Path(
    "../outputs/panel_2025-04-01_90d.csv"
)

PANEL_JULIO = Path(
    "../outputs/panel_2025-07-01_90d.csv"
)

PANEL_OCTUBRE = Path(
    "../outputs/panel_2025-10-01_90d.csv"
)

SALIDA = Path(
    "../outputs/validacion_ml_octubre2025.csv"
)

TARGET = "y_kev_90d"


# ============================================================
# CHECKPOINTS YA REPLICADOS
# ============================================================

EXPECTED = {
    "abril": {
        "n": 152_422,
        "positivos": 15,
    },
    "julio": {
        "n": 158_281,
        "positivos": 11,
    },
    "octubre": {
        "n": 158_076,
        "positivos": 11,
    },
}


# ============================================================
# FEATURES DEL PAPER
# ============================================================

NUMERIC_CVSS = [
    "cvss_base",
    "cvss_exploitability",
    "cvss_impact",
]

CATEGORICAL = [
    "base_severity",
    "attack_vector",
    "attack_complexity",
    "privileges_required",
    "user_interaction",
    "scope",
    "confidentiality_impact",
    "integrity_impact",
    "availability_impact",
]


# ============================================================
# LOGISTIC
# ============================================================

LOGISTIC_C = [
    0.1,
    1,
    10,
]


# ============================================================
# XGBOOST - TABLA B2
# ============================================================

XGB_CONFIGS = [
    {
        "max_depth": 2,
        "n_estimators": 300,
        "learning_rate": 0.03,
        "min_child_weight": 1,
    },
    {
        "max_depth": 2,
        "n_estimators": 600,
        "learning_rate": 0.03,
        "min_child_weight": 1,
    },
    {
        "max_depth": 3,
        "n_estimators": 300,
        "learning_rate": 0.03,
        "min_child_weight": 1,
    },
    {
        "max_depth": 3,
        "n_estimators": 600,
        "learning_rate": 0.03,
        "min_child_weight": 5,
    },
    {
        "max_depth": 4,
        "n_estimators": 300,
        "learning_rate": 0.03,
        "min_child_weight": 1,
    },
    {
        "max_depth": 4,
        "n_estimators": 600,
        "learning_rate": 0.05,
        "min_child_weight": 5,
    },
    {
        "max_depth": 5,
        "n_estimators": 300,
        "learning_rate": 0.03,
        "min_child_weight": 5,
    },
    {
        "max_depth": 5,
        "n_estimators": 600,
        "learning_rate": 0.05,
        "min_child_weight": 10,
    },
]


# ============================================================
# RESULTADOS DEL PAPER - TABLA B2
# ============================================================

PAPER_XGB_CVSS_AP = [
    0.000134,
    0.000140,
    0.000139,
    0.000139,
    0.000139,
    0.000141,
    0.000144,
    0.000146,
]

PAPER_XGB_FUSION_AP = [
    0.000713,
    0.000737,
    0.000683,
    0.000571,
    0.000938,
    0.000470,
    0.000767,
    0.000467,
]


# ============================================================
# VALIDACIÓN DE ARCHIVOS
# ============================================================

def verificar_archivos():

    faltantes = []

    for archivo in [
        PANEL_ABRIL,
        PANEL_JULIO,
        PANEL_OCTUBRE,
    ]:

        if not archivo.exists():
            faltantes.append(
                archivo
            )

    if faltantes:

        print("=" * 78)
        print("FALTAN PANELES")
        print("=" * 78)

        for archivo in faltantes:
            print(
                f"- {archivo}"
            )

        return False

    return True


def verificar_panel(
    nombre,
    df,
):

    esperado = EXPECTED[
        nombre
    ]

    n = len(
        df
    )

    positivos = int(
        df[TARGET].sum()
    )

    print(
        f"{nombre.capitalize():<10} "
        f"N={n:,} "
        f"| positivos={positivos}"
    )

    if (
        n != esperado["n"]
        or
        positivos != esperado["positivos"]
    ):

        raise RuntimeError(
            f"\nEl panel de {nombre} "
            f"no coincide con el checkpoint "
            f"replicado.\n"
            f"Esperado: "
            f"N={esperado['n']:,}, "
            f"positivos="
            f"{esperado['positivos']}\n"
            f"Actual: "
            f"N={n:,}, "
            f"positivos={positivos}"
        )


# ============================================================
# LOGIT EPSS
# ============================================================

def agregar_epss_logit(
    df
):

    df = df.copy()

    epss_clip = np.clip(
        df["epss"].astype(float),
        1e-6,
        1 - 1e-6,
    )

    df["epss_logit"] = np.log(
        epss_clip
        /
        (
            1
            - epss_clip
        )
    )

    return df


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def make_logistic_preprocessor(
    numeric_features
):

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                CATEGORICAL,
            ),
        ]
    )


def make_xgb_preprocessor(
    numeric_features
):

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                CATEGORICAL,
            ),
        ]
    )


# ============================================================
# MODELOS
# ============================================================

def make_logistic(
    C,
    numeric_features,
):

    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_logistic_preprocessor(
                    numeric_features
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def make_xgb(
    config,
    numeric_features,
    scale_pos_weight,
):

    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_xgb_preprocessor(
                    numeric_features
                ),
            ),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",

                    max_depth=(
                        config[
                            "max_depth"
                        ]
                    ),

                    n_estimators=(
                        config[
                            "n_estimators"
                        ]
                    ),

                    learning_rate=(
                        config[
                            "learning_rate"
                        ]
                    ),

                    min_child_weight=(
                        config[
                            "min_child_weight"
                        ]
                    ),

                    subsample=0.8,

                    colsample_bytree=0.8,

                    reg_lambda=1.0,

                    scale_pos_weight=(
                        scale_pos_weight
                    ),

                    random_state=42,

                    n_jobs=-1,

                    eval_metric="logloss",

                    verbosity=0,
                ),
            ),
        ]
    )


# ============================================================
# EVALUACIÓN
# ============================================================

def calcular_ap(
    modelo,
    X_train,
    y_train,
    X_val,
    y_val,
):

    modelo.fit(
        X_train,
        y_train,
    )

    scores = (
        modelo.predict_proba(
            X_val
        )[:, 1]
    )

    return average_precision_score(
        y_val,
        scores,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not verificar_archivos():
        return

    print("=" * 78)
    print(
        "REPLICACIÓN ML - WANG & XU"
    )
    print(
        "ABRIL + JULIO -> OCTUBRE"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # CARGAR PANELES
    # --------------------------------------------------------

    abril = pd.read_csv(
        PANEL_ABRIL
    )

    julio = pd.read_csv(
        PANEL_JULIO
    )

    octubre = pd.read_csv(
        PANEL_OCTUBRE
    )

    print(
        "\nCHECKPOINT DE PANELES"
    )

    verificar_panel(
        "abril",
        abril,
    )

    verificar_panel(
        "julio",
        julio,
    )

    verificar_panel(
        "octubre",
        octubre,
    )

    # --------------------------------------------------------
    # DEVELOPMENT
    # --------------------------------------------------------

    development = pd.concat(
        [
            abril,
            julio,
        ],
        ignore_index=True,
    )

    validation = (
        octubre.copy()
    )

    development = (
        agregar_epss_logit(
            development
        )
    )

    validation = (
        agregar_epss_logit(
            validation
        )
    )

    print()
    print("=" * 78)
    print(
        "DESARROLLO Y VALIDACIÓN"
    )
    print("=" * 78)

    print(
        f"Development N: "
        f"{len(development):,}"
    )

    print(
        "Development positivos: "
        f"{int(development[TARGET].sum())}"
    )

    print(
        f"Validation N: "
        f"{len(validation):,}"
    )

    print(
        "Validation positivos: "
        f"{int(validation[TARGET].sum())}"
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    y_dev = (
        development[
            TARGET
        ]
        .astype(int)
    )

    y_val = (
        validation[
            TARGET
        ]
        .astype(int)
    )

    positivos = int(
        y_dev.sum()
    )

    negativos = (
        len(y_dev)
        - positivos
    )

    scale_pos_weight = (
        negativos
        /
        positivos
    )

    print(
        "\nscale_pos_weight:"
    )

    print(
        f"{scale_pos_weight:.6f}"
    )

    resultados = []

    # ========================================================
    # LOGISTIC CVSS
    # ========================================================

    print()
    print("=" * 78)
    print(
        "LOGISTIC CVSS-ONLY"
    )
    print("=" * 78)

    X_dev_cvss = development[
        NUMERIC_CVSS
        + CATEGORICAL
    ]

    X_val_cvss = validation[
        NUMERIC_CVSS
        + CATEGORICAL
    ]

    for C in LOGISTIC_C:

        modelo = make_logistic(
            C=C,
            numeric_features=(
                NUMERIC_CVSS
            ),
        )

        ap = calcular_ap(
            modelo,
            X_dev_cvss,
            y_dev,
            X_val_cvss,
            y_val,
        )

        print(
            f"C={C:<4} "
            f"AP={ap:.9f}"
        )

        resultados.append(
            {
                "modelo":
                    "Logistic CVSS",

                "config":
                    f"C={C}",

                "validation_ap":
                    ap,

                "paper_ap":
                    np.nan,
            }
        )

    # ========================================================
    # LOGISTIC FUSION
    # ========================================================

    print()
    print("=" * 78)
    print(
        "LOGISTIC FUSION"
    )
    print("=" * 78)

    NUMERIC_LOGISTIC_FUSION = (
        NUMERIC_CVSS
        + [
            "epss_logit"
        ]
    )

    X_dev_logfusion = (
        development[
            NUMERIC_LOGISTIC_FUSION
            + CATEGORICAL
        ]
    )

    X_val_logfusion = (
        validation[
            NUMERIC_LOGISTIC_FUSION
            + CATEGORICAL
        ]
    )

    for C in LOGISTIC_C:

        modelo = make_logistic(
            C=C,
            numeric_features=(
                NUMERIC_LOGISTIC_FUSION
            ),
        )

        ap = calcular_ap(
            modelo,
            X_dev_logfusion,
            y_dev,
            X_val_logfusion,
            y_val,
        )

        print(
            f"C={C:<4} "
            f"AP={ap:.9f}"
        )

        resultados.append(
            {
                "modelo":
                    "Logistic fusion",

                "config":
                    f"C={C}",

                "validation_ap":
                    ap,

                "paper_ap":
                    np.nan,
            }
        )

    # ========================================================
    # XGBOOST CVSS
    # ========================================================

    print()
    print("=" * 78)
    print(
        "XGBOOST CVSS-ONLY"
    )
    print("=" * 78)

    for i, config in enumerate(
        XGB_CONFIGS
    ):

        modelo = make_xgb(
            config=config,
            numeric_features=(
                NUMERIC_CVSS
            ),
            scale_pos_weight=(
                scale_pos_weight
            ),
        )

        ap = calcular_ap(
            modelo,
            X_dev_cvss,
            y_dev,
            X_val_cvss,
            y_val,
        )

        paper_ap = (
            PAPER_XGB_CVSS_AP[
                i
            ]
        )

        diferencia = (
            ap
            - paper_ap
        )

        config_text = (
            f"depth="
            f"{config['max_depth']}, "
            f"trees="
            f"{config['n_estimators']}, "
            f"lr="
            f"{config['learning_rate']}, "
            f"child="
            f"{config['min_child_weight']}"
        )

        print(
            f"\n{i + 1}. "
            f"{config_text}"
        )

        print(
            f"   Réplica: "
            f"{ap:.9f}"
        )

        print(
            f"   Paper:   "
            f"{paper_ap:.9f}"
        )

        print(
            f"   Dif:     "
            f"{diferencia:+.9f}"
        )

        resultados.append(
            {
                "modelo":
                    "XGBoost CVSS",

                "config":
                    config_text,

                "validation_ap":
                    ap,

                "paper_ap":
                    paper_ap,
            }
        )

    # ========================================================
    # XGBOOST FUSION
    # ========================================================

    print()
    print("=" * 78)
    print(
        "XGBOOST FUSION"
    )
    print("=" * 78)

    NUMERIC_XGB_FUSION = (
        NUMERIC_CVSS
        + [
            "epss"
        ]
    )

    X_dev_xgbfusion = (
        development[
            NUMERIC_XGB_FUSION
            + CATEGORICAL
        ]
    )

    X_val_xgbfusion = (
        validation[
            NUMERIC_XGB_FUSION
            + CATEGORICAL
        ]
    )

    for i, config in enumerate(
        XGB_CONFIGS
    ):

        modelo = make_xgb(
            config=config,
            numeric_features=(
                NUMERIC_XGB_FUSION
            ),
            scale_pos_weight=(
                scale_pos_weight
            ),
        )

        ap = calcular_ap(
            modelo,
            X_dev_xgbfusion,
            y_dev,
            X_val_xgbfusion,
            y_val,
        )

        paper_ap = (
            PAPER_XGB_FUSION_AP[
                i
            ]
        )

        diferencia = (
            ap
            - paper_ap
        )

        config_text = (
            f"depth="
            f"{config['max_depth']}, "
            f"trees="
            f"{config['n_estimators']}, "
            f"lr="
            f"{config['learning_rate']}, "
            f"child="
            f"{config['min_child_weight']}"
        )

        print(
            f"\n{i + 1}. "
            f"{config_text}"
        )

        print(
            f"   Réplica: "
            f"{ap:.9f}"
        )

        print(
            f"   Paper:   "
            f"{paper_ap:.9f}"
        )

        print(
            f"   Dif:     "
            f"{diferencia:+.9f}"
        )

        resultados.append(
            {
                "modelo":
                    "XGBoost fusion",

                "config":
                    config_text,

                "validation_ap":
                    ap,

                "paper_ap":
                    paper_ap,
            }
        )

    # ========================================================
    # GUARDAR
    # ========================================================

    resultados_df = (
        pd.DataFrame(
            resultados
        )
    )

    resultados_df[
        "diferencia_vs_paper"
    ] = (
        resultados_df[
            "validation_ap"
        ]
        -
        resultados_df[
            "paper_ap"
        ]
    )

    resultados_df.to_csv(
        SALIDA,
        index=False,
    )

    # ========================================================
    # MEJORES
    # ========================================================

    print()
    print("=" * 78)
    print(
        "MEJORES CONFIGURACIONES"
    )
    print("=" * 78)

    for nombre in [
        "Logistic CVSS",
        "Logistic fusion",
        "XGBoost CVSS",
        "XGBoost fusion",
    ]:

        subset = (
            resultados_df[
                resultados_df[
                    "modelo"
                ]
                == nombre
            ]
        )

        indice = (
            subset[
                "validation_ap"
            ].idxmax()
        )

        mejor = (
            resultados_df.loc[
                indice
            ]
        )

        print(
            f"\n{nombre}"
        )

        print(
            f"Config: "
            f"{mejor['config']}"
        )

        print(
            f"AP: "
            f"{mejor['validation_ap']:.9f}"
        )

    # ========================================================
    # CHECKPOINT DEL PAPER
    # ========================================================

    print()
    print("=" * 78)
    print(
        "CHECKPOINT WANG & XU"
    )
    print("=" * 78)

    print(
        "\nXGBoost CVSS esperado:"
    )

    print(
        "depth=5, trees=600, "
        "lr=0.05, child=10"
    )

    print(
        "AP = 0.000146"
    )

    print(
        "\nXGBoost fusion esperado:"
    )

    print(
        "depth=4, trees=300, "
        "lr=0.03, child=1"
    )

    print(
        "AP = 0.000938"
    )

    print()
    print(
        "Archivo generado:"
    )

    print(
        SALIDA
    )


if __name__ == "__main__":
    main()