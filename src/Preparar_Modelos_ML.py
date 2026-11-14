import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier


PANEL_PATH = Path(
    "../outputs/panel_2025-10-01_PROVISIONAL.csv"
)

TARGET = "y_kev_90d"

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

# Grid EXACTO de XGBoost reportado en el paper
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
                    handle_unknown="ignore"
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
                    handle_unknown="ignore"
                ),
                CATEGORICAL,
            ),
        ]
    )


def make_logistic_cvss(C):

    preprocessor = (
        make_logistic_preprocessor(
            NUMERIC_CVSS
        )
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
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


def make_logistic_fusion(C):

    # El paper usa una variante logística
    # con logit(EPSS).
    numeric_features = (
        NUMERIC_CVSS
        + ["epss_logit"]
    )

    preprocessor = (
        make_logistic_preprocessor(
            numeric_features
        )
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
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
    scale_pos_weight
):

    preprocessor = (
        make_xgb_preprocessor(
            numeric_features
        )
    )

    model = XGBClassifier(
        objective="binary:logistic",

        max_depth=(
            config["max_depth"]
        ),

        n_estimators=(
            config["n_estimators"]
        ),

        learning_rate=(
            config["learning_rate"]
        ),

        min_child_weight=(
            config["min_child_weight"]
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
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            ),
        ]
    )


def main():

    df = pd.read_csv(
        PANEL_PATH
    )

    # --------------------------
    # logit(EPSS)
    # --------------------------

    epss_clip = np.clip(
        df["epss"].astype(float),
        1e-6,
        1 - 1e-6,
    )

    df["epss_logit"] = np.log(
        epss_clip
        / (1 - epss_clip)
    )

    print("=" * 72)
    print("PREPARACIÓN DE MODELOS ML")
    print("=" * 72)

    print(
        f"\nPanel diagnóstico: "
        f"{len(df):,}"
    )

    print(
        f"Positivos: "
        f"{int(df[TARGET].sum())}"
    )

    print("\nEPSS:")

    print(
        f"  mínimo: "
        f"{df['epss'].min()}"
    )

    print(
        f"  máximo: "
        f"{df['epss'].max()}"
    )

    print("\nlogit(EPSS):")

    print(
        f"  mínimo: "
        f"{df['epss_logit'].min():.6f}"
    )

    print(
        f"  máximo: "
        f"{df['epss_logit'].max():.6f}"
    )

    print(
        f"  infinitos: "
        f"{np.isinf(df['epss_logit']).sum()}"
    )

    print(
        f"  missing: "
        f"{df['epss_logit'].isna().sum()}"
    )

    # --------------------------
    # Logistic
    # --------------------------

    print("\n" + "=" * 72)
    print("LOGISTIC REGRESSION")
    print("=" * 72)

    for C in [0.1, 1, 10]:

        make_logistic_cvss(C)

        make_logistic_fusion(C)

        print(
            f"C={C}: "
            f"CVSS-only OK | "
            f"Fusion OK"
        )

    # --------------------------
    # XGBoost
    # --------------------------

    print("\n" + "=" * 72)
    print("XGBOOST — 8 CONFIGURACIONES")
    print("=" * 72)

    # IMPORTANTE:
    # Esto es solo para comprobar que
    # los modelos se construyen.
    #
    # El scale_pos_weight REAL se
    # calculará con Abril + Julio.

    diagnostic_spw = 1.0

    for i, config in enumerate(
        XGB_CONFIGS,
        start=1
    ):

        make_xgb(
            config,
            NUMERIC_CVSS,
            diagnostic_spw,
        )

        make_xgb(
            config,
            NUMERIC_CVSS + ["epss"],
            diagnostic_spw,
        )

        print(
            f"{i}. "
            f"depth={config['max_depth']} | "
            f"trees={config['n_estimators']} | "
            f"lr={config['learning_rate']} | "
            f"child={config['min_child_weight']} "
            f"→ CVSS-only OK | Fusion OK"
        )

    print("\n" + "=" * 72)
    print("RESULTADO")
    print("=" * 72)

    print(
        "\nModelos preparados correctamente."
    )

    print(
        "NO se realizó entrenamiento."
    )

    print(
        "\nPendiente para entrenamiento fiel:"
    )

    print(
        "1. Panel 2025-04-01"
    )

    print(
        "2. Panel 2025-07-01"
    )

    print(
        "3. Pool abril + julio"
    )

    print(
        "4. Calcular scale_pos_weight"
    )

    print(
        "5. Entrenar con desarrollo"
    )

    print(
        "6. Validar en octubre"
    )


if __name__ == "__main__":
    main()
    