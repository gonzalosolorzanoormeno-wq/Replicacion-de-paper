from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


PANEL_PATH = Path(
    "../outputs/panel_2025-10-01_PROVISIONAL.csv"
)


NUMERIC_FEATURES = [
    "cvss_base",
    "cvss_exploitability",
    "cvss_impact",
]


CATEGORICAL_FEATURES = [
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


TARGET = "y_kev_90d"


def main():

    df = pd.read_csv(
        PANEL_PATH
    )

    print("=" * 70)
    print("PREPARACIÓN DEL PIPELINE ML")
    print("=" * 70)

    print(
        f"\nRegistros: {len(df):,}"
    )

    print(
        f"Positivos: "
        f"{int(df[TARGET].sum())}"
    )

    print("\nFeatures numéricas:")

    for col in NUMERIC_FEATURES:
        print(
            f"  {col}: "
            f"missing={df[col].isna().sum():,}"
        )

    print("\nFeatures categóricas:")

    for col in CATEGORICAL_FEATURES:

        print(
            f"  {col}: "
            f"missing={df[col].isna().sum():,} | "
            f"niveles={df[col].nunique(dropna=False)}"
        )

    # ------------------------------------
    # Preprocesamiento del paper
    # ------------------------------------

    numeric_transformer = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES
            ),
        ]
    )

    # ------------------------------------
    # Logistic Regression CVSS-only
    # ------------------------------------

    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42
                )
            ),
        ]
    )

    print("\n" + "=" * 70)
    print("PIPELINE CONSTRUIDO CORRECTAMENTE")
    print("=" * 70)

    print(
        "\nLogistic CVSS-only preparado."
    )

    print(
        "Todavía NO se entrena:"
    )

    print(
        "abril + julio serán desarrollo;"
    )

    print(
        "octubre será validación."
    )

    print(
        "\nValores C que evaluará el paper:"
    )

    print(
        "[0.1, 1, 10]"
    )


if __name__ == "__main__":
    main()
    