from pathlib import Path
import math

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


# ============================================================
# ARCHIVOS
# ============================================================

PANELES = {
    "2025-04-01": Path("../outputs/panel_2025-04-01_90d.csv"),
    "2025-07-01": Path("../outputs/panel_2025-07-01_90d.csv"),
    "2025-10-01": Path("../outputs/panel_2025-10-01_90d.csv"),
    "2026-01-01": Path("../outputs/panel_2026-01-01_90d.csv"),
    "2026-03-25": Path("../outputs/panel_2026-03-25_90d.csv"),
}

SALIDA = Path(
    "../outputs/evaluacion_heldout_ml_independiente.csv"
)

TARGET = "y_kev_90d"


# ============================================================
# FEATURES
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
# CONFIGURACIONES SELECCIONADAS POR NUESTRA REPLICACIÓN
# ============================================================

# Nuestra validación sobre octubre seleccionó C=0.1
# tanto para Logistic CVSS como para Logistic Fusion.

LOGISTIC_C_CVSS = 0.1
LOGISTIC_C_FUSION = 0.1


# Nuestra mejor configuración XGBoost CVSS
# en la validación del 01/10/2025:
#
# depth = 3
# trees = 300
# learning_rate = 0.03
# min_child_weight = 1

XGB_CVSS_CONFIG = {
    "max_depth": 3,
    "n_estimators": 300,
    "learning_rate": 0.03,
    "min_child_weight": 1,
}


# Nuestra mejor configuración XGBoost Fusion
# en la validación del 01/10/2025:
#
# depth = 2
# trees = 600
# learning_rate = 0.03
# min_child_weight = 1

XGB_FUSION_CONFIG = {
    "max_depth": 2,
    "n_estimators": 600,
    "learning_rate": 0.03,
    "min_child_weight": 1,
}


# ============================================================
# RESULTADOS DEL PAPER - TABLA III
# Se usan únicamente como referencia comparativa.
# ============================================================

PAPER = {

    # -----------------------
    # ENERO 2026
    # -----------------------

    ("2026-01-01", "CVSS base"): {
        "ap": 0.000994,
        "auc": 0.7740,
        "r1": 10.53,
        "r5": 15.79,
        "r10": 31.58,
    },

    ("2026-01-01", "CVSS exploitability"): {
        "ap": 0.000207,
        "auc": 0.6367,
        "r1": 0.00,
        "r5": 5.26,
        "r10": 10.53,
    },

    ("2026-01-01", "Logistic CVSS"): {
        "ap": 0.000231,
        "auc": 0.6979,
        "r1": 0.00,
        "r5": 0.00,
        "r10": 15.79,
    },

    ("2026-01-01", "XGBoost CVSS"): {
        "ap": 0.000264,
        "auc": 0.7145,
        "r1": 0.00,
        "r5": 15.79,
        "r10": 21.05,
    },

    ("2026-01-01", "EPSS"): {
        "ap": 0.002860,
        "auc": 0.7283,
        "r1": 26.32,
        "r5": 47.37,
        "r10": 57.89,
    },

    ("2026-01-01", "Transparent fusion"): {
        "ap": 0.002860,
        "auc": 0.7283,
        "r1": 26.32,
        "r5": 47.37,
        "r10": 57.89,
    },

    ("2026-01-01", "Logistic fusion"): {
        "ap": 0.001624,
        "auc": 0.7470,
        "r1": 21.05,
        "r5": 47.37,
        "r10": 47.37,
    },

    ("2026-01-01", "XGBoost fusion"): {
        "ap": 0.001280,
        "auc": 0.8392,
        "r1": 15.79,
        "r5": 47.37,
        "r10": 57.89,
    },


    # -----------------------
    # MARZO 2026
    # -----------------------

    ("2026-03-25", "CVSS base"): {
        "ap": 0.000206,
        "auc": 0.6068,
        "r1": 8.33,
        "r5": 8.33,
        "r10": 8.33,
    },

    ("2026-03-25", "CVSS exploitability"): {
        "ap": 0.000076,
        "auc": 0.4956,
        "r1": 0.00,
        "r5": 0.00,
        "r10": 0.00,
    },

    ("2026-03-25", "Logistic CVSS"): {
        "ap": 0.000139,
        "auc": 0.5705,
        "r1": 0.00,
        "r5": 8.33,
        "r10": 16.67,
    },

    ("2026-03-25", "XGBoost CVSS"): {
        "ap": 0.000124,
        "auc": 0.5726,
        "r1": 0.00,
        "r5": 16.67,
        "r10": 25.00,
    },

    ("2026-03-25", "EPSS"): {
        "ap": 0.000783,
        "auc": 0.7884,
        "r1": 8.33,
        "r5": 58.33,
        "r10": 66.67,
    },

    ("2026-03-25", "Transparent fusion"): {
        "ap": 0.000783,
        "auc": 0.7884,
        "r1": 8.33,
        "r5": 58.33,
        "r10": 66.67,
    },

    ("2026-03-25", "Logistic fusion"): {
        "ap": 0.000167,
        "auc": 0.6715,
        "r1": 0.00,
        "r5": 16.67,
        "r10": 25.00,
    },

    ("2026-03-25", "XGBoost fusion"): {
        "ap": 0.001007,
        "auc": 0.7205,
        "r1": 16.67,
        "r5": 41.67,
        "r10": 50.00,
    },
}


# ============================================================
# PREPARACIÓN DE DATOS
# ============================================================

def agregar_epss_logit(df):

    df = df.copy()

    epss = np.clip(
        df["epss"].astype(float),
        1e-6,
        1 - 1e-6,
    )

    df["epss_logit"] = np.log(
        epss / (1 - epss)
    )

    return df


# ============================================================
# PREPROCESAMIENTO LOGISTIC
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
                    handle_unknown="ignore"
                ),
                CATEGORICAL,
            ),
        ]
    )


# ============================================================
# PREPROCESAMIENTO XGBOOST
# ============================================================

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


# ============================================================
# MODELOS LOGISTIC
# ============================================================

def crear_logistic(
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


# ============================================================
# MODELOS XGBOOST
# ============================================================

def crear_xgb(
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

                    max_depth=config[
                        "max_depth"
                    ],

                    n_estimators=config[
                        "n_estimators"
                    ],

                    learning_rate=config[
                        "learning_rate"
                    ],

                    min_child_weight=config[
                        "min_child_weight"
                    ],

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
# MÉTRICAS POR PRESUPUESTO
# ============================================================

def ranking_budget(
    y,
    score,
    budget,
):

    temp = pd.DataFrame(
        {
            "y": np.asarray(y),
            "score": np.asarray(score),
        }
    )

    # Orden estable, igual que en nuestros scripts anteriores.
    temp = temp.sort_values(
        "score",
        ascending=False,
        kind="mergesort",
    )

    n = len(temp)

    k = math.ceil(
        budget * n
    )

    seleccion = temp.iloc[
        :k
    ]

    positivos_totales = int(
        temp["y"].sum()
    )

    hits = int(
        seleccion["y"].sum()
    )

    recall = (
        hits / positivos_totales
        if positivos_totales > 0
        else np.nan
    )

    precision = (
        hits / k
        if k > 0
        else np.nan
    )

    return {
        "k": k,
        "hits": hits,
        "recall": recall,
        "precision": precision,
    }


# ============================================================
# EVALUACIÓN
# ============================================================

def evaluar(
    fecha,
    nombre,
    y,
    score,
):

    ap = average_precision_score(
        y,
        score,
    )

    auc = roc_auc_score(
        y,
        score,
    )

    b1 = ranking_budget(
        y,
        score,
        0.01,
    )

    b5 = ranking_budget(
        y,
        score,
        0.05,
    )

    b10 = ranking_budget(
        y,
        score,
        0.10,
    )

    paper = PAPER.get(
        (fecha, nombre)
    )

    fila = {
        "fecha": fecha,
        "modelo": nombre,

        "ap": ap,
        "roc_auc": auc,

        "r1": (
            b1["recall"] * 100
        ),

        "p1": (
            b1["precision"] * 100
        ),

        "r5": (
            b5["recall"] * 100
        ),

        "p5": (
            b5["precision"] * 100
        ),

        "r10": (
            b10["recall"] * 100
        ),

        "p10": (
            b10["precision"] * 100
        ),

        "hits5": b5["hits"],
        "k5": b5["k"],
    }

    if paper is not None:

        fila["paper_ap"] = (
            paper["ap"]
        )

        fila["paper_auc"] = (
            paper["auc"]
        )

        fila["paper_r5"] = (
            paper["r5"]
        )

        fila["dif_ap"] = (
            ap
            - paper["ap"]
        )

        fila["dif_auc"] = (
            auc
            - paper["auc"]
        )

        fila["dif_r5"] = (
            fila["r5"]
            - paper["r5"]
        )

    return fila


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "EVALUACIÓN HELD-OUT INDEPENDIENTE - WANG & XU"
    )
    print("=" * 78)

    print(
        "Configuraciones ML seleccionadas por nuestra "
        "validación de octubre de 2025."
    )

    paneles = {}

    # --------------------------------------------------------
    # CARGAR LOS CINCO PANELES
    # --------------------------------------------------------

    for fecha, path in PANELES.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Falta el archivo: {path}"
            )

        paneles[fecha] = pd.read_csv(
            path
        )

        positivos = int(
            paneles[fecha][TARGET].sum()
        )

        print(
            f"{fecha}: "
            f"N={len(paneles[fecha]):,}, "
            f"positivos={positivos}"
        )

    # --------------------------------------------------------
    # REFIT:
    # ABRIL + JULIO + OCTUBRE
    # --------------------------------------------------------

    train = pd.concat(
        [
            paneles[
                "2025-04-01"
            ],

            paneles[
                "2025-07-01"
            ],

            paneles[
                "2025-10-01"
            ],
        ],
        ignore_index=True,
    )

    train = agregar_epss_logit(
        train
    )

    enero = agregar_epss_logit(
        paneles[
            "2026-01-01"
        ]
    )

    marzo = agregar_epss_logit(
        paneles[
            "2026-03-25"
        ]
    )

    y_train = (
        train[TARGET]
        .astype(int)
    )

    positivos_train = int(
        y_train.sum()
    )

    negativos_train = (
        len(y_train)
        - positivos_train
    )

    scale_pos_weight = (
        negativos_train
        / positivos_train
    )

    print()
    print("=" * 78)
    print("REFIT")
    print("=" * 78)

    print(
        f"Training N: "
        f"{len(train):,}"
    )

    print(
        f"Training positivos: "
        f"{positivos_train}"
    )

    print(
        "scale_pos_weight: "
        f"{scale_pos_weight:.6f}"
    )

    print()
    print(
        "XGBoost CVSS seleccionado:"
    )

    print(
        XGB_CVSS_CONFIG
    )

    print(
        "XGBoost Fusion seleccionado:"
    )

    print(
        XGB_FUSION_CONFIG
    )

    # ========================================================
    # MATRICES DE ENTRENAMIENTO
    # ========================================================

    X_train_cvss = train[
        NUMERIC_CVSS
        + CATEGORICAL
    ]

    LOGISTIC_FUSION_NUMERIC = (
        NUMERIC_CVSS
        + ["epss_logit"]
    )

    X_train_logfusion = train[
        LOGISTIC_FUSION_NUMERIC
        + CATEGORICAL
    ]

    XGB_FUSION_NUMERIC = (
        NUMERIC_CVSS
        + ["epss"]
    )

    X_train_xgbfusion = train[
        XGB_FUSION_NUMERIC
        + CATEGORICAL
    ]

    # ========================================================
    # CREAR MODELOS
    # ========================================================

    logistic_cvss = crear_logistic(
        C=LOGISTIC_C_CVSS,
        numeric_features=NUMERIC_CVSS,
    )

    logistic_fusion = crear_logistic(
        C=LOGISTIC_C_FUSION,
        numeric_features=(
            LOGISTIC_FUSION_NUMERIC
        ),
    )

    xgb_cvss = crear_xgb(
        config=XGB_CVSS_CONFIG,
        numeric_features=NUMERIC_CVSS,
        scale_pos_weight=(
            scale_pos_weight
        ),
    )

    xgb_fusion = crear_xgb(
        config=XGB_FUSION_CONFIG,
        numeric_features=(
            XGB_FUSION_NUMERIC
        ),
        scale_pos_weight=(
            scale_pos_weight
        ),
    )

    # ========================================================
    # ENTRENAMIENTO
    # ========================================================

    print()
    print(
        "Entrenando Logistic CVSS..."
    )

    logistic_cvss.fit(
        X_train_cvss,
        y_train,
    )

    print(
        "Entrenando Logistic Fusion..."
    )

    logistic_fusion.fit(
        X_train_logfusion,
        y_train,
    )

    print(
        "Entrenando XGBoost CVSS..."
    )

    xgb_cvss.fit(
        X_train_cvss,
        y_train,
    )

    print(
        "Entrenando XGBoost Fusion..."
    )

    xgb_fusion.fit(
        X_train_xgbfusion,
        y_train,
    )

    resultados = []

    # ========================================================
    # EVALUACIÓN HELD-OUT
    # ========================================================

    for fecha, test in [

        (
            "2026-01-01",
            enero,
        ),

        (
            "2026-03-25",
            marzo,
        ),

    ]:

        print()
        print("=" * 78)
        print(
            f"HELD-OUT {fecha}"
        )
        print("=" * 78)

        y = (
            test[TARGET]
            .astype(int)
        )

        X_cvss = test[
            NUMERIC_CVSS
            + CATEGORICAL
        ]

        X_logfusion = test[
            LOGISTIC_FUSION_NUMERIC
            + CATEGORICAL
        ]

        X_xgbfusion = test[
            XGB_FUSION_NUMERIC
            + CATEGORICAL
        ]

        # ----------------------------------------------------
        # SCORES
        # ----------------------------------------------------

        scores = {

            "CVSS base":
                test[
                    "cvss_base"
                ].astype(float),

            "CVSS exploitability":
                test[
                    "cvss_exploitability"
                ].astype(float),

            "EPSS":
                test[
                    "epss"
                ].astype(float),

            # En el paper alpha=1.0.
            # Por tanto la fusión transparente es EPSS.
            "Transparent fusion":
                test[
                    "epss"
                ].astype(float),

            "Logistic CVSS":
                logistic_cvss.predict_proba(
                    X_cvss
                )[:, 1],

            "XGBoost CVSS":
                xgb_cvss.predict_proba(
                    X_cvss
                )[:, 1],

            "Logistic fusion":
                logistic_fusion.predict_proba(
                    X_logfusion
                )[:, 1],

            "XGBoost fusion":
                xgb_fusion.predict_proba(
                    X_xgbfusion
                )[:, 1],
        }

        # ----------------------------------------------------
        # MÉTRICAS
        # ----------------------------------------------------

        for nombre, score in scores.items():

            fila = evaluar(
                fecha,
                nombre,
                y,
                score,
            )

            resultados.append(
                fila
            )

            print()
            print(
                nombre
            )

            print(
                f"AP:      "
                f"{fila['ap']:.9f} "
                f"| paper "
                f"{fila.get('paper_ap', np.nan):.6f}"
            )

            print(
                f"ROC-AUC: "
                f"{fila['roc_auc']:.6f} "
                f"| paper "
                f"{fila.get('paper_auc', np.nan):.4f}"
            )

            print(
                f"R@5%:    "
                f"{fila['r5']:.2f}% "
                f"({fila['hits5']}/"
                f"{int(y.sum())}) "
                f"| paper "
                f"{fila.get('paper_r5', np.nan):.2f}%"
            )

    # ========================================================
    # GUARDAR RESULTADOS
    # ========================================================

    df_resultados = pd.DataFrame(
        resultados
    )

    df_resultados.to_csv(
        SALIDA,
        index=False,
    )

    # ========================================================
    # RESUMEN EPSS VS FUSIÓN
    # ========================================================

    print()
    print("=" * 78)
    print(
        "COMPARACIÓN PRINCIPAL: EPSS VS XGBOOST FUSION"
    )
    print("=" * 78)

    for fecha in [
        "2026-01-01",
        "2026-03-25",
    ]:

        epss = df_resultados[
            (
                df_resultados["fecha"]
                == fecha
            )
            &
            (
                df_resultados["modelo"]
                == "EPSS"
            )
        ].iloc[0]

        fusion = df_resultados[
            (
                df_resultados["fecha"]
                == fecha
            )
            &
            (
                df_resultados["modelo"]
                == "XGBoost fusion"
            )
        ].iloc[0]

        print()
        print(
            fecha
        )

        print(
            f"EPSS:"
            f" AP={epss['ap']:.9f},"
            f" R@5%={epss['r5']:.2f}%"
        )

        print(
            f"XGB Fusion:"
            f" AP={fusion['ap']:.9f},"
            f" R@5%={fusion['r5']:.2f}%"
        )

        if (
            fusion["ap"] > epss["ap"]
            and fusion["r5"] > epss["r5"]
        ):

            conclusion = (
                "La fusión mejora ambas métricas."
            )

        elif (
            fusion["ap"] > epss["ap"]
            or fusion["r5"] > epss["r5"]
        ):

            conclusion = (
                "La mejora no es consistente "
                "entre las métricas."
            )

        else:

            conclusion = (
                "EPSS iguala o supera a la "
                "fusión en ambas métricas."
            )

        print(
            "Conclusión:",
            conclusion
        )

    print()
    print("=" * 78)
    print("FINAL")
    print("=" * 78)

    print(
        "Resultados guardados en:"
    )

    print(
        SALIDA
    )


if __name__ == "__main__":
    main()