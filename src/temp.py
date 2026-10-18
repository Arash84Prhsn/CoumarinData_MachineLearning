import json
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabulate import tabulate


DATA_PATH = "Data/Processed_Data.csv"
RESULTS_DIR = "Results/Regression/"
PARAMETERS_DIR = RESULTS_DIR + "BestParameters/"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PARAMETERS_DIR, exist_ok=True)


CELL_LINE_THRESHOLD = 10


TRAIN_COUMARINS = [
    "auraptene",
    "esculetin",
    "galbanicacid",
    "herniarin",
    "osthole",
    "umbelliferone",
    "umbelliprenin",
]

EXTERNAL_COUMARINS = [
    "scopoletin",
    "imperatorin",
]


categorical_features = [
    "CellLine",
]


numerical_features = [
    "LogDose",
    "Time",
    "MW",
    "LogP",
    "NumHDonors",
    "NumHAcceptors",
    "TPSA",
    "RingCounts",
    "fCsp3",
]


morgan_features = [
    f"Morgan_{i}"
    for i in range(2048)
]


data = pd.read_csv(DATA_PATH)


train_data = data[
    data["Coumarin"].isin(TRAIN_COUMARINS)
].copy()

test_data = data[
    data["Coumarin"].isin(EXTERNAL_COUMARINS)
].copy()


cell_line_counts = train_data["CellLine"].value_counts()

test_data["CellLineTrainingCount"] = (
    test_data["CellLine"]
    .map(cell_line_counts)
    .fillna(0)
)


test_case_1 = test_data[
    test_data["CellLineTrainingCount"] >= CELL_LINE_THRESHOLD
].copy()

test_case_2 = test_data[
    test_data["CellLineTrainingCount"] < CELL_LINE_THRESHOLD
].copy()


test_case_1 = test_case_1.drop(
    columns=["CellLineTrainingCount"]
)

test_case_2 = test_case_2.drop(
    columns=["CellLineTrainingCount"]
)


print("=" * 100)
print("Morgan Fingerprint + Biological Features PLS")
print("=" * 100)

print(f"Training rows: {len(train_data)}")
print(f"External test rows: {len(test_data)}")
print(f"External Case 1 rows: {len(test_case_1)}")
print(f"External Case 2 rows: {len(test_case_2)}")

print()
print("Training coumarins:")
print(sorted(train_data["Coumarin"].unique()))

print()
print("External coumarins:")
print(sorted(test_data["Coumarin"].unique()))


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),
        "MedAE": median_absolute_error(
            y_true,
            y_pred,
        ),
        "R2": r2_score(
            y_true,
            y_pred,
        ),
    }


def prepare_fold_data(
    fold_train,
    fold_validation,
):
    """
    Prepare features for one LOCO fold.

    Morgan bits are filtered using ONLY fold_train.
    """

    train_morgan = fold_train[
        morgan_features
    ]

    validation_morgan = fold_validation[
        morgan_features
    ]

    variable_morgan_features = [
        column
        for column in morgan_features
        if train_morgan[column].nunique() > 1
    ]

    X_train_base = fold_train[
        numerical_features + categorical_features
    ].copy()

    X_validation_base = fold_validation[
        numerical_features + categorical_features
    ].copy()

    X_train_morgan = train_morgan[
        variable_morgan_features
    ].copy()

    X_validation_morgan = validation_morgan[
        variable_morgan_features
    ].copy()

    return (
        X_train_base,
        X_validation_base,
        X_train_morgan,
        X_validation_morgan,
        variable_morgan_features,
    )


def transform_base_features(
    X_train,
    X_validation,
):
    """
    Standardize numerical features and one-hot encode CellLine.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                numerical_features,
            ),
            (
                "cellline",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    X_validation_transformed = (
        preprocessor.transform(
            X_validation
        )
    )

    return (
        X_train_transformed,
        X_validation_transformed,
        preprocessor,
    )


def transform_morgan_features(
    X_train,
    X_validation,
):
    """
    Standardize Morgan features.

    The feature set has already been filtered using
    the training fold only.
    """

    if X_train.shape[1] == 0:
        return (
            np.empty(
                (len(X_train), 0)
            ),
            np.empty(
                (len(X_validation), 0)
            ),
            None,
        )

    scaler = StandardScaler()

    X_train_transformed = scaler.fit_transform(
        X_train
    )

    X_validation_transformed = (
        scaler.transform(
            X_validation
        )
    )

    return (
        X_train_transformed,
        X_validation_transformed,
        scaler,
    )


logo = LeaveOneGroupOut()

groups = train_data["Coumarin"]

component_candidates = [
    1,
    2,
    3,
    4,
]


loco_results = []

best_components = None
best_loco_r2 = -np.inf


print()
print("=" * 100)
print("LOCO COMPONENT SEARCH")
print("=" * 100)


for n_components in component_candidates:
    fold_scores = []

    print()
    print(
        f"Testing {n_components} PLS component(s)"
    )

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        logo.split(
            train_data,
            groups=groups,
        ),
        start=1,
    ):
        fold_train = train_data.iloc[
            train_indices
        ].copy()

        fold_validation = train_data.iloc[
            validation_indices
        ].copy()

        held_out_coumarin = (
            fold_validation["Coumarin"]
            .iloc[0]
        )

        (
            X_train_base,
            X_validation_base,
            X_train_morgan,
            X_validation_morgan,
            variable_morgan_features,
        ) = prepare_fold_data(
            fold_train,
            fold_validation,
        )

        (
            X_train_base_transformed,
            X_validation_base_transformed,
            base_preprocessor,
        ) = transform_base_features(
            X_train_base,
            X_validation_base,
        )

        (
            X_train_morgan_transformed,
            X_validation_morgan_transformed,
            morgan_scaler,
        ) = transform_morgan_features(
            X_train_morgan,
            X_validation_morgan,
        )

        X_train = np.hstack([
            X_train_base_transformed,
            X_train_morgan_transformed,
        ])

        X_validation = np.hstack([
            X_validation_base_transformed,
            X_validation_morgan_transformed,
        ])

        y_train = fold_train[
            "Viability"
        ].to_numpy()

        y_validation = fold_validation[
            "Viability"
        ].to_numpy()

        max_components = min(
            n_components,
            X_train.shape[1],
            len(fold_train) - 1,
        )

        model = PLSRegression(
            n_components=max_components,
            scale=False,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        ).ravel()

        fold_r2 = r2_score(
            y_validation,
            predictions,
        )

        fold_scores.append(
            fold_r2
        )

        print(
            f"  Fold {fold_number}/7 | "
            f"Left out: {held_out_coumarin:<16} | "
            f"Morgan features: "
            f"{len(variable_morgan_features):4d} | "
            f"R²: {fold_r2:.6f}"
        )

    mean_loco_r2 = np.mean(
        fold_scores
    )

    std_loco_r2 = np.std(
        fold_scores
    )

    loco_results.append({
        "Components": n_components,
        "LOCO R2": mean_loco_r2,
        "LOCO R2 Std": std_loco_r2,
    })

    print(
        f"  Mean LOCO R²: {mean_loco_r2:.6f}"
    )

    print(
        f"  Std LOCO R²:  {std_loco_r2:.6f}"
    )

    if mean_loco_r2 > best_loco_r2:
        best_loco_r2 = mean_loco_r2
        best_components = n_components


loco_results = pd.DataFrame(
    loco_results
)


print()
print("=" * 100)
print("LOCO RESULTS")
print("=" * 100)

print(
    tabulate(
        loco_results,
        headers="keys",
        floatfmt=".6f",
        tablefmt="rounded_outline",
        showindex=False,
    )
)


print()
print(
    f"Best number of PLS components: "
    f"{best_components}"
)

print(
    f"Best LOCO R²: "
    f"{best_loco_r2:.6f}"
)


best_parameters = {
    "n_components": int(
        best_components
    ),
}


parameters_path = (
    PARAMETERS_DIR
    + "MorganPLS_LOCO_BestParameters.json"
)


with open(
    parameters_path,
    "w",
) as file:
    json.dump(
        best_parameters,
        file,
        indent=4,
    )


print()
print(
    "Best parameters saved to:"
)

print(parameters_path)


print()
print("=" * 100)
print("TRAINING FINAL MODEL")
print("=" * 100)


# ------------------------------------------------------------
# Final feature filtering
# ------------------------------------------------------------

training_morgan = train_data[
    morgan_features
]

final_morgan_features = [
    column
    for column in morgan_features
    if training_morgan[column].nunique() > 1
]


print(
    f"Total Morgan features: "
    f"{len(morgan_features)}"
)

print(
    f"Non-constant training Morgan features: "
    f"{len(final_morgan_features)}"
)


X_train_base = train_data[
    numerical_features + categorical_features
].copy()

X_case_1_base = test_case_1[
    numerical_features + categorical_features
].copy()

X_case_2_base = test_case_2[
    numerical_features + categorical_features
].copy()


X_train_morgan = train_data[
    final_morgan_features
].copy()

X_case_1_morgan = test_case_1[
    final_morgan_features
].copy()

X_case_2_morgan = test_case_2[
    final_morgan_features
].copy()


(
    X_train_base_transformed,
    X_case_1_base_transformed,
    base_preprocessor,
) = transform_base_features(
    X_train_base,
    X_case_1_base,
)


# Transform Case 2 using the same fitted
# preprocessing as Case 1.

X_case_2_base_transformed = (
    base_preprocessor.transform(
        X_case_2_base
    )
)


(
    X_train_morgan_transformed,
    X_case_1_morgan_transformed,
    morgan_scaler,
) = transform_morgan_features(
    X_train_morgan,
    X_case_1_morgan,
)


X_case_2_morgan_transformed = (
    morgan_scaler.transform(
        X_case_2_morgan
    )
)


X_train = np.hstack([
    X_train_base_transformed,
    X_train_morgan_transformed,
])

X_case_1 = np.hstack([
    X_case_1_base_transformed,
    X_case_1_morgan_transformed,
])

X_case_2 = np.hstack([
    X_case_2_base_transformed,
    X_case_2_morgan_transformed,
])


y_train = train_data[
    "Viability"
].to_numpy()

y_case_1 = test_case_1[
    "Viability"
].to_numpy()

y_case_2 = test_case_2[
    "Viability"
].to_numpy()


final_model = PLSRegression(
    n_components=best_components,
    scale=False,
)


final_model.fit(
    X_train,
    y_train,
)


predictions_case_1 = (
    final_model.predict(
        X_case_1
    )
    .ravel()
)

predictions_case_2 = (
    final_model.predict(
        X_case_2
    )
    .ravel()
)


metrics_case_1 = calculate_metrics(
    y_case_1,
    predictions_case_1,
)

metrics_case_2 = calculate_metrics(
    y_case_2,
    predictions_case_2,
)


print()
print("=" * 100)
print("EXTERNAL TEST — CASE 1")
print("=" * 100)

print(
    tabulate(
        [metrics_case_1],
        headers="keys",
        floatfmt=".6f",
        tablefmt="rounded_outline",
    )
)


print()
print("=" * 100)
print("EXTERNAL TEST — CASE 2")
print("=" * 100)

print(
    tabulate(
        [metrics_case_2],
        headers="keys",
        floatfmt=".6f",
        tablefmt="rounded_outline",
    )
)


summary = pd.DataFrame([
    {
        "Model": "MorganPLS",
        "Components": best_components,
        "Morgan Features": len(
            final_morgan_features
        ),
        "LOCO R2": best_loco_r2,
        "Case 1 MAE": metrics_case_1["MAE"],
        "Case 1 RMSE": metrics_case_1["RMSE"],
        "Case 1 MedAE": metrics_case_1["MedAE"],
        "Case 1 R2": metrics_case_1["R2"],
        "Case 2 MAE": metrics_case_2["MAE"],
        "Case 2 RMSE": metrics_case_2["RMSE"],
        "Case 2 MedAE": metrics_case_2["MedAE"],
        "Case 2 R2": metrics_case_2["R2"],
    }
])


print()
print("=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

print(
    tabulate(
        summary,
        headers="keys",
        floatfmt=".6f",
        tablefmt="rounded_outline",
        showindex=False,
    )
)


print()
print("=" * 100)
print("DETAILED EXTERNAL GENERALIZATION ANALYSIS")
print("=" * 100)


THRESHOLDS = [
    10,
    20,
    30,
    50,
]


external_predictions = test_data.copy()

external_predictions["Prediction"] = (
    final_model.predict(
        np.hstack([
            base_preprocessor.transform(
                external_predictions[
                    numerical_features
                    + categorical_features
                ]
            ),
            morgan_scaler.transform(
                external_predictions[
                    final_morgan_features
                ]
            ),
        ])
    )
    .ravel()
)


external_predictions[
    "CellLineTrainingCount"
] = (
    external_predictions["CellLine"]
    .map(cell_line_counts)
    .fillna(0)
)


def evaluate_subset(
    subset,
):
    if len(subset) == 0:
        return {
            "N": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MedAE": np.nan,
            "R2": np.nan,
        }

    y_true = subset[
        "Viability"
    ].to_numpy()

    y_pred = subset[
        "Prediction"
    ].to_numpy()

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    return {
        "N": len(subset),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MedAE": metrics["MedAE"],
        "R2": metrics["R2"],
    }


# ------------------------------------------------------------
# Per-coumarin results without cell-line filtering
# ------------------------------------------------------------

print()
print("=" * 100)
print("PER-COUMARIN RESULTS — ALL EXTERNAL DATA")
print("=" * 100)


coumarin_results = []

for coumarin in EXTERNAL_COUMARINS:
    subset = external_predictions[
        external_predictions["Coumarin"]
        == coumarin
    ]

    metrics = evaluate_subset(
        subset
    )

    coumarin_results.append({
        "Coumarin": coumarin,
        **metrics,
    })


coumarin_results = pd.DataFrame(
    coumarin_results
)


print(
    tabulate(
        coumarin_results,
        headers="keys",
        floatfmt=".6f",
        tablefmt="rounded_outline",
        showindex=False,
    )
)


# ------------------------------------------------------------
# Per-coumarin results at each threshold
# ------------------------------------------------------------

threshold_results = []


for threshold in THRESHOLDS:
    print()
    print("=" * 100)
    print(
        f"CELL LINE THRESHOLD >= {threshold}"
    )
    print("=" * 100)

    threshold_data = external_predictions[
        external_predictions[
            "CellLineTrainingCount"
        ] >= threshold
    ].copy()

    print(
        f"Total observations: "
        f"{len(threshold_data)}"
    )

    print()

    for coumarin in EXTERNAL_COUMARINS:
        subset = threshold_data[
            threshold_data["Coumarin"]
            == coumarin
        ]

        metrics = evaluate_subset(
            subset
        )

        result = {
            "Threshold": threshold,
            "Coumarin": coumarin,
            **metrics,
        }

        threshold_results.append(
            result
        )

        print(
            f"{coumarin:<16} "
            f"N={metrics['N']:3d} | "
            f"MAE={metrics['MAE']:.6f} | "
            f"RMSE={metrics['RMSE']:.6f} | "
            f"MedAE={metrics['MedAE']:.6f} | "
            f"R²={metrics['R2']:.6f}"
        )


threshold_results = pd.DataFrame(
    threshold_results
)


print()
print("=" * 100)
print("ALL PER-COUMARIN THRESHOLD RESULTS")
print("=" * 100)

print(
    tabulate(
        threshold_results,
        headers="keys",
        floatfmt=".6f",
        tablefmt="rounded_outline",
        showindex=False,
    )
)


# ------------------------------------------------------------
# Combined results at each threshold
# ------------------------------------------------------------

combined_threshold_results = []


for threshold in THRESHOLDS:
    subset = external_predictions[
        external_predictions[
            "CellLineTrainingCount"
        ] >= threshold
    ]

    metrics = evaluate_subset(
        subset
    )

    combined_threshold_results.append({
        "Threshold": threshold,
        **metrics,
    })


combined_threshold_results = pd.DataFrame(
    combined_threshold_results
)


print()
print("=" * 100)
print("COMBINED EXTERNAL RESULTS BY CELL LINE THRESHOLD")
print("=" * 100)

print(
    tabulate(
        combined_threshold_results,
        headers="keys",
        floatfmt=".6f",
        tablefmt="rounded_outline",
        showindex=False,
    )
)


# ------------------------------------------------------------
# Cell-line coverage
# ------------------------------------------------------------

print()
print("=" * 100)
print("CELL LINE COVERAGE")
print("=" * 100)


coverage_results = []


for threshold in THRESHOLDS:
    threshold_data = external_predictions[
        external_predictions[
            "CellLineTrainingCount"
        ] >= threshold
    ]

    for coumarin in EXTERNAL_COUMARINS:
        subset = threshold_data[
            threshold_data["Coumarin"]
            == coumarin
        ]

        coverage_results.append({
            "Threshold": threshold,
            "Coumarin": coumarin,
            "Observations": len(subset),
            "CellLines": subset[
                "CellLine"
            ].nunique(),
        })


coverage_results = pd.DataFrame(
    coverage_results
)


print(
    tabulate(
        coverage_results,
        headers="keys",
        tablefmt="rounded_outline",
        showindex=False,
    )
)

# ============================================================
# Prepare External Test Predictions
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score
)

OUTPUT_DIR = "Results/Regression/Evaluations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXTERNAL_COUMARINS = ["imperatorin", "scopoletin", "fanesiferola"]

# Use the same training data and external test split as before.
train_data = data[
    data["Coumarin"].isin(TRAIN_COUMARINS)
].copy()

test_data = data[
    data["Coumarin"].isin(EXTERNAL_COUMARINS)
].copy()

# Actual external test targets.
y_test = test_data["Viability"].to_numpy()

# Extract the biological and molecular features.
X_test_base = test_data[
    categorical_features + numerical_features
]

# Apply the fitted preprocessing pipeline.
X_test_processed = base_preprocessor.transform(X_test_base)

# Extract Morgan fingerprints.
X_test_morgan = test_data[final_morgan_features].to_numpy()

# Apply the fitted Morgan scaler.
X_test_morgan_scaled = morgan_scaler.transform(X_test_morgan)

# Combine base features and scaled Morgan fingerprints.
X_test_final = np.hstack([
    X_test_processed,
    X_test_morgan_scaled
])

# Generate predictions using the already-fitted final model.
y_test_pred = np.asarray(final_model.predict(X_test_final)).ravel()

print(f"Training observations: {len(train_data)}")
print(f"External test observations: {len(test_data)}")
print(f"Actual target shape: {y_test.shape}")
print(f"Prediction shape: {y_test_pred.shape}")

assert len(y_test) == len(y_test_pred) == len(test_data)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "Results/Regression/Evaluations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Prepare predictions
# ============================================================

analysis_data = test_data.copy().reset_index(drop=True)
analysis_data["Actual"] = np.asarray(y_test)
analysis_data["Predicted"] = np.asarray(y_test_pred)
analysis_data["Error"] = analysis_data["Predicted"] - analysis_data["Actual"]
analysis_data["AbsoluteError"] = analysis_data["Error"].abs()

# Count training observations per cell line.
training_cell_counts = (
    train_data["CellLine"]
    .value_counts()
    .to_dict()
)

analysis_data["TrainingN"] = (
    analysis_data["CellLine"]
    .map(training_cell_counts)
    .fillna(0)
    .astype(int)
)

# ============================================================
# Per-coumarin, per-cell-line metrics
# ============================================================

rows = []

for (coumarin, cell_line), group in analysis_data.groupby(
    ["Coumarin", "CellLine"], dropna=False
):
    actual = group["Actual"].to_numpy()
    predicted = group["Predicted"].to_numpy()
    errors = predicted - actual

    rows.append({
        "Coumarin": coumarin,
        "CellLine": cell_line,
        "TrainingN": int(group["TrainingN"].iloc[0]),
        "TestN": len(group),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MedAE": median_absolute_error(actual, predicted),
        "R2": r2_score(actual, predicted) if len(group) >= 2 else np.nan,
        "MeanBias": np.mean(errors),
        "MeanActual": np.mean(actual),
        "MeanPredicted": np.mean(predicted)
    })

cell_line_results = pd.DataFrame(rows).sort_values(
    ["Coumarin", "TrainingN", "TestN"],
    ascending=[True, False, False]
)

print("\n" + "=" * 110)
print("CELL-LINE-LEVEL EXTERNAL GENERALIZATION")
print("=" * 110)
print(cell_line_results.round(3).to_string(index=False))

cell_line_results.to_csv(
    f"{OUTPUT_DIR}/MorganPLS_CellLineLevel_Results.csv",
    index=False
)

# ============================================================
# Correlation between training coverage and absolute error
# ============================================================

print("\n" + "=" * 110)
print("TRAINING CELL-LINE COVERAGE VS ABSOLUTE PREDICTION ERROR")
print("=" * 110)

correlation_rows = []

for coumarin, group in analysis_data.groupby("Coumarin"):
    pearson = group["TrainingN"].corr(
        group["AbsoluteError"], method="pearson"
    )
    spearman = group["TrainingN"].corr(
        group["AbsoluteError"], method="spearman"
    )

    correlation_rows.append({
        "Coumarin": coumarin,
        "N": len(group),
        "UniqueCellLines": group["CellLine"].nunique(),
        "PearsonR": pearson,
        "SpearmanR": spearman
    })

# Combined results
correlation_rows.append({
    "Coumarin": "combined",
    "N": len(analysis_data),
    "UniqueCellLines": analysis_data["CellLine"].nunique(),
    "PearsonR": analysis_data["TrainingN"].corr(
        analysis_data["AbsoluteError"], method="pearson"
    ),
    "SpearmanR": analysis_data["TrainingN"].corr(
        analysis_data["AbsoluteError"], method="spearman"
    )
})

correlation_results = pd.DataFrame(correlation_rows)

print(correlation_results.round(3).to_string(index=False))

correlation_results.to_csv(
    f"{OUTPUT_DIR}/MorganPLS_Coverage_Error_Correlations.csv",
    index=False
)

# ============================================================
# Observation-level error summary by coverage
# ============================================================

coverage_summary = (
    analysis_data.groupby(["Coumarin", "TrainingN"])
    .agg(
        TestN=("AbsoluteError", "size"),
        CellLines=("CellLine", "nunique"),
        MAE=("AbsoluteError", "mean"),
        MeanBias=("Error", "mean"),
        MedianAbsoluteError=("AbsoluteError", "median")
    )
    .reset_index()
    .sort_values(["Coumarin", "TrainingN"])
)

print("\n" + "=" * 110)
print("ERROR BY TRAINING CELL-LINE COUNT")
print("=" * 110)
print(coverage_summary.round(3).to_string(index=False))

coverage_summary.to_csv(
    f"{OUTPUT_DIR}/MorganPLS_Error_By_TrainingCoverage.csv",
    index=False
)

print(f"\nResults saved to: {OUTPUT_DIR}")