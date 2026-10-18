import os
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    SplineTransformer,
    StandardScaler,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)
from tabulate import tabulate


DATA_DIR = "Data/Processed_Data.csv"
RESULTS_DIR = "Results/Regression/"
PARAMETERS_DIR = RESULTS_DIR + "BestParameters/"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PARAMETERS_DIR, exist_ok=True)

CELL_LINE_THRESHOLD = 10

LINE = "=" * 100
TRIANGLE = "▶"


test_coumarins = [
    "scopoletin",
    "imperatorin",
]


spline_features = [
    "LogDose",
    "Time",
]


molecular_features = [
    "MW",
    "LogP",
    "NumHDonors",
    "NumHAcceptors",
    "TPSA",
    "RingCounts",
    "fCsp3",
]


categorical_features = [
    "CellLine",
]


data = pd.read_csv(DATA_DIR)


train_data = data[
    ~data["Coumarin"].isin(test_coumarins)
].copy()

test_data = data[
    data["Coumarin"].isin(test_coumarins)
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


test_case_1_features = test_case_1.drop(
    columns=["CellLineTrainingCount"]
)

test_case_2_features = test_case_2.drop(
    columns=["CellLineTrainingCount"]
)


print(LINE)
print(f"{TRIANGLE} Spline Regression / GAM-like Model")
print(LINE)

print(f"Training rows: {len(train_data)}")
print(f"External test rows: {len(test_data)}")
print(f"External Case 1 rows: {len(test_case_1)}")
print(f"External Case 2 rows: {len(test_case_2)}")
print()

print("Training coumarins:")
print(sorted(train_data["Coumarin"].unique()))
print()

print("External test coumarins:")
print(sorted(test_data["Coumarin"].unique()))
print()


def create_model(
    n_knots,
    alpha,
):
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "spline",
                SplineTransformer(
                    n_knots=n_knots,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                    extrapolation="linear",
                ),
                spline_features,
            ),
            (
                "molecular",
                StandardScaler(),
                molecular_features,
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

    model = Pipeline([
        (
            "preprocessing",
            preprocessing,
        ),
        (
            "ridge",
            Ridge(
                alpha=alpha,
            ),
        ),
    ])

    return model


def calculate_metrics(
    y_true,
    y_pred,
):
    return {
        "MAE": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            ),
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


loco = LeaveOneGroupOut()

groups = train_data["Coumarin"]


n_knots_candidates = [
    3,
    4,
    5,
    6,
    7,
]

alpha_candidates = [
    0.01,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
]


best_n_knots = None
best_alpha = None
best_loco_r2 = -np.inf


total_combinations = (
    len(n_knots_candidates)
    * len(alpha_candidates)
)

iteration = 0


print(LINE)
print(f"{TRIANGLE} Searching spline parameters with LOCO")
print(LINE)

print(
    f"Testing {total_combinations} parameter combinations..."
)
print()


for n_knots in n_knots_candidates:
    for alpha in alpha_candidates:
        iteration += 1

        fold_scores = []

        for train_indices, validation_indices in loco.split(
            train_data,
            groups=groups,
        ):
            fold_train = train_data.iloc[
                train_indices
            ]

            fold_validation = train_data.iloc[
                validation_indices
            ]

            X_train = fold_train[
                spline_features
                + molecular_features
                + categorical_features
            ]

            X_validation = fold_validation[
                spline_features
                + molecular_features
                + categorical_features
            ]

            y_train = fold_train[
                "Viability"
            ].to_numpy()

            y_validation = fold_validation[
                "Viability"
            ].to_numpy()

            model = create_model(
                n_knots=n_knots,
                alpha=alpha,
            )

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_validation,
            )

            fold_r2 = r2_score(
                y_validation,
                predictions,
            )

            fold_scores.append(
                fold_r2
            )

        mean_loco_r2 = np.mean(
            fold_scores
        )

        print(
            f"Iteration {iteration:3d}/{total_combinations} | "
            f"Knots: {n_knots} | "
            f"Alpha: {alpha:<7g} | "
            f"LOCO R²: {mean_loco_r2:.6f}"
        )

        if mean_loco_r2 > best_loco_r2:
            best_loco_r2 = mean_loco_r2
            best_n_knots = n_knots
            best_alpha = alpha


print()
print(f"{TRIANGLE} Best number of knots: {best_n_knots}")
print(f"{TRIANGLE} Best Ridge alpha: {best_alpha}")
print(f"{TRIANGLE} Best LOCO R²: {best_loco_r2:.6f}")


best_parameters = {
    "n_knots": best_n_knots,
    "alpha": best_alpha,
}


parameters_path = (
    PARAMETERS_DIR
    + "SplineRidge_LOCO_BestParameters.json"
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
print(f"{TRIANGLE} Best parameters saved to:")
print(parameters_path)


print()
print(LINE)
print(f"{TRIANGLE} Training final Spline Regression model")
print(LINE)


X_train = train_data[
    spline_features
    + molecular_features
    + categorical_features
]

y_train = train_data[
    "Viability"
].to_numpy()


final_model = create_model(
    n_knots=best_n_knots,
    alpha=best_alpha,
)


final_model.fit(
    X_train,
    y_train,
)


X_case_1 = test_case_1_features[
    spline_features
    + molecular_features
    + categorical_features
]

X_case_2 = test_case_2_features[
    spline_features
    + molecular_features
    + categorical_features
]


y_case_1 = test_case_1_features[
    "Viability"
].to_numpy()

y_case_2 = test_case_2_features[
    "Viability"
].to_numpy()


predictions_case_1 = final_model.predict(
    X_case_1,
)

predictions_case_2 = final_model.predict(
    X_case_2,
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
print(f"{TRIANGLE} External test Case 1:")
print(
    tabulate(
        [metrics_case_1],
        headers="keys",
        floatfmt=".6f",
        tablefmt="rounded_outline",
    )
)


print()
print(f"{TRIANGLE} External test Case 2:")
print(
    tabulate(
        [metrics_case_2],
        headers="keys",
        floatfmt=".6f",
        tablefmt="rounded_outline",
    )
)


print()
print(LINE)
print(f"{TRIANGLE} Spline Regression Summary")
print(LINE)


summary = pd.DataFrame([
    {
        "Model": "SplineRidge",
        "Knots": best_n_knots,
        "Alpha": best_alpha,
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


print(
    tabulate(
        summary,
        headers="keys",
        floatfmt=".6f",
        tablefmt="rounded_outline",
        showindex=False,
    )
)