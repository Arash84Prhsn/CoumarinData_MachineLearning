import os
import json
import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder
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

RANDOM_STATE = 42
CELL_LINE_THRESHOLD = 10

LINE = "=" * 100
TRIANGLE = "▶"


test_coumarins = [
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
print(f"{TRIANGLE} PLS Regression")
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


def fit_encoder(train_df):
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    encoder.fit(
        train_df[categorical_features]
    )

    return encoder


def transform_features(df, encoder):
    X_num = df[
        numerical_features
    ].to_numpy(dtype=float)

    X_cat = encoder.transform(
        df[categorical_features]
    )

    return np.hstack([
        X_num,
        X_cat,
    ])


def create_model(n_components):
    return PLSRegression(
        n_components=n_components,
        scale=True,
        max_iter=1000,
        tol=1e-06,
    )


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


loco = LeaveOneGroupOut()

groups = train_data["Coumarin"]

n_components_candidates = range(1, 11)

best_n_components = None
best_loco_r2 = -np.inf

print(LINE)
print(f"{TRIANGLE} Searching PLS components with LOCO")
print(LINE)


for n_components in n_components_candidates:
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

        encoder = fit_encoder(
            fold_train
        )

        X_train = transform_features(
            fold_train,
            encoder,
        )

        X_validation = transform_features(
            fold_validation,
            encoder,
        )

        y_train = fold_train["Viability"].to_numpy()
        y_validation = fold_validation["Viability"].to_numpy()

        model = create_model(
            n_components
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

        fold_scores.append(fold_r2)

    mean_loco_r2 = np.mean(fold_scores)

    print(
        f"Components {n_components:2d} | "
        f"LOCO R²: {mean_loco_r2:.6f}"
    )

    if mean_loco_r2 > best_loco_r2:
        best_loco_r2 = mean_loco_r2
        best_n_components = n_components


print()
print(f"{TRIANGLE} Best number of PLS components: {best_n_components}")
print(f"{TRIANGLE} Best LOCO R²: {best_loco_r2:.6f}")


best_parameters = {
    "n_components": best_n_components,
}

parameters_path = (
    PARAMETERS_DIR
    + "PLS_LOCO_BestParameters.json"
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
print(f"{TRIANGLE} Best PLS parameters saved to:")
print(parameters_path)


print()
print(LINE)
print(f"{TRIANGLE} Training final PLS model")
print(LINE)


encoder = fit_encoder(
    train_data
)

X_train = transform_features(
    train_data,
    encoder,
)

y_train = train_data[
    "Viability"
].to_numpy()


final_model = create_model(
    best_n_components
)

final_model.fit(
    X_train,
    y_train,
)


X_case_1 = transform_features(
    test_case_1_features,
    encoder,
)

X_case_2 = transform_features(
    test_case_2_features,
    encoder,
)


y_case_1 = test_case_1_features[
    "Viability"
].to_numpy()

y_case_2 = test_case_2_features[
    "Viability"
].to_numpy()


predictions_case_1 = final_model.predict(
    X_case_1
).ravel()

predictions_case_2 = final_model.predict(
    X_case_2
).ravel()


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
print(f"{TRIANGLE} PLS Summary")
print(LINE)

summary = pd.DataFrame([
    {
        "Model": "PLS",
        "Components": best_n_components,
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