import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.stats import randint, uniform, loguniform

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from xgboost import XGBRegressor


warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "Data/Processed_Data.csv"
OUTPUT_DIR = "Results/XGBoost"

RANDOM_STATE = 42
OUTER_FOLDS = 5
N_ITER = 50

# Compounds reserved exclusively for external testing.
EXTERNAL_COUMARINS = [
    "scopoletin",
    "imperatorin",
    "farnesiferola",
    "farnesiferolb",
    "farnesiferolc",
]

MOLECULAR_FEATURES = [
    "MW",
    "LogP",
    "NumHDonors",
    "NumHAcceptors",
    "RotatableBonds",
    "TPSA",
    "RingCounts",
    "fCsp3",
]

# ============================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================

PARAM_DISTRIBUTIONS = {
    "xgb__n_estimators": randint(200, 1001),
    "xgb__max_depth": randint(2, 9),
    "xgb__learning_rate": loguniform(0.01, 0.2),
    "xgb__subsample": uniform(0.6, 0.4),
    "xgb__colsample_bytree": uniform(0.5, 0.5),
    "xgb__min_child_weight": loguniform(0.5, 20),
    "xgb__gamma": uniform(0, 5),
    "xgb__reg_alpha": loguniform(1e-8, 10),
    "xgb__reg_lambda": loguniform(0.1, 100),
}

# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

data = pd.read_csv(DATA_PATH)

# Normalize coumarin names.
data["Coumarin"] = (
    data["Coumarin"]
    .astype(str)
    .str.replace(r"\s+", "", regex=True)
    .str.lower()
)

# Identify Morgan fingerprint columns.
fingerprint_columns = sorted(
    [
        column
        for column in data.columns
        if column.startswith("Morgan_")
    ],
    key=lambda column: int(column.split("_")[1]),
)

if not fingerprint_columns:
    raise ValueError(
        "No Morgan fingerprint columns were found. "
        "Check Data/Processed_Data.csv."
    )

# Validate required columns.
required_columns = [
    "Coumarin",
    "CellLine",
    "Dose",
    "Time",
    "Viability",
    *MOLECULAR_FEATURES,
    *fingerprint_columns,
]

missing_columns = sorted(
    set(required_columns) - set(data.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Validate numerical data.
numeric_columns = [
    "Dose",
    "Time",
    "Viability",
    *MOLECULAR_FEATURES,
    *fingerprint_columns,
]

data[numeric_columns] = data[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce",
)

data = data.replace([np.inf, -np.inf], np.nan)

if data[required_columns].isna().any().any():
    raise ValueError(
        "Missing or non-finite values were found "
        "in required columns."
    )

if (data["Dose"] < 0).any() or (data["Time"] < 0).any():
    raise ValueError("Negative Dose or Time values were found.")

# ============================================================
# ENGINEER FEATURES
# ============================================================

data["LogDose"] = np.log1p(data["Dose"])

EXPERIMENTAL_FEATURES = [
    "LogDose",
    "Time",
]

CATEGORICAL_FEATURES = [
    "CellLine",
]

DESCRIPTOR_FEATURES = [
    *EXPERIMENTAL_FEATURES,
    *MOLECULAR_FEATURES,
]

MORGAN_FEATURES = [
    *DESCRIPTOR_FEATURES,
    *fingerprint_columns,
]

# ============================================================
# SEPARATE TRAINING AND EXTERNAL TEST DATA
# ============================================================

external_mask = data["Coumarin"].isin(EXTERNAL_COUMARINS)

train_data = (
    data.loc[~external_mask]
    .copy()
    .reset_index(drop=True)
)

external_data = (
    data.loc[external_mask]
    .copy()
    .reset_index(drop=True)
)

if train_data.empty or external_data.empty:
    raise ValueError(
        "The training or external test dataset is empty."
    )

missing_external = (
    set(EXTERNAL_COUMARINS)
    - set(external_data["Coumarin"].unique())
)

if missing_external:
    raise ValueError(
        "External compounds missing from the data: "
        f"{missing_external}"
    )

print("=" * 80)
print("DATASET SUMMARY")
print("=" * 80)

print(f"Total observations: {len(data)}")
print(f"Training observations: {len(train_data)}")
print(f"External observations: {len(external_data)}")

print("\nTraining coumarins:")
print(
    train_data["Coumarin"]
    .value_counts()
    .sort_index()
)

print("\nExternal coumarins:")
print(
    external_data["Coumarin"]
    .value_counts()
    .sort_index()
)

# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    return {
        "N": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "MedAE": median_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# ============================================================
# BUILD XGBOOST PIPELINE
# ============================================================

def build_xgb_pipeline(numeric_features):
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
            (
                "cellline",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
        tree_method="hist",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("xgb", model),
        ]
    )


# ============================================================
# LOCO-BASED RANDOM SEARCH
# ============================================================

def select_hyperparameters(
    X,
    y,
    groups,
    numeric_features,
):
    """
    Select XGBoost hyperparameters using RandomizedSearchCV
    with 100 candidates and Leave-One-Coumarin-Out CV.

    Each inner validation fold holds out one complete coumarin.
    """

    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))

    if n_groups < 2:
        raise ValueError(
            "At least two coumarins are required "
            "for LOCO hyperparameter selection."
        )

    pipeline = build_xgb_pipeline(numeric_features)

    inner_cv = GroupKFold(
        n_splits=n_groups,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=0,
        error_score="raise",
        return_train_score=False,
    )

    search.fit(
        X,
        y,
        groups=groups,
    )

    return search


# ============================================================
# NESTED CROSS-VALIDATION
# ============================================================

def evaluate_nested_cv(
    training_data,
    numeric_features,
    model_name,
    validation_strategy,
):
    X = training_data[
        numeric_features + CATEGORICAL_FEATURES
    ].copy()

    y = training_data["Viability"].to_numpy()
    groups = training_data["Coumarin"].to_numpy()

    # --------------------------------------------------------
    # Define outer validation strategy.
    # --------------------------------------------------------

    if validation_strategy == "LOCO":
        outer_cv = GroupKFold(
            n_splits=pd.Series(groups).nunique(),
        )

        outer_splits = outer_cv.split(
            X,
            y,
            groups=groups,
        )

    elif validation_strategy == "KFold":
        outer_cv = KFold(
            n_splits=OUTER_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        outer_splits = outer_cv.split(X, y)

    else:
        raise ValueError(
            f"Unknown validation strategy: "
            f"{validation_strategy}"
        )

    prediction_records = []
    fold_records = []

    # --------------------------------------------------------
    # Evaluate each outer fold.
    # --------------------------------------------------------

    for fold, (train_idx, valid_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        groups_train = groups[train_idx]

        # Hyperparameter selection always uses LOCO.
        search = select_hyperparameters(
            X=X_train,
            y=y_train,
            groups=groups_train,
            numeric_features=numeric_features,
        )

        # The outer validation fold was not used to select
        # the hyperparameters.
        predictions = np.asarray(
            search.predict(X_valid)
        ).ravel()

        fold_metrics = calculate_metrics(
            y_valid,
            predictions,
        )

        # ----------------------------------------------------
        # Record fold-level metrics.
        # ----------------------------------------------------

        fold_record = {
            "Model": model_name,
            "Validation": validation_strategy,
            "Fold": fold,
            "BestInnerLOCOMAE": -search.best_score_,
            **fold_metrics,
        }

        if validation_strategy == "LOCO":
            held_out_coumarin = np.unique(
                groups[valid_idx]
            )

            if len(held_out_coumarin) != 1:
                raise RuntimeError(
                    "Expected exactly one held-out coumarin "
                    "in an outer LOCO fold."
                )

            fold_record["HeldOutCoumarin"] = (
                held_out_coumarin[0]
            )

        fold_records.append(fold_record)

        # ----------------------------------------------------
        # Record individual predictions.
        # ----------------------------------------------------

        for idx, actual, predicted in zip(
            valid_idx,
            y_valid,
            predictions,
        ):
            prediction_records.append(
                {
                    "Model": model_name,
                    "Validation": validation_strategy,
                    "Fold": fold,
                    "Coumarin": groups[idx],
                    "CellLine": training_data.iloc[idx]["CellLine"],
                    "Actual": actual,
                    "Predicted": predicted,
                    "Error": predicted - actual,
                }
            )

        held_out_label = ""

        if validation_strategy == "LOCO":
            held_out_label = (
                f" | Held out: {held_out_coumarin[0]}"
            )

        print(
            f"{model_name} | {validation_strategy}"
            f" | Fold {fold}{held_out_label}"
            f" | Best inner LOCO MAE: "
            f"{-search.best_score_:.4f}"
            f" | MAE: {fold_metrics['MAE']:.4f}"
            f" | RMSE: {fold_metrics['RMSE']:.4f}"
            f" | R2: {fold_metrics['R2']:.4f}"
        )

    # --------------------------------------------------------
    # Aggregate outer-fold predictions.
    # --------------------------------------------------------

    predictions_df = pd.DataFrame(prediction_records)
    folds_df = pd.DataFrame(fold_records)

    overall_metrics = calculate_metrics(
        predictions_df["Actual"],
        predictions_df["Predicted"],
    )

    overall_record = {
        "Model": model_name,
        "Validation": validation_strategy,
        **overall_metrics,
    }

    return (
        predictions_df,
        folds_df,
        overall_record,
    )


# ============================================================
# EVALUATE BOTH XGBOOST REPRESENTATIONS
# ============================================================

model_features = {
    "Descriptor_XGBoost": DESCRIPTOR_FEATURES,
    "Morgan_XGBoost": MORGAN_FEATURES,
}

all_predictions = []
all_fold_results = []
all_summary_results = []

for model_name, numeric_features in model_features.items():
    for validation_strategy in ["KFold", "LOCO"]:
        print("\n" + "=" * 80)
        print(
            f"{model_name} | {validation_strategy}"
        )
        print("=" * 80)

        (
            predictions_df,
            folds_df,
            summary,
        ) = evaluate_nested_cv(
            training_data=train_data,
            numeric_features=numeric_features,
            model_name=model_name,
            validation_strategy=validation_strategy,
        )

        all_predictions.append(predictions_df)
        all_fold_results.append(folds_df)
        all_summary_results.append(summary)

# ============================================================
# SAVE CROSS-VALIDATION RESULTS
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)

fold_results_df = pd.concat(
    all_fold_results,
    ignore_index=True,
)

summary_df = pd.DataFrame(all_summary_results)

predictions_df.to_csv(
    f"{OUTPUT_DIR}/CV_Predictions.csv",
    index=False,
)

fold_results_df.to_csv(
    f"{OUTPUT_DIR}/CV_Fold_Results.csv",
    index=False,
)

summary_df.to_csv(
    f"{OUTPUT_DIR}/CV_Summary.csv",
    index=False,
)

print("\n" + "=" * 80)
print("CROSS-VALIDATION SUMMARY")
print("=" * 80)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

# ============================================================
# FINAL FIT AND EXTERNAL EVALUATION
# ============================================================

external_predictions = []

for model_name, numeric_features in model_features.items():
    print("\n" + "=" * 80)
    print(f"FINAL MODEL: {model_name}")
    print("=" * 80)

    X_train = train_data[
        numeric_features + CATEGORICAL_FEATURES
    ].copy()

    y_train = train_data["Viability"].to_numpy()
    groups_train = train_data["Coumarin"].to_numpy()

    X_external = external_data[
        numeric_features + CATEGORICAL_FEATURES
    ].copy()

    y_external = external_data["Viability"].to_numpy()

    # Select final hyperparameters using training coumarins
    # only, with LOCO-based RandomizedSearchCV.
    final_search = select_hyperparameters(
        X=X_train,
        y=y_train,
        groups=groups_train,
        numeric_features=numeric_features,
    )

    final_model = final_search.best_estimator_

    print(
        f"Best inner LOCO MAE: "
        f"{-final_search.best_score_:.4f}"
    )

    print("\nBest hyperparameters:")

    for parameter, value in sorted(
        final_search.best_params_.items()
    ):
        print(f"{parameter}: {value}")

    # Save selected hyperparameters.
    params_path = (
        f"{OUTPUT_DIR}/{model_name}_BestParameters.json"
    )

    with open(params_path, "w", encoding="utf-8") as file:
        json.dump(
            final_search.best_params_,
            file,
            indent=4,
        )

    # Predict the external observations without tuning on them.
    y_pred = np.asarray(
        final_model.predict(X_external)
    ).ravel()

    external_metrics = calculate_metrics(
        y_external,
        y_pred,
    )

    print("\nOverall external metrics:")

    for metric, value in external_metrics.items():
        if metric == "N":
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.4f}")

    # --------------------------------------------------------
    # Save individual external predictions.
    # --------------------------------------------------------

    for i, (_, row) in enumerate(
        external_data.iterrows()
    ):
        external_predictions.append(
            {
                "Model": model_name,
                "Coumarin": row["Coumarin"],
                "CellLine": row["CellLine"],
                "Dose": row["Dose"],
                "Time": row["Time"],
                "Actual": y_external[i],
                "Predicted": y_pred[i],
                "Error": y_pred[i] - y_external[i],
            }
        )

    # --------------------------------------------------------
    # External performance by coumarin.
    # --------------------------------------------------------

    compound_results = []

    for coumarin in EXTERNAL_COUMARINS:
        mask = (
            external_data["Coumarin"].to_numpy()
            == coumarin
        )

        if not mask.any():
            continue

        compound_metrics = calculate_metrics(
            y_external[mask],
            y_pred[mask],
        )

        compound_results.append(
            {
                "Model": model_name,
                "Coumarin": coumarin,
                **compound_metrics,
            }
        )

    compound_results_df = pd.DataFrame(
        compound_results
    )

    print("\nExternal performance by coumarin:")

    print(
        compound_results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    compound_results_df.to_csv(
        f"{OUTPUT_DIR}/{model_name}_External_By_Coumarin.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save fitted final model.
    # --------------------------------------------------------

    joblib.dump(
        final_model,
        f"{OUTPUT_DIR}/{model_name}_Final.joblib",
    )

# ============================================================
# SAVE ALL EXTERNAL PREDICTIONS
# ============================================================

external_predictions_df = pd.DataFrame(
    external_predictions
)

external_predictions_df.to_csv(
    f"{OUTPUT_DIR}/External_Predictions.csv",
    index=False,
)

print("\n" + "=" * 80)
print("XGBOOST EVALUATION COMPLETE")
print("=" * 80)

print(f"Results directory: {OUTPUT_DIR}/")

print("\nSaved files:")
print("- CV_Predictions.csv")
print("- CV_Fold_Results.csv")
print("- CV_Summary.csv")
print("- External_Predictions.csv")
print("- Descriptor_XGBoost_External_By_Coumarin.csv")
print("- Morgan_XGBoost_External_By_Coumarin.csv")
print("- Descriptor_XGBoost_BestParameters.json")
print("- Morgan_XGBoost_BestParameters.json")
print("- Descriptor_XGBoost_Final.joblib")
print("- Morgan_XGBoost_Final.joblib")