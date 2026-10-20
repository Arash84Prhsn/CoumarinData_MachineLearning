import os
import re
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.stats import randint, uniform, loguniform

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Configuration
# ============================================================

DATA_PATH = "Data/Classification_Data.csv"
OUTPUT_DIR = "Results/Classification/XGBoost_RandomSearch"

RANDOM_STATE = 42
N_ITER = 50
N_JOBS = -1

EXTERNAL_COUMARINS = {
    "scopoletin",
    "imperatorin",
    "farnesiferola",
    "farnesiferolb",
    "farnesiferolc",
}

CELLLINE_THRESHOLDS = range(5, 51, 5)
CLASS_LABELS = [1, 2, 3, 4, 5]
ENCODED_CLASSES = [0, 1, 2, 3, 4]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================

def normalize_coumarin(value):
    if pd.isna(value):
        return ""

    return re.sub(r"[\s_-]+", "", str(value).strip().lower())


def normalize_cellline(value):
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {
            "n_samples": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
        }

    return {
        "n_samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="weighted",
            zero_division=0,
        ),
    }


def evaluate_subset(
    predictions,
    strategy,
    evaluation,
    coumarin,
    threshold,
    metrics_rows,
    confusion_rows,
):
    y_true = predictions["TrueClass"].to_numpy(dtype=int)
    y_pred = predictions["PredictedClass"].to_numpy(dtype=int)

    metrics_rows.append({
        "strategy": strategy,
        "evaluation": evaluation,
        "coumarin": coumarin,
        "cellline_threshold": threshold,
        **calculate_metrics(y_true, y_pred),
    })

    if len(predictions) == 0:
        return

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
    )

    for i, true_class in enumerate(CLASS_LABELS):
        for j, predicted_class in enumerate(CLASS_LABELS):
            confusion_rows.append({
                "strategy": strategy,
                "evaluation": evaluation,
                "coumarin": coumarin,
                "cellline_threshold": threshold,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "count": int(matrix[i, j]),
            })


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


# ============================================================
# Load and prepare data
# ============================================================

data = pd.read_csv(DATA_PATH).copy()

required_columns = {"Coumarin", "CellLine", "Viability"}
missing_columns = required_columns - set(data.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

data["_CoumarinKey"] = data["Coumarin"].apply(normalize_coumarin)
data["CellLine"] = data["CellLine"].apply(normalize_cellline)

data = data.dropna(
    subset=["Coumarin", "CellLine", "Viability"]
).copy()

data = data[data["CellLine"] != ""].copy()

# Convert raw viability percentages into classes if necessary.
viability = pd.to_numeric(data["Viability"], errors="raise")

already_classified = (
    viability.between(1, 5).all()
    and np.allclose(viability, np.round(viability))
)

if already_classified:
    data["Viability"] = viability.astype(int)
else:
    data["Viability"] = np.minimum(
        np.floor(viability / 20).astype(int) + 1,
        5,
    )

if not data["Viability"].isin(CLASS_LABELS).all():
    raise ValueError("Viability must contain classes 1 through 5.")

# Create LogDose if the file does not already contain it.
if "LogDose" not in data.columns and "Dose" in data.columns:
    dose = pd.to_numeric(data["Dose"], errors="coerce")

    if (dose.dropna() < 0).any():
        raise ValueError(
            "Negative Dose values exist; resolve them before applying log1p."
        )

    data["LogDose"] = np.log1p(dose)

# ============================================================
# Split training and external data
# ============================================================

external_mask = data["_CoumarinKey"].isin(EXTERNAL_COUMARINS)

train_data = data.loc[~external_mask].copy()
external_data = data.loc[external_mask].copy()

if train_data.empty or external_data.empty:
    raise ValueError("Training or external dataset is empty.")

found_external = set(external_data["_CoumarinKey"].unique())
missing_external = EXTERNAL_COUMARINS - found_external

if missing_external:
    print("External coumarins not found:", sorted(missing_external))

print(f"Training observations: {len(train_data)}")
print(f"External observations: {len(external_data)}")
print(f"Training coumarins: {train_data['_CoumarinKey'].nunique()}")
print(f"External coumarins: {external_data['_CoumarinKey'].nunique()}")

# ============================================================
# Cell-line frequencies from training data ONLY
# ============================================================

training_cellline_counts = train_data["CellLine"].value_counts()

train_data["_TrainingCellLineCount"] = (
    train_data["CellLine"]
    .map(training_cellline_counts)
    .fillna(0)
    .astype(int)
)

external_data["_TrainingCellLineCount"] = (
    external_data["CellLine"]
    .map(training_cellline_counts)
    .fillna(0)
    .astype(int)
)

training_cellline_counts.rename_axis("CellLine").reset_index(
    name="TrainingObservations"
).to_csv(
    os.path.join(OUTPUT_DIR, "Training_CellLine_Frequencies.csv"),
    index=False,
)

# ============================================================
# Feature preparation
# ============================================================

excluded_features = {
    "Viability",
    "Coumarin",
    "_CoumarinKey",
    "Cancer",
    "Author",
    "Year",
    "Dose",
    "_TrainingCellLineCount",
}

excluded_features.update(
    column
    for column in data.columns
    if column.lower().startswith("unnamed:")
)

categorical_features = ["CellLine"]

numeric_features = [
    column
    for column in train_data.select_dtypes(include=np.number).columns
    if column not in excluded_features
]

if not numeric_features:
    raise ValueError("No numeric predictor features were found.")

feature_columns = categorical_features + numeric_features

X_train = train_data[feature_columns].copy()
X_external = external_data[feature_columns].copy()

# Original class labels are 1-5.
# XGBoost receives encoded labels 0-4.
y_train_original = train_data["Viability"].astype(int)

class_to_encoded = {
    class_label: class_label - 1
    for class_label in CLASS_LABELS
}

encoded_to_class = {
    encoded: original
    for original, encoded in class_to_encoded.items()
}

y_train = y_train_original.map(class_to_encoded).astype(int)

if set(y_train.unique()) != set(ENCODED_CLASSES):
    raise ValueError(
        "The full training set must contain all five classes."
    )

groups = train_data["_CoumarinKey"].to_numpy()

print(f"Numeric features: {len(numeric_features)}")
print(f"Total features before encoding: {len(feature_columns)}")

# ============================================================
# Preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cellline",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
            categorical_features,
        ),
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features,
        ),
    ],
    remainder="drop",
)

# ============================================================
# XGBoost and search space
# ============================================================

base_classifier = XGBClassifier(
    objective="multi:softprob",
    num_class=5,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=1,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", base_classifier),
    ]
)

# Both searches use exactly the same search space.
param_distributions = {
    "classifier__n_estimators": randint(200, 1001),
    "classifier__max_depth": randint(2, 9),
    "classifier__learning_rate": loguniform(0.01, 0.2),
    "classifier__subsample": uniform(0.65, 0.35),
    "classifier__colsample_bytree": uniform(0.5, 0.5),
    "classifier__min_child_weight": randint(1, 10),
    "classifier__reg_lambda": loguniform(0.1, 20.0),
    "classifier__reg_alpha": loguniform(1e-4, 5.0),
    "classifier__gamma": uniform(0.0, 2.0),
}

# Score all five classes consistently, including classes absent
# from a particular validation fold.
macro_f1_scorer = make_scorer(
    f1_score,
    labels=ENCODED_CLASSES,
    average="macro",
    zero_division=0,
)

# ============================================================
# Define validation strategies
# ============================================================

kfold_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

n_coumarins = train_data["_CoumarinKey"].nunique()

if n_coumarins < 2:
    raise ValueError("LOCO requires at least two training coumarins.")

loco_cv = GroupKFold(n_splits=n_coumarins)

validation_strategies = {
    "KFold": {
        "cv": kfold_cv,
        "use_groups": False,
    },
    "LOCO": {
        "cv": loco_cv,
        "use_groups": True,
    },
}

# ============================================================
# Randomized search
# ============================================================

fitted_models = {}
search_summaries = []

for strategy, settings in validation_strategies.items():

    print("\n" + "=" * 80)
    print(f"RANDOMIZED SEARCH: {strategy}")
    print("=" * 80)

    search = RandomizedSearchCV(
        estimator=clone(pipeline),
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring=macro_f1_scorer,
        n_jobs=N_JOBS,
        cv=settings["cv"],
        refit=True,
        random_state=RANDOM_STATE,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    if settings["use_groups"]:
        search.fit(
            X_train,
            y_train,
            groups=groups,
        )
    else:
        search.fit(
            X_train,
            y_train,
        )

    fitted_models[strategy] = search.best_estimator_

    print(f"\nBest CV macro-F1 ({strategy}): {search.best_score_:.4f}")
    print("Best parameters:")
    print(json.dumps(search.best_params_, indent=4, default=str))

    strategy_dir = os.path.join(OUTPUT_DIR, strategy)
    os.makedirs(strategy_dir, exist_ok=True)

    # Save the best fitted pipeline.
    joblib.dump(
        search.best_estimator_,
        os.path.join(strategy_dir, "Best_Model.joblib"),
    )

    # Save best parameters and CV results.
    save_json(
        search.best_params_,
        os.path.join(strategy_dir, "Best_Parameters.json"),
    )

    pd.DataFrame(search.cv_results_).to_csv(
        os.path.join(strategy_dir, "CV_Results.csv"),
        index=False,
    )

    search_summaries.append({
        "strategy": strategy,
        "best_cv_macro_f1": search.best_score_,
        "best_parameters": json.dumps(
            search.best_params_,
            default=str,
        ),
    })

# ============================================================
# Save search comparison
# ============================================================

pd.DataFrame(search_summaries).to_csv(
    os.path.join(OUTPUT_DIR, "Search_Comparison.csv"),
    index=False,
)

# ============================================================
# External evaluation
# ============================================================

all_metrics = []
all_confusion_matrices = []
all_predictions = []

for strategy, model in fitted_models.items():

    print("\n" + "=" * 80)
    print(f"EXTERNAL EVALUATION: {strategy}")
    print("=" * 80)

    predictions = external_data.copy()

    encoded_predictions = model.predict(X_external).astype(int)

    predictions["TrueClass"] = predictions["Viability"].astype(int)
    predictions["PredictedClass"] = [
        encoded_to_class[value]
        for value in encoded_predictions
    ]

    probabilities = model.predict_proba(X_external)

    predictions["PredictionConfidence"] = probabilities.max(axis=1)
    predictions["Strategy"] = strategy

    # Save probabilities for each predicted class.
    for encoded_class, original_class in encoded_to_class.items():
        predictions[f"Probability_Class_{original_class}"] = (
            probabilities[:, encoded_class]
        )

    all_predictions.append(predictions)

    # --------------------------------------------------------
    # Overall external test and per-coumarin results
    # --------------------------------------------------------

    evaluate_subset(
        predictions,
        strategy,
        evaluation="Overall",
        coumarin="All external coumarins",
        threshold=0,
        metrics_rows=all_metrics,
        confusion_rows=all_confusion_matrices,
    )

    for coumarin_key in sorted(found_external):

        subset = predictions[
            predictions["_CoumarinKey"] == coumarin_key
        ]

        evaluate_subset(
            subset,
            strategy,
            evaluation="Overall",
            coumarin=coumarin_key,
            threshold=0,
            metrics_rows=all_metrics,
            confusion_rows=all_confusion_matrices,
        )

    # --------------------------------------------------------
    # Cell-line availability thresholds
    # --------------------------------------------------------

    for threshold in CELLLINE_THRESHOLDS:

        filtered = predictions[
            predictions["_TrainingCellLineCount"] >= threshold
        ].copy()

        print(
            f"{strategy} | Cell-line threshold >= {threshold}: "
            f"{len(filtered)}/{len(predictions)} observations"
        )

        evaluate_subset(
            filtered,
            strategy,
            evaluation="CellLineFiltered",
            coumarin="All external coumarins",
            threshold=threshold,
            metrics_rows=all_metrics,
            confusion_rows=all_confusion_matrices,
        )

        for coumarin_key in sorted(found_external):

            subset = filtered[
                filtered["_CoumarinKey"] == coumarin_key
            ]

            evaluate_subset(
                subset,
                strategy,
                evaluation="CellLineFiltered",
                coumarin=coumarin_key,
                threshold=threshold,
                metrics_rows=all_metrics,
                confusion_rows=all_confusion_matrices,
            )

# ============================================================
# Save external predictions and evaluation tables
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)

predictions_df.to_csv(
    os.path.join(OUTPUT_DIR, "External_Predictions.csv"),
    index=False,
)

metrics_df = pd.DataFrame(all_metrics)

metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "Classification_Metrics.csv"),
    index=False,
)

confusion_df = pd.DataFrame(all_confusion_matrices)

confusion_df.to_csv(
    os.path.join(OUTPUT_DIR, "Confusion_Matrices.csv"),
    index=False,
)

# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 80)
print("SEARCH COMPARISON")
print("=" * 80)

print(
    pd.DataFrame(search_summaries)[
        ["strategy", "best_cv_macro_f1"]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n" + "=" * 80)
print("OVERALL EXTERNAL RESULTS")
print("=" * 80)

overall = metrics_df[
    metrics_df["evaluation"] == "Overall"
]

print(
    overall[
        [
            "strategy",
            "coumarin",
            "n_samples",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n" + "=" * 80)
print("COMBINED CELL-LINE THRESHOLD RESULTS")
print("=" * 80)

threshold_results = metrics_df[
    (metrics_df["evaluation"] == "CellLineFiltered")
    & (metrics_df["coumarin"] == "All external coumarins")
]

print(
    threshold_results[
        [
            "strategy",
            "cellline_threshold",
            "n_samples",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
        ]
    ].sort_values(
        ["strategy", "cellline_threshold"]
    ).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n" + "=" * 80)
print("COMPLETED")
print("=" * 80)
print(f"Results directory: {OUTPUT_DIR}")