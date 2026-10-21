from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import pandas as pd

from model_utils import (
    DATA_DIR,
    RESULTS_DIR,
    RANDOM_STATE,
    DEFAULT_N_ITER,
    get_feature_sets,
    build_preprocessor,
    split_training_external,
    save_model_results,
    build_scaled_preprocessor,
    print_dataframe,
)

DEFAULT_N_ITER = 72

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

MODEL_NAME = "LogisticRegression"
EXTERNAL_COUMARINS = [
    "scopoletin", "imperatorin", "farnesiferola",
    "farnesiferolb", "farnesiferolc",
]

data = pd.read_csv("Data/Classification_Data.csv")
training_data, external_data = split_training_external(data, EXTERNAL_COUMARINS)
feature_sets = get_feature_sets(training_data)

training_data = training_data.copy()
external_data = external_data.copy()

param_distributions = {
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "model__solver": ["lbfgs", "newton-cg"],
    "model__class_weight": [None, "balanced"],
    "model__max_iter": [1000, 2000, 5000],
}

for feature_name, features in feature_sets.items():
    for validation in ["KFold", "LOCO"]:
        print(f"\n{'=' * 70}\n{MODEL_NAME} | {feature_name} | {validation}\n{'=' * 70}")

        estimator = Pipeline([
            ("preprocessor", build_scaled_preprocessor(features)),
            ("model", LogisticRegression(random_state=RANDOM_STATE)),
        ])

        X = training_data[features]
        y = training_data["Viability"]
        groups = training_data["Coumarin"]

        cv = (
            KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            if validation == "KFold"
            else LeaveOneGroupOut()
        )

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=DEFAULT_N_ITER,
            scoring="f1_macro",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        if validation == "LOCO":
            search.fit(X, y, groups=groups)
        else:
            search.fit(X, y)

        best_model = search.best_estimator_
        best_index = search.best_index_

        print(f"\nBest CV Macro-F1: {search.best_score_:.4f}")
        print("\nBest parameters:")
        print(search.best_params_)

        fold_results = []
        splits = cv.split(X, y, groups) if validation == "LOCO" else cv.split(X, y)

        for fold, (_, test_idx) in enumerate(splits, start=1):
            fold_result = {
                "Fold": fold,
                "F1": search.cv_results_[f"split{fold - 1}_test_score"][best_index],
            }

            if validation == "LOCO":
                fold_result["HeldOutCoumarin"] = groups.iloc[test_idx].iloc[0]

            fold_results.append(fold_result)

        fold_results = pd.DataFrame(fold_results)
        print("\nCV fold scores:")
        print_dataframe(fold_results)

        X_external = external_data[features]
        y_external = external_data["Viability"]
        external_pred = best_model.predict(X_external)
        external_prob = best_model.predict_proba(X_external)[:, 1]

        external_predictions = external_data[
            ["Coumarin", "CellLine", "Viability"]
        ].copy()
        external_predictions["Predicted"] = external_pred
        external_predictions["Probability"] = external_prob

        external_metrics = pd.DataFrame([{
            "N": len(y_external),
            "Accuracy": accuracy_score(y_external, external_pred),
            "Precision_Macro": precision_score(y_external, external_pred, average="macro", zero_division=0),
            "Recall_Macro": recall_score(y_external, external_pred, average="macro", zero_division=0),
            "F1_Macro": f1_score(y_external, external_pred, average="macro", zero_division=0),
            "ROC_AUC": (
                roc_auc_score(y_external, external_prob)
                if y_external.nunique() == 2 else float("nan")
            ),
        }])

        print("\nExternal evaluation:")
        print_dataframe(external_metrics)

        results = {
            "CVFoldResults": fold_results,
            "CVResults": pd.DataFrame([{
                "Validation": validation,
                "FeatureSet": feature_name,
                "BestCV_Macro-F1": search.best_score_,
            }]),
            "ExternalMetrics": external_metrics,
            "ExternalPredictions": external_predictions,
        }

        save_model_results(
            results=results,
            output_directory=RESULTS_DIR / "Classification" / MODEL_NAME / feature_name,
            model_name=f"{MODEL_NAME}_{feature_name}_{validation}",
            best_parameters=search.best_params_,
            fitted_model=best_model,
        )