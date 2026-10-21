from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold, LeaveOneGroupOut

import pandas as pd

from model_utils import (
    DATA_DIR,
    RESULTS_DIR,
    RANDOM_STATE,
    DEFAULT_N_ITER,
    get_feature_sets,
    build_preprocessor,
    split_training_external,
    evaluate_external_model,
    save_model_results,
    print_dataframe,
)

MODEL_NAME = "ManyTrees"
EXTERNAL_COUMARINS = [
    "scopoletin", "imperatorin", "farnesiferola",
    "farnesiferolb", "farnesiferolc",
]

data = pd.read_csv(DATA_DIR / "Processed_Data.csv")
training_data, external_data = split_training_external(data, EXTERNAL_COUMARINS)
feature_sets = get_feature_sets(training_data)

param_distributions = {
    "model__n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "model__max_depth": [None, 5, 10, 15, 20, 30, 40],
    "model__min_samples_split": [2, 5, 10, 20, 30],
    "model__min_samples_leaf": [1, 2, 4, 8, 12],
    "model__max_features": [1.0, "sqrt", "log2"],
    "model__criterion": ["squared_error", "absolute_error", "poisson"],
    "model__bootstrap": [True, False],
}

for feature_name, features in feature_sets.items():
    for validation in ["KFold", "LOCO"]:
        print(f"\n{'=' * 70}\n{MODEL_NAME} | {feature_name} | {validation}\n{'=' * 70}")

        estimator = Pipeline([
            ("preprocessor", build_preprocessor(features)),
            ("model", RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=1,
            )),
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
            scoring="r2",
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

        print(f"\nBest CV R²: {search.best_score_:.4f}")
        print("\nBest parameters:")
        print(search.best_params_)

        fold_results = []
        splits = (
            cv.split(X, y, groups)
            if validation == "LOCO"
            else cv.split(X, y)
        )

        for fold, (_, test_idx) in enumerate(splits, start=1):
            result = {
                "Fold": fold,
                "R2": search.cv_results_[f"split{fold - 1}_test_score"][best_index],
            }

            if validation == "LOCO":
                result["HeldOutCoumarin"] = groups.iloc[test_idx].iloc[0]

            fold_results.append(result)

        fold_results = pd.DataFrame(fold_results)
        print("\nCV fold scores:")
        print_dataframe(fold_results)

        external_results = evaluate_external_model(
            model=best_model,
            training_data=training_data,
            external_data=external_data,
            features=features,
        )

        print("\nExternal overall results:")
        print_dataframe(external_results["Overall"])
        print("\nExternal results by coumarin:")
        print_dataframe(external_results["ByCoumarin"])
        print("\nExternal coverage results:")
        print_dataframe(external_results["CoverageSummary"])
        print("\nCoverage By Coumarin results:")
        print_dataframe(external_results["CoverageByCoumarin"])

        results = {
            "CVFoldResults": fold_results,
            "CVResults": pd.DataFrame([{
                "Validation": validation,
                "FeatureSet": feature_name,
                "BestCV_R2": search.best_score_,
            }]),
            **{f"External_{name}": df for name, df in external_results.items()},
        }

        save_model_results(
            results=results,
            output_directory=RESULTS_DIR / "Regression" / MODEL_NAME / feature_name,
            model_name=f"{MODEL_NAME}_{feature_name}_{validation}",
            best_parameters=search.best_params_,
            fitted_model=best_model,
        )