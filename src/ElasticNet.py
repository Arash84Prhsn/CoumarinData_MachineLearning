import json

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    RandomizedSearchCV,
    KFold,
    LeaveOneGroupOut,
    cross_val_score,
)

from model_utils import (
    DATA_DIR,
    RESULTS_DIR,
    RANDOM_STATE,
    DEFAULT_N_ITER,
    TRIANGLES,
    get_feature_sets,
    build_preprocessor,
    split_training_external,
    evaluate_external_model,
    calculate_metrics,
    save_model_results,
    print_dataframe,
    plot_model_performance
)

MODEL_NAME = "ElasticNet"

EXTERNAL_COUMARINS = [
    "scopoletin",
    "imperatorin",
    "farnesiferola",
    "farnesiferolb",
    "farnesiferolc",
]

VALIDATION_METHODS = ["KFold", "LOCO"]

param_distributions = {
    "model__alpha": np.logspace(-3, 2, 100),
    "model__l1_ratio": np.linspace(0.05, 1.0, 20),
    "model__max_iter": [10000, 25000, 50000, 100000],
    "model__tol": [1e-4, 1e-3],
    "model__selection": ["cyclic", "random"],
}

data = pd.read_csv(DATA_DIR / "Processed_Data.csv")

training_data, external_data = split_training_external(
    data,
    EXTERNAL_COUMARINS,
)

feature_sets = get_feature_sets(training_data)

experiments = [
    {
        "name": "AllTrainingCoumarins",
        "data": training_data,
        "output_directory": RESULTS_DIR / "Regression" / MODEL_NAME,
    },
    {
        "name": "Esculetin_Herniarin",
        "data": training_data[
            training_data["Coumarin"].isin(["esculetin", "herniarin"])
        ].copy(),
        "output_directory": (
            RESULTS_DIR
            / "Regression"
            / MODEL_NAME
            / "Subsets"
            / "Esculetin_Herniarin"
        ),
    },
    {
        "name": "Other_Coumarins",
        "data": training_data[
            ~training_data["Coumarin"].isin(["esculetin", "herniarin"])
        ].copy(),
        "output_directory": (
            RESULTS_DIR
            / "Regression"
            / MODEL_NAME
            / "Subsets"
            / "Other_Coumarins"
        ),
    },
]

for experiment in experiments:
    experiment_name = experiment["name"]
    experiment_data = experiment["data"]
    output_directory = experiment["output_directory"]

    if experiment_data.empty:
        print(f"\n{TRIANGLES} Skipping empty subset: {experiment_name}")
        continue

    print(f"\n{TRIANGLES} Experiment: {experiment_name}")
    print(f"Training observations: {len(experiment_data)}")
    print(
        f"Training coumarins: {experiment_data['Coumarin'].nunique()}"
    )

    for feature_name, features in feature_sets.items():
        for validation in VALIDATION_METHODS:
            model_name = f"{validation}_{feature_name}"

            print(
                f"\n{'=' * 70}\n"
                f"{MODEL_NAME} | {experiment_name} | "
                f"{feature_name} | {validation}\n"
                f"{'=' * 70}"
            )

            X = experiment_data[features]
            y = experiment_data["Viability"]
            groups = experiment_data["Coumarin"]

            cv = (
                KFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=RANDOM_STATE,
                )
                if validation == "KFold"
                else LeaveOneGroupOut()
            )

            estimator = Pipeline([
                ("preprocessor", build_preprocessor(features)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(random_state=RANDOM_STATE, max_iter=10000),
                ),
            ])

            config_directory = output_directory / model_name
            search_directory = (
                config_directory / "Random Search Results"
            )

            parameters_path = search_directory / "BestParameters.json"
            cv_results_path = search_directory / "CVResults.csv"
            fold_results_path = search_directory / "CVFoldResults.csv"

            if parameters_path.exists():
                print(f"\n{TRIANGLES} Loading existing best parameters.")

                with open(parameters_path, encoding="utf-8") as file:
                    best_parameters = json.load(file)

                estimator.set_params(**best_parameters)

                if cv_results_path.exists() and fold_results_path.exists():
                    cached_cv = pd.read_csv(cv_results_path)
                    fold_results = pd.read_csv(fold_results_path)
                    best_cv_score = cached_cv["BestCV_R2"].iloc[0]

                    print(f"{TRIANGLES} Loaded saved CV results.")

                else:
                    print(
                        f"\n{TRIANGLES} Evaluating cached parameters "
                        "using cross-validation."
                    )

                    scores = cross_val_score(
                        estimator,
                        X,
                        y,
                        groups=groups if validation == "LOCO" else None,
                        scoring="r2",
                        cv=cv,
                        n_jobs=-1,
                    )

                    splits = (
                        cv.split(X, y, groups)
                        if validation == "LOCO"
                        else cv.split(X, y)
                    )

                    fold_results = []

                    for fold, ((_, test_idx), score) in enumerate(
                        zip(splits, scores),
                        start=1,
                    ):
                        result = {
                            "Fold": fold,
                            "R2": score,
                        }

                        if validation == "LOCO":
                            result["HeldOutCoumarin"] = (
                                groups.iloc[test_idx].iloc[0]
                            )

                        fold_results.append(result)

                    fold_results = pd.DataFrame(fold_results)
                    best_cv_score = scores.mean()

                best_model = estimator.fit(X, y)

            else:
                print(f"\n{TRIANGLES} Starting RandomizedSearchCV.")

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
                best_parameters = search.best_params_
                best_cv_score = search.best_score_
                best_index = search.best_index_

                fold_results = []

                splits = (
                    cv.split(X, y, groups)
                    if validation == "LOCO"
                    else cv.split(X, y)
                )

                for fold, (_, test_idx) in enumerate(splits, start=1):
                    result = {
                        "Fold": fold,
                        "R2": search.cv_results_[
                            f"split{fold - 1}_test_score"
                        ][best_index],
                    }

                    if validation == "LOCO":
                        result["HeldOutCoumarin"] = (
                            groups.iloc[test_idx].iloc[0]
                        )

                    fold_results.append(result)

                fold_results = pd.DataFrame(fold_results)

            print(
                f"\n{TRIANGLES} Best CV R² score: "
                f"{best_cv_score:.4f}"
            )
            print(f"\n{TRIANGLES} Best parameters:")
            print(best_parameters)

            print(f"\n{TRIANGLES} CV fold scores:")
            print_dataframe(fold_results)

            # ----------------------------------------------------------
            # External evaluation
            # ----------------------------------------------------------

            print(f"\n{TRIANGLES} Evaluating external coumarins.")

            external_results = evaluate_external_model(
                model=best_model,
                training_data=experiment_data,
                external_data=external_data,
                features=features,
            )

            print(f"\n{TRIANGLES} External overall results:")
            print_dataframe(external_results["Overall"])

            print(f"\n{TRIANGLES} External results by coumarin:")
            print_dataframe(external_results["ByCoumarin"])

            print(f"\n{TRIANGLES} External coverage results:")
            print_dataframe(external_results["CoverageSummary"])

            print(f"\n{TRIANGLES} Coverage by coumarin:")
            print_dataframe(external_results["CoverageByCoumarin"])

            # ----------------------------------------------------------
            # External predictions
            # ----------------------------------------------------------

            prediction_columns = [
                column
                for column in [
                    "Coumarin",
                    "CellLine",
                    "Dose",
                    "LogDose",
                    "Time",
                ]
                if column in external_data.columns
            ]

            external_predictions = external_data[
                prediction_columns
            ].copy()

            external_predictions["ActualViability"] = (
                external_data["Viability"].to_numpy()
            )

            external_predictions["PredictedViability"] = (
                np.asarray(best_model.predict(external_data[features])).reshape(-1)
            )

            external_predictions["Residual"] = (
                external_predictions["ActualViability"]
                - external_predictions["PredictedViability"]
            )

            # ----------------------------------------------------------
            # Training coverage for each cell line
            # ----------------------------------------------------------

            training_coverage = (
                experiment_data.groupby("CellLine")
                .agg(
                    TrainingObservations=("Viability", "size"),
                    TrainingCoumarins=("Coumarin", "nunique"),
                )
                .reset_index()
            )

            external_predictions = external_predictions.merge(
                training_coverage,
                on="CellLine",
                how="left",
            )

            coverage_columns = [
                "TrainingObservations",
                "TrainingCoumarins",
            ]

            external_predictions[coverage_columns] = (
                external_predictions[coverage_columns]
                .fillna(0)
                .astype(int)
            )

            # ----------------------------------------------------------
            # Metrics by cell line across external coumarins
            # ----------------------------------------------------------

            cell_line_results = []

            for cell_line, group in external_predictions.groupby("CellLine"):
                metrics = calculate_metrics(
                    group["ActualViability"],
                    group["PredictedViability"],
                )

                coverage = group.iloc[0]

                cell_line_results.append({
                    "CellLine": cell_line,
                    **metrics,
                    "TrainingObservations": int(
                        coverage["TrainingObservations"]
                    ),
                    "TrainingCoumarins": int(
                        coverage["TrainingCoumarins"]
                    ),
                    "ExternalCoumarins": group["Coumarin"].nunique(),
                })

            cell_line_results = pd.DataFrame(cell_line_results)
            cell_line_results = cell_line_results.sort_values(
                "MAE",
                ascending=True
                ).reset_index(drop=True)

            print(f"\n{TRIANGLES} External metrics by cell line:")
            print_dataframe(cell_line_results)

            # ----------------------------------------------------------
            # Metrics by cell line and external coumarin
            # ----------------------------------------------------------

            cell_line_coumarin_results = []

            for (cell_line, coumarin), group in (
                external_predictions.groupby(["CellLine", "Coumarin"])
            ):
                metrics = calculate_metrics(
                    group["ActualViability"],
                    group["PredictedViability"],
                )

                coverage = group.iloc[0]

                cell_line_coumarin_results.append({
                    "CellLine": cell_line,
                    "Coumarin": coumarin,
                    **metrics,
                    "TrainingObservations": int(
                        coverage["TrainingObservations"]
                    ),
                    "TrainingCoumarins": int(
                        coverage["TrainingCoumarins"]
                    ),
                })

            cell_line_coumarin_results = pd.DataFrame(
                cell_line_coumarin_results
            )
            
            cell_line_coumarin_results = cell_line_coumarin_results.sort_values(
                ["Coumarin", "MAE"],
                ascending=[True, True],
            ).reset_index(drop=True)

            print(
                f"\n{TRIANGLES} External metrics by cell line "
                "and coumarin:"
            )
            print_dataframe(cell_line_coumarin_results)

            # ----------------------------------------------------------
            # Save results
            # ----------------------------------------------------------

            results = {
                "CVFoldResults": fold_results,
                "CVResults": pd.DataFrame([{
                    "Experiment": experiment_name,
                    "Validation": validation,
                    "FeatureSet": feature_name,
                    "BestCV_R2": best_cv_score,
                    "TrainingObservations": len(experiment_data),
                    "TrainingCoumarins": experiment_data[
                        "Coumarin"
                    ].nunique(),
                }]),
                "ExternalPredictions": external_predictions,
                "External_CellLineResults": cell_line_results,
                "External_CellLineCoumarinResults": (
                    cell_line_coumarin_results
                ),
                **{
                    f"External_{name}": dataframe
                    for name, dataframe in external_results.items()
                },
            }

            save_model_results(
                results=results,
                output_directory=output_directory,
                model_name=model_name,
                best_parameters=best_parameters,
                fitted_model=best_model,
            )
            
plot_model_performance(MODEL_NAME)