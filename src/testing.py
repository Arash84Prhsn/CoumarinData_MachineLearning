import json

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
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
)

MODEL_NAME = "DecisionTree"

EXTERNAL_COUMARINS = [
    "scopoletin",
    "imperatorin",
    "farnesiferola",
    "farnesiferolb",
    "farnesiferolc",
]

VALIDATION_METHODS = ["KFold", "LOCO"]

param_distributions = {
    "model__max_depth": [None, 3, 5, 7, 10, 15, 20, 30],
    "model__min_samples_split": [2, 5, 10, 20, 30],
    "model__min_samples_leaf": [1, 2, 4, 8, 12, 16],
    "model__max_features": [None, "sqrt", "log2"],
    "model__criterion": [
        "poisson",
        "squared_error",
        "absolute_error",
    ],
    "model__ccp_alpha": [0.0, 0.001, 0.005, 0.01, 0.02],
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
                (
                    "model",
                    DecisionTreeRegressor(random_state=RANDOM_STATE),
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
                best_model.predict(external_data[features])
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
            
# ============================================================
# MODEL PERFORMANCE PLOTS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


print(f"\n{'=' * 70}")
print("GENERATING MODEL PERFORMANCE PLOTS")
print(f"{'=' * 70}")

RESULTS_DIRECTORY = RESULTS_DIR / "Regression" / MODEL_NAME
PLOTS_DIRECTORY = RESULTS_DIRECTORY / "Performance Plots"

COUMARIN_PLOTS_DIRECTORY = PLOTS_DIRECTORY / "Coumarin Comparisons"
DESCRIPTOR_MORGAN_DIRECTORY = PLOTS_DIRECTORY / "Descriptor vs Morgan"
CELL_LINE_COUMARIN_DIRECTORY = PLOTS_DIRECTORY / "CellLine Coumarin Heatmaps"

COUMARIN_PLOTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
DESCRIPTOR_MORGAN_DIRECTORY.mkdir(parents=True, exist_ok=True)
CELL_LINE_COUMARIN_DIRECTORY.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1. Load saved external evaluation results
# ------------------------------------------------------------

coumarin_results = []
cell_line_coumarin_results = []

for csv_path in RESULTS_DIRECTORY.rglob("*.csv"):

    if "Performance Plots" in csv_path.parts:
        continue

    if "External Metric Results" not in csv_path.parts:
        continue

    df = pd.read_csv(csv_path)

    relative_parts = csv_path.relative_to(RESULTS_DIRECTORY).parts
    metric_directory_index = relative_parts.index("External Metric Results")
    configuration_parts = relative_parts[:metric_directory_index]

    if not configuration_parts:
        continue

    configuration_name = configuration_parts[-1]

    if configuration_name not in {
        "KFold_Descriptor",
        "KFold_Morgan",
        "LOCO_Descriptor",
        "LOCO_Morgan",
    }:
        continue

    validation, feature_set = configuration_name.split("_", maxsplit=1)

    if len(configuration_parts) == 1:
        experiment = "AllTrainingCoumarins"
    else:
        experiment = " | ".join(configuration_parts[:-1])

    model_label = f"{experiment} | {validation} | {feature_set}"

    # External MAE by coumarin.
    # Exclude coverage-specific results and cell-line-level results.
    if (
        "Coumarin" in df.columns
        and "MAE" in df.columns
        and "CellLine" not in df.columns
        and "Coverage" not in csv_path.stem
        and "ByCoumarin" in csv_path.stem
    ):
        result = df[["Coumarin", "MAE"]].copy()
        result["Experiment"] = experiment
        result["Validation"] = validation
        result["FeatureSet"] = feature_set
        result["Model"] = model_label

        coumarin_results.append(result)

    # External MAE by CellLine + Coumarin.
    if (
        {"CellLine", "Coumarin", "MAE"}.issubset(df.columns)
        and "CellLineCoumarin" in csv_path.stem
    ):
        result = df[["CellLine", "Coumarin", "MAE"]].copy()
        result["Experiment"] = experiment
        result["Validation"] = validation
        result["FeatureSet"] = feature_set
        result["Model"] = model_label

        cell_line_coumarin_results.append(result)


if not coumarin_results:
    print(
        "\nNo external by-coumarin metric files were found. "
        "Check the filenames saved by save_model_results()."
    )

else:

    coumarin_df = pd.concat(coumarin_results, ignore_index=True)

    coumarin_df["MAE"] = pd.to_numeric(
        coumarin_df["MAE"],
        errors="coerce",
    )

    coumarin_df = coumarin_df.dropna(subset=["MAE"])


    # --------------------------------------------------------
    # 2. Separate sorted cluster plots:
    #    Descriptor models and Morgan models
    # --------------------------------------------------------

    for feature_set in ["Descriptor", "Morgan"]:

        
        feature_data = coumarin_df[
            (coumarin_df["FeatureSet"] == feature_set)
            & (coumarin_df["Validation"] == "LOCO")
        ].copy()


        if feature_data.empty:
            continue

        pivot = feature_data.pivot_table(
            index="Coumarin",
            columns="Model",
            values="MAE",
            aggfunc="mean",
        )

        # Sort coumarins by their average MAE across the
        # configurations included in this plot.
        pivot = pivot.loc[
            pivot.mean(axis=1).sort_values().index
        ]

        ax = pivot.plot(
            kind="bar",
            figsize=(max(12, len(pivot) * 2.2), 8),
            width=0.85,
        )

        ax.set_title(
            f"External MAE by Coumarin — {feature_set} Models",
            fontsize=15,
        )
        ax.set_xlabel("External Coumarin")
        ax.set_ylabel("MAE (lower is better)")
        ax.tick_params(axis="x", rotation=25)

        for container in ax.containers:
            labels = [
                f"{bar.get_height():.2f}"
                if np.isfinite(bar.get_height())
                else ""
                for bar in container
            ]

            ax.bar_label(
                container,
                labels=labels,
                padding=3,
                fontsize=8,
                rotation=90,
            )

        ax.legend(
            title="Experiment | Validation | Features",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

        plt.tight_layout()

        plt.savefig(
            COUMARIN_PLOTS_DIRECTORY
            / f"External_MAE_by_Coumarin_{feature_set}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()
        plt.close()


    # --------------------------------------------------------
    # 3. Descriptor vs Morgan:
    #    matched plots for each experiment and validation
    # --------------------------------------------------------

    for experiment in coumarin_df["Experiment"].unique():

        for validation in ["KFold", "LOCO"]:

            comparison_data = coumarin_df[
                (coumarin_df["Experiment"] == experiment)
                & (coumarin_df["Validation"] == validation)
            ]

            descriptor_data = comparison_data[
                comparison_data["FeatureSet"] == "Descriptor"
            ]

            morgan_data = comparison_data[
                comparison_data["FeatureSet"] == "Morgan"
            ]

            if descriptor_data.empty or morgan_data.empty:
                continue

            # Use only coumarins evaluated for both feature sets.
            common_coumarins = sorted(
                set(descriptor_data["Coumarin"])
                & set(morgan_data["Coumarin"])
            )

            if not common_coumarins:
                continue

            paired_data = comparison_data[
                comparison_data["Coumarin"].isin(common_coumarins)
            ]

            pivot = paired_data.pivot_table(
                index="Coumarin",
                columns="FeatureSet",
                values="MAE",
                aggfunc="mean",
            )

            # Ensure the two feature sets are both represented.
            if not {"Descriptor", "Morgan"}.issubset(pivot.columns):
                continue

            # Sort by mean MAE across the paired feature sets.
            pivot = pivot.sort_values(
                by=["Descriptor", "Morgan"],
                ascending=True,
            )

            ax = pivot[["Descriptor", "Morgan"]].plot(
                kind="bar",
                figsize=(max(10, len(pivot) * 1.8), 7),
                width=0.8,
            )

            ax.set_title(
                f"Descriptor vs Morgan — {experiment} | {validation}",
                fontsize=14,
            )
            ax.set_xlabel("External Coumarin")
            ax.set_ylabel("MAE (lower is better)")
            ax.tick_params(axis="x", rotation=25)

            for container in ax.containers:
                labels = [
                    f"{bar.get_height():.2f}"
                    if np.isfinite(bar.get_height())
                    else ""
                    for bar in container
                ]

                ax.bar_label(
                    container,
                    labels=labels,
                    padding=3,
                    fontsize=9,
                )

            ax.legend(title="Feature Set")

            plt.tight_layout()

            safe_experiment = "".join(
                char if char.isalnum() else "_"
                for char in experiment
            )

            plt.savefig(
                DESCRIPTOR_MORGAN_DIRECTORY
                / f"Descriptor_vs_Morgan_{safe_experiment}_{validation}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.show()
            plt.close()


# ------------------------------------------------------------
# 4. CellLine + Coumarin heatmaps
#    Save only; do not display
# ------------------------------------------------------------

if cell_line_coumarin_results:

    cell_line_coumarin_df = pd.concat(
        cell_line_coumarin_results,
        ignore_index=True,
    )

    cell_line_coumarin_df["MAE"] = pd.to_numeric(
        cell_line_coumarin_df["MAE"],
        errors="coerce",
    )

    cell_line_coumarin_df = cell_line_coumarin_df.dropna(
        subset=["MAE"]
    )

    for model_label, model_data in cell_line_coumarin_df.groupby("Model"):

        heatmap_data = model_data.pivot_table(
            index="CellLine",
            columns="Coumarin",
            values="MAE",
            aggfunc="mean",
        )

        if heatmap_data.empty:
            continue

        fig, ax = plt.subplots(
            figsize=(
                max(8, len(heatmap_data.columns) * 2),
                max(6, len(heatmap_data.index) * 0.45),
            )
        )

        image = ax.imshow(
            heatmap_data.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_xticklabels(
            heatmap_data.columns,
            rotation=35,
            ha="right",
        )

        ax.set_yticks(np.arange(len(heatmap_data.index)))
        ax.set_yticklabels(heatmap_data.index)

        ax.set_title(
            f"External MAE by Cell Line and Coumarin\n{model_label}"
        )
        ax.set_xlabel("External Coumarin")
        ax.set_ylabel("Cell Line")

        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("MAE (lower is better)")

        values = heatmap_data.to_numpy(dtype=float)

        for row in range(values.shape[0]):
            for col in range(values.shape[1]):

                value = values[row, col]

                if np.isfinite(value):
                    ax.text(
                        col,
                        row,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        fig.tight_layout()

        safe_model_label = "".join(
            char if char.isalnum() else "_"
            for char in model_label
        )

        fig.savefig(
            CELL_LINE_COUMARIN_DIRECTORY
            / f"External_MAE_{safe_model_label}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


print(f"\n{TRIANGLES} All performance plots have been saved.")
print(f"Plot directory: {PLOTS_DIRECTORY}")
print(f"Coumarin comparisons: {COUMARIN_PLOTS_DIRECTORY}")
print(f"Descriptor vs Morgan: {DESCRIPTOR_MORGAN_DIRECTORY}")
print(f"CellLine-Coumarin heatmaps: {CELL_LINE_COUMARIN_DIRECTORY}")