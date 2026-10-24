from pathlib import Path
import json
import re
import joblib

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, LeaveOneGroupOut, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from tabulate import tabulate


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"

RANDOM_STATE = 42
TARGET_COLUMN = "Viability"
GROUP_COLUMN = "Coumarin"
CELL_LINE_COLUMN = "CellLine"

DEFAULT_N_SPLITS = 5
DEFAULT_N_ITER = 200
DEFAULT_COVERAGE_THRESHOLDS = [25, 50, 75, 100]
DEFAULT_MIN_TRAINING_COUMARINS = 4

TRIANGLES = "\u001b[32m\u001b[1m" + "▶▶▶▶" + "\u001b[0m\u001b[22m"

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

EXPERIMENTAL_FEATURES = ["LogDose", "Time"]
CATEGORICAL_FEATURES = [CELL_LINE_COLUMN]
DESCRIPTOR_FEATURES = EXPERIMENTAL_FEATURES + MOLECULAR_FEATURES


def normalize_identifiers(data):
    data = data.copy()

    for column in [GROUP_COLUMN, CELL_LINE_COLUMN]:
        if column in data.columns:
            data[column] = data[column].astype("string").str.strip().str.lower()

    return data


def get_morgan_features(data):
    return sorted(
        [column for column in data.columns if re.fullmatch(r"Morgan_\d+", column)],
        key=lambda column: int(column.split("_")[1]),
    )


def get_feature_sets(data):
    missing = [
        column
        for column in DESCRIPTOR_FEATURES + CATEGORICAL_FEATURES
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    morgan_features = get_morgan_features(data)

    if not morgan_features:
        raise ValueError("No Morgan fingerprint columns were found.")

    return {
        "Descriptor": DESCRIPTOR_FEATURES + CATEGORICAL_FEATURES,
        "Morgan": DESCRIPTOR_FEATURES + morgan_features + CATEGORICAL_FEATURES,
    }


def validate_features(data, features, target=TARGET_COLUMN):
    required = features + [target]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if data[required].isna().any().any():
        missing_values = data[required].isna().sum()
        missing_values = missing_values[missing_values > 0]
        raise ValueError(f"Missing values detected:\n{missing_values}")


def build_preprocessor(features):
    categorical = [
        column for column in CATEGORICAL_FEATURES if column in features
    ]
    numeric = [column for column in features if column not in categorical]

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("numeric", "passthrough", numeric),
        ],
        remainder="drop",
    )


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return {
            "N": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MedAE": np.nan,
            "R2": np.nan,
        }

    return {
        "N": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MedAE": median_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan,
    }


def split_training_external(data, external_coumarins):
    data = normalize_identifiers(data)

    external_coumarins = [
        compound.strip().lower() for compound in external_coumarins
    ]

    external_mask = data[GROUP_COLUMN].isin(external_coumarins)

    training_data = data.loc[~external_mask].copy()
    external_data = data.loc[external_mask].copy()

    if training_data.empty:
        raise ValueError("Training dataset is empty.")

    if external_data.empty:
        raise ValueError("External dataset is empty.")

    overlap = set(training_data[GROUP_COLUMN].unique()) & set(
        external_data[GROUP_COLUMN].unique()
    )

    if overlap:
        raise ValueError(f"External compounds leaked into training: {overlap}")

    return training_data, external_data


def run_cv_experiment(
    data,
    features,
    estimator,
    param_distributions,
    validation="KFold",
    n_splits=DEFAULT_N_SPLITS,
    n_iter=DEFAULT_N_ITER,
):
    validate_features(data, features)

    X = data[features]
    y = data[TARGET_COLUMN]

    if validation == "KFold":
        outer_cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        inner_cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        groups = None

    elif validation == "LOCO":
        outer_cv = LeaveOneGroupOut()
        inner_cv = LeaveOneGroupOut()
        groups = data[GROUP_COLUMN]

    else:
        raise ValueError("validation must be 'KFold' or 'LOCO'.")

    predictions = []
    fold_results = []

    splits = (
        outer_cv.split(X, y)
        if groups is None
        else outer_cv.split(X, y, groups)
    )

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if groups is not None:
            train_groups = groups.iloc[train_idx]
            held_out = groups.iloc[test_idx].iloc[0]

            print(
                f"{validation} fold {fold}: "
                f"held-out coumarin = {held_out}"
            )
        else:
            train_groups = None
            held_out = None

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="r2",
            cv=inner_cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        if train_groups is None:
            search.fit(X_train, y_train)
        else:
            search.fit(X_train, y_train, groups=train_groups)

        y_pred = search.predict(X_test)

        fold_predictions = pd.DataFrame(
            {
                "RowIndex": data.index[test_idx],
                "Actual": y_test.to_numpy(),
                "Predicted": y_pred,
                "Fold": fold,
                "Validation": validation,
                GROUP_COLUMN: data[GROUP_COLUMN].iloc[test_idx].to_numpy(),
            }
        )

        predictions.append(fold_predictions)

        fold_metrics = calculate_metrics(y_test, y_pred)
        fold_metrics.update(
            {
                "Fold": fold,
                "Validation": validation,
                "HeldOutCoumarin": held_out,
                # "BestParameters": str(search.best_params_),
            }
        )
        fold_results.append(fold_metrics)

    predictions_df = pd.concat(predictions, ignore_index=True)
    fold_results_df = pd.DataFrame(fold_results)

    summary = calculate_metrics(
        predictions_df["Actual"],
        predictions_df["Predicted"],
    )
    summary.update(
        {
            "Validation": validation,
            "FeatureCount": len(features),
        }
    )

    return {
        "Predictions": predictions_df,
        "FoldResults": fold_results_df,
        "Summary": pd.DataFrame([summary]),
    }


def fit_final_model(
    data,
    features,
    estimator,
    param_distributions,
    validation="KFold",
    n_splits=DEFAULT_N_SPLITS,
    n_iter=DEFAULT_N_ITER,
):
    validate_features(data, features)

    X = data[features]
    y = data[TARGET_COLUMN]

    if validation == "KFold":
        cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        groups = None

    elif validation == "LOCO":
        cv = LeaveOneGroupOut()
        groups = data[GROUP_COLUMN]

    else:
        raise ValueError("validation must be 'KFold' or 'LOCO'.")

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="r2",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    if groups is None:
        search.fit(X, y)
    else:
        search.fit(X, y, groups=groups)

    return search.best_estimator_, search.best_params_


def calculate_training_coverage(training_data):
    coverage = (
        training_data.groupby(CELL_LINE_COLUMN)
        .agg(
            TrainingObservations=(TARGET_COLUMN, "size"),
            TrainingCoumarins=(GROUP_COLUMN, "nunique"),
        )
        .reset_index()
    )

    return coverage

def evaluate_external_model(
    model,
    training_data,
    external_data,
    features,
    coverage_thresholds=DEFAULT_COVERAGE_THRESHOLDS,
    min_training_coumarins=DEFAULT_MIN_TRAINING_COUMARINS,
):
    validate_features(training_data, features)
    validate_features(external_data, features)

    external_predictions = external_data[
        [GROUP_COLUMN, CELL_LINE_COLUMN, TARGET_COLUMN]
    ].copy()

    external_predictions["Predicted"] = model.predict(
        external_data[features]
    )

    overall_metrics = calculate_metrics(
        external_predictions[TARGET_COLUMN],
        external_predictions["Predicted"],
    )

    overall_results = pd.DataFrame([overall_metrics])

    by_coumarin = []

    for coumarin, group in external_predictions.groupby(GROUP_COLUMN):
        metrics = calculate_metrics(
            group[TARGET_COLUMN],
            group["Predicted"],
        )
        metrics[GROUP_COLUMN] = coumarin
        by_coumarin.append(metrics)

    by_coumarin_results = pd.DataFrame(by_coumarin)

    coverage = calculate_training_coverage(training_data)

    external_predictions = external_predictions.merge(
        coverage,
        on=CELL_LINE_COLUMN,
        how="left",
    )

    external_predictions[
        ["TrainingObservations", "TrainingCoumarins"]
    ] = external_predictions[
        ["TrainingObservations", "TrainingCoumarins"]
    ].fillna(0)

    coverage_summary = []
    coverage_by_coumarin = []

    for threshold in coverage_thresholds:
        eligible = external_predictions[
            (external_predictions["TrainingObservations"] >= threshold)
            & (
                external_predictions["TrainingCoumarins"]
                >= min_training_coumarins
            )
        ]

        metrics = calculate_metrics(
            eligible[TARGET_COLUMN],
            eligible["Predicted"],
        )
        metrics["MinimumTrainingObservations"] = threshold
        metrics["MinimumTrainingCoumarins"] = min_training_coumarins
        coverage_summary.append(metrics)

        for coumarin, group in eligible.groupby(GROUP_COLUMN):
            compound_metrics = calculate_metrics(
                group[TARGET_COLUMN],
                group["Predicted"],
            )
            compound_metrics[GROUP_COLUMN] = coumarin
            compound_metrics["MinimumTrainingObservations"] = threshold
            compound_metrics["MinimumTrainingCoumarins"] = (
                min_training_coumarins
            )
            coverage_by_coumarin.append(compound_metrics)

    coverage_by_coumarin_results = pd.DataFrame(coverage_by_coumarin)

    if coverage_by_coumarin_results.empty:
        coverage_by_coumarin_results = pd.DataFrame(
            columns=[
                GROUP_COLUMN,
                "N",
                "MAE",
                "RMSE",
                "MedAE",
                "R2",
                "MinimumTrainingObservations",
                "MinimumTrainingCoumarins",
            ]
        )
    else:
        coverage_by_coumarin_results = (
            coverage_by_coumarin_results
            .sort_values(
                [GROUP_COLUMN, "MinimumTrainingObservations"]
            )
            .reset_index(drop=True)
        )

    return {
        "Overall": overall_results,
        "ByCoumarin": by_coumarin_results,
        "CoverageSummary": pd.DataFrame(coverage_summary),
        "CoverageByCoumarin": coverage_by_coumarin_results,
        "ExternalPredictions": external_predictions,
    }

def save_model_results(
    results,
    output_directory,
    model_name,
    best_parameters=None,
    fitted_model=None,
):
    config_directory = Path(output_directory) / model_name

    directories = {
        "predictions": config_directory / "External Prediction Results",
        "metrics": config_directory / "External Metric Results",
        "search": config_directory / "Random Search Results",
        "models": config_directory / "Model Files",
        "other": config_directory / "Other Results",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    for name, dataframe in results.items():
        if not isinstance(dataframe, pd.DataFrame):
            continue

        if name.startswith("ExternalPredictions"):
            directory = directories["predictions"]
        elif name.startswith("External_"):
            directory = directories["metrics"]
        elif name in {"CVResults", "CVFoldResults"}:
            directory = directories["search"]
        else:
            directory = directories["other"]

        dataframe.to_csv(directory / f"{name}.csv", index=False)

    if best_parameters is not None:
        with open(
            directories["search"] / "BestParameters.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(best_parameters, file, indent=4, default=str)

    if fitted_model is not None:
        joblib.dump(
            fitted_model,
            directories["models"] / f"{model_name}.joblib",
        )

    print(f"\n{TRIANGLES} Results saved to: {config_directory}")
        
def print_dataframe(df: pd.DataFrame):
    print(tabulate(df, headers='keys', tablefmt='rounded_grid'))
    
def build_scaled_preprocessor(features):
    categorical = [column for column in CATEGORICAL_FEATURES if column in features]
    numeric = [column for column in features if column not in categorical]

    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", StandardScaler(), numeric),
        ],
        remainder="drop",
    )

def plot_model_performance(model_name: str) -> None:
    """
    Generate external-evaluation plots for a regression model.

    Main clustered bar plots:
        - External MAE by coumarin, LOCO only.
        - External R2 by coumarin, LOCO only.
        - Separate Descriptor and Morgan plots.

    Additional plots:
        - Descriptor vs Morgan comparisons for KFold and LOCO.
        - CellLine-Coumarin MAE heatmaps, saved without displaying them.

    All plots are saved under:
        Results/Regression/<model_name>/Performance Plots/
    """
    import re

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    results_directory = RESULTS_DIR / "Regression" / model_name
    plots_directory = results_directory / "Performance Plots"

    coumarin_plots_directory = plots_directory / "Coumarin Comparisons"
    descriptor_morgan_directory = plots_directory / "Descriptor vs Morgan"
    heatmaps_directory = plots_directory / "CellLine Coumarin Heatmaps"

    for directory in [
        coumarin_plots_directory,
        descriptor_morgan_directory,
        heatmaps_directory,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    print(
        f"\n{'=' * 70}\n"
        f"GENERATING PERFORMANCE PLOTS | {model_name}\n"
        f"{'=' * 70}"
    )

    coumarin_results = []
    cell_line_coumarin_results = []

    # ============================================================
    # 1. LOAD EXTERNAL METRIC RESULTS
    # ============================================================

    for csv_path in results_directory.rglob("*.csv"):

        if "Performance Plots" in csv_path.parts:
            continue

        if "External Metric Results" not in csv_path.parts:
            continue

        try:
            df = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError):
            continue

        relative_parts = csv_path.relative_to(results_directory).parts

        metric_directory_index = relative_parts.index(
            "External Metric Results"
        )

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

        validation, feature_set = configuration_name.split(
            "_",
            maxsplit=1,
        )

        if len(configuration_parts) == 1:
            experiment = "AllTrainingCoumarins"
        else:
            experiment = " | ".join(configuration_parts[:-1])

        model_label = (
            f"{experiment} | {validation} | {feature_set}"
        )

        # --------------------------------------------------------
        # External metrics grouped by coumarin
        # --------------------------------------------------------

        if (
            "Coumarin" in df.columns
            and "MAE" in df.columns
            and "CellLine" not in df.columns
            and "Coverage" not in csv_path.stem
            and "ByCoumarin" in csv_path.stem
        ):
            result = df.copy()

            result["Experiment"] = experiment
            result["Validation"] = validation
            result["FeatureSet"] = feature_set
            result["Model"] = model_label

            coumarin_results.append(result)

        # --------------------------------------------------------
        # External metrics grouped by CellLine + Coumarin
        # --------------------------------------------------------

        if (
            {"CellLine", "Coumarin", "MAE"}.issubset(df.columns)
            and "CellLineCoumarin" in csv_path.stem
        ):
            result = df[
                ["CellLine", "Coumarin", "MAE"]
            ].copy()

            result["Experiment"] = experiment
            result["Validation"] = validation
            result["FeatureSet"] = feature_set
            result["Model"] = model_label

            cell_line_coumarin_results.append(result)

    if not coumarin_results:
        print(
            f"\n{TRIANGLES} No external by-coumarin metric files found."
        )
        print("Check the saved filenames and output directories.")
        return

    coumarin_df = pd.concat(
        coumarin_results,
        ignore_index=True,
    )

    for metric in ["MAE", "R2"]:
        if metric in coumarin_df.columns:
            coumarin_df[metric] = pd.to_numeric(
                coumarin_df[metric],
                errors="coerce",
            )

    # ============================================================
    # 2. MAIN CLUSTERED BAR PLOTS
    #    LOCO only; separate Descriptor and Morgan plots
    #    Metrics: MAE and R2
    # ============================================================

    for metric in ["MAE", "R2"]:

        if metric not in coumarin_df.columns:
            continue

        for feature_set in ["Descriptor", "Morgan"]:

            feature_data = coumarin_df[
                (coumarin_df["FeatureSet"] == feature_set)
                & (coumarin_df["Validation"] == "LOCO")
            ].copy()

            feature_data = feature_data.dropna(
                subset=["Coumarin", metric]
            )

            if feature_data.empty:
                continue

            pivot = feature_data.pivot_table(
                index="Coumarin",
                columns="Model",
                values=metric,
                aggfunc="mean",
            )

            # MAE: lower is better.
            # R2: higher is better.
            pivot = pivot.loc[
                pivot.mean(axis=1).sort_values(
                    ascending=(metric == "MAE")
                ).index
            ]

            fig, ax = plt.subplots(
                figsize=(max(12, len(pivot) * 2.2), 8)
            )

            pivot.plot(
                kind="bar",
                ax=ax,
                width=0.78,
                edgecolor="#303030",
                linewidth=0.7,
                alpha=0.92,
            )

            direction = (
                "lower is better"
                if metric == "MAE"
                else "higher is better"
            )

            ax.set_title(
                f"External {metric} by Coumarin — "
                f"{feature_set} Models (LOCO)",
                fontsize=15,
                fontweight="bold",
                pad=16,
            )

            ax.set_xlabel(
                "External Coumarin",
                fontsize=11,
                labelpad=10,
            )

            ax.set_ylabel(
                f"{metric} ({direction})",
                fontsize=11,
                labelpad=10,
            )

            # Subtle horizontal gridlines behind the bars.
            ax.set_axisbelow(True)

            ax.grid(
                axis="y",
                linestyle="--",
                linewidth=0.6,
                alpha=0.35,
            )

            # Remove unnecessary borders.
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_alpha(0.4)
            ax.spines["bottom"].set_alpha(0.4)

            ax.tick_params(
                axis="both",
                labelsize=10,
            )

            ax.tick_params(
                axis="x",
                rotation=25,
            )

            # Add numerical labels above bars.
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
                frameon=False,
            )

            fig.tight_layout()

            fig.savefig(
                coumarin_plots_directory
                / (
                    f"External_{metric}_by_Coumarin_"
                    f"{feature_set}_LOCO.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )

            plt.show()
            plt.close(fig)

    # ============================================================
    # 3. DESCRIPTOR VS MORGAN COMPARISONS
    #    Both KFold and LOCO; metrics: MAE and R2
    # ============================================================

    for metric in ["MAE", "R2"]:

        if metric not in coumarin_df.columns:
            continue

        for experiment in coumarin_df["Experiment"].unique():

            for validation in ["KFold", "LOCO"]:

                comparison_data = coumarin_df[
                    (coumarin_df["Experiment"] == experiment)
                    & (coumarin_df["Validation"] == validation)
                ].copy()

                descriptor_data = comparison_data[
                    comparison_data["FeatureSet"] == "Descriptor"
                ]

                morgan_data = comparison_data[
                    comparison_data["FeatureSet"] == "Morgan"
                ]

                if descriptor_data.empty or morgan_data.empty:
                    continue

                common_coumarins = sorted(
                    set(descriptor_data["Coumarin"])
                    & set(morgan_data["Coumarin"])
                )

                if not common_coumarins:
                    continue

                paired_data = comparison_data[
                    comparison_data["Coumarin"].isin(common_coumarins)
                ].dropna(subset=[metric])

                pivot = paired_data.pivot_table(
                    index="Coumarin",
                    columns="FeatureSet",
                    values=metric,
                    aggfunc="mean",
                )

                if not {"Descriptor", "Morgan"}.issubset(
                    pivot.columns
                ):
                    continue

                pivot = pivot[["Descriptor", "Morgan"]]

                pivot["_MeanMetric"] = pivot.mean(axis=1)

                pivot = pivot.sort_values(
                    "_MeanMetric",
                    ascending=(metric == "MAE"),
                ).drop(columns="_MeanMetric")

                fig, ax = plt.subplots(
                    figsize=(max(10, len(pivot) * 1.8), 7)
                )

                pivot.plot(
                    kind="bar",
                    ax=ax,
                    width=0.72,
                    edgecolor="#303030",
                    linewidth=0.7,
                    alpha=0.92,
                )

                direction = (
                    "lower is better"
                    if metric == "MAE"
                    else "higher is better"
                )

                ax.set_title(
                    f"Descriptor vs Morgan — {metric}\n"
                    f"{experiment} | {validation}",
                    fontsize=14,
                    fontweight="bold",
                    pad=16,
                )

                ax.set_xlabel(
                    "External Coumarin",
                    fontsize=11,
                    labelpad=10,
                )

                ax.set_ylabel(
                    f"{metric} ({direction})",
                    fontsize=11,
                    labelpad=10,
                )

                ax.set_axisbelow(True)

                ax.grid(
                    axis="y",
                    linestyle="--",
                    linewidth=0.6,
                    alpha=0.35,
                )

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_alpha(0.4)
                ax.spines["bottom"].set_alpha(0.4)

                ax.tick_params(
                    axis="both",
                    labelsize=10,
                )

                ax.tick_params(
                    axis="x",
                    rotation=25,
                )

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

                ax.legend(
                    title="Feature Set",
                    frameon=False,
                )

                fig.tight_layout()

                safe_experiment = re.sub(
                    r"[^A-Za-z0-9]+",
                    "_",
                    experiment,
                ).strip("_")

                fig.savefig(
                    descriptor_morgan_directory
                    / (
                        f"Descriptor_vs_Morgan_{metric}_"
                        f"{safe_experiment}_{validation}.png"
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )

                plt.show()
                plt.close(fig)

    # ============================================================
    # 4. CELL LINE + COUMARIN MAE HEATMAPS
    #    Save only; do not display
    # ============================================================

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

        for model_label, model_data in (
            cell_line_coumarin_df.groupby("Model")
        ):

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

            heatmap_values = heatmap_data.to_numpy(dtype=float)

            image = ax.imshow(
                heatmap_values,
                aspect="auto",
                interpolation="nearest",
            )

            ax.set_xticks(
                np.arange(len(heatmap_data.columns))
            )

            ax.set_xticklabels(
                heatmap_data.columns,
                rotation=35,
                ha="right",
            )

            ax.set_yticks(
                np.arange(len(heatmap_data.index))
            )

            ax.set_yticklabels(heatmap_data.index)

            ax.set_title(
                f"External MAE by Cell Line and Coumarin\n"
                f"{model_label}",
                fontsize=12,
                fontweight="bold",
                pad=12,
            )

            ax.set_xlabel("External Coumarin")
            ax.set_ylabel("Cell Line")

            colorbar = fig.colorbar(image, ax=ax)
            colorbar.set_label("MAE (lower is better)")

            for row in range(heatmap_values.shape[0]):
                for col in range(heatmap_values.shape[1]):

                    value = heatmap_values[row, col]

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

            safe_model_label = re.sub(
                r"[^A-Za-z0-9]+",
                "_",
                model_label,
            ).strip("_")

            fig.savefig(
                heatmaps_directory
                / f"External_MAE_{safe_model_label}.png",
                dpi=300,
                bbox_inches="tight",
            )

            # Intentionally do not call plt.show() for heatmaps.
            plt.close(fig)

    print(f"\n{TRIANGLES} Performance plots saved.")
    print(f"Plot directory: {plots_directory}")