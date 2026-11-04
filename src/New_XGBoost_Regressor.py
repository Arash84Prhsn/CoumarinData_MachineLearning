from xgboost import XGBRegressor
import model_utils2 as utils


MODEL_NAME = "XGBoost"

N_ITER = 200
N_JOBS = -1
SCORING = "r2"

FEATURE_CONFIGURATIONS = [
    "Descriptors_BaseLine",
    "Descriptor&Morgans_Baseline",
    "PCA_All_Numerical",
    "PCA_Descriptors_Fingerprints",
    "PCA_Fingerprints",
]

PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [
        10, 20, 30, 50, 80
    ],
    "model__max_depth": [
        2, 3, 4, 5, 6, 8, 10
    ],
    "model__learning_rate": [
        0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2
    ],
    "model__subsample": [
        0.6, 0.7, 0.8, 0.9, 1.0
    ],
    "model__colsample_bytree": [
        0.6, 0.7, 0.8, 0.9, 1.0
    ],
    "model__min_child_weight": [
        1, 2, 3, 5, 7, 10
    ],
    "model__gamma": [
        0, 0.01, 0.05, 0.1, 0.2, 0.5, 1
    ],
    "model__reg_alpha": [
        0, 0.001, 0.01, 0.1, 1, 5, 10
    ],
    "model__reg_lambda": [
        0.1, 0.5, 1, 2, 5, 10, 20
    ],
}


estimator = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    random_state=utils.RANDOM_STATE,
    # n_jobs=1,
)

data = utils.load_data()


# ============================================================
# KFold
# ============================================================

for configuration in FEATURE_CONFIGURATIONS:

    print(f"\n{'=' * 70}")
    print(f"XGBoost | KFold | {configuration}")
    print(f"{'=' * 70}")

    experiment = utils.run_kfold_experiment(
        data=data,
        estimator=estimator,
        param_distributions=PARAM_DISTRIBUTIONS,
        model_name=MODEL_NAME,
        feature_configuration=configuration,
        n_iter=N_ITER,
        scoring=SCORING,
        n_jobs=N_JOBS,
    )

    utils.save_experiment_results(
        experiment,
        model_name=MODEL_NAME,
        feature_configuration=configuration,
        validation="KFold",
    )


# ============================================================
# LOCO
# ============================================================

for configuration in FEATURE_CONFIGURATIONS:

    print(f"\n{'=' * 70}")
    print(f"XGBoost | LOCO | {configuration}")
    print(f"{'=' * 70}")

    experiment = utils.run_loco_experiment(
        data=data,
        estimator=estimator,
        param_distributions=PARAM_DISTRIBUTIONS,
        model_name=MODEL_NAME,
        feature_configuration=configuration,
        n_iter=N_ITER,
        scoring=SCORING,
        n_jobs=N_JOBS,
    )

    utils.save_experiment_results(
        experiment,
        model_name=MODEL_NAME,
        feature_configuration=configuration,
        validation="LOCO",
    )


# ============================================================
# LOGO
# ============================================================

for configuration in FEATURE_CONFIGURATIONS:

    print(f"\n{'=' * 70}")
    print(f"XGBoost | LOGO | {configuration}")
    print(f"{'=' * 70}")

    experiment = utils.run_logo_experiment(
        data=data,
        estimator=estimator,
        param_distributions=PARAM_DISTRIBUTIONS,
        model_name=MODEL_NAME,
        feature_configuration=configuration,
        n_iter=N_ITER,
        scoring=SCORING,
        n_jobs=N_JOBS,
    )

    utils.save_experiment_results(
        experiment,
        model_name=MODEL_NAME,
        feature_configuration=configuration,
        validation="LOGO",
    )


# ============================================================
# Author Holdout
# ============================================================

for configuration in FEATURE_CONFIGURATIONS:

    print(f"\n{'=' * 70}")
    print(f"XGBoost | Author Holdout | {configuration}")
    print(f"{'=' * 70}")

    experiment = utils.run_author_holdout_experiment(
        data=data,
        estimator=estimator,
        param_distributions=PARAM_DISTRIBUTIONS,
        model_name=MODEL_NAME,
        feature_configuration=configuration,
        test_size=0.10,
        n_iter=N_ITER,
        scoring=SCORING,
        n_jobs=N_JOBS,
    )

    utils.save_experiment_results(
        experiment,
        model_name=MODEL_NAME,
        feature_configuration=configuration,
        validation="AuthorHoldout_90_10",
    )


# ============================================================
# Group-Specific LOCO
# ============================================================

for group_name in utils.COUMARIN_GROUPS:

    for configuration in FEATURE_CONFIGURATIONS:

        print(f"\n{'=' * 70}")
        print(f"XGBoost | {group_name} | {configuration}")
        print(f"{'=' * 70}")

        experiment = utils.run_group_specific_experiment(
            data=data,
            estimator=estimator,
            param_distributions=PARAM_DISTRIBUTIONS,
            model_name=MODEL_NAME,
            feature_configuration=configuration,
            group_name=group_name,
            n_iter=N_ITER,
            scoring=SCORING,
            n_jobs=N_JOBS,
        )

        utils.save_experiment_results(
            experiment,
            model_name=MODEL_NAME,
            feature_configuration=configuration,
            validation="GroupSpecific_LOCO",
        )


# ============================================================
# Plot Results
# ============================================================

utils.plot_model_results(MODEL_NAME)