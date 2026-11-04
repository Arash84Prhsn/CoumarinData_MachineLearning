from sklearn.ensemble import RandomForestRegressor

import model_utils2 as utils


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "RandomForest"

N_ITER = 200
N_JOBS = -1
SCORING = "r2"

CONFIGURATIONS = [
    "Descriptors_BaseLine",
    "Descriptor&Morgans_Baseline",
    "PCA_All_Numerical",
    "PCA_Descriptors_Fingerprints",
    "PCA_Fingerprints",
]

COUMARIN_GROUPS = [
    "Group_1",
    "Group_2",
    "Group_3",
]


# ============================================================
# HYPERPARAMETER DISTRIBUTIONS
# ============================================================

PARAM_DISTRIBUTIONS = [
    # Bootstrap enabled
    {
        "model__n_estimators": [
            10, 20, 30, 50, 80
        ],

        "model__criterion": [
            "squared_error",
            "absolute_error",
            "poisson",
        ],

        "model__max_depth": [
            None, 5, 10, 15, 20, 30, 40,
        ],

        "model__min_samples_split": [
            2, 5, 10, 15, 20, 30,
        ],

        "model__min_samples_leaf": [
            1, 2, 4, 6, 10, 15, 20,
        ],

        "model__max_features": [
            1.0, "sqrt", "log2", 0.25, 0.5, 0.75,
        ],

        "model__bootstrap": [True],

        "model__max_samples": [
            0.6, 0.7, 0.8, 0.9, None,
        ],

        "model__ccp_alpha": [
            0.0, 0.0001, 0.0005, 0.001,
            0.005, 0.01, 0.02,
        ],
    },

    # Bootstrap disabled
    {
        "model__n_estimators": [
            10, 20, 30, 50, 80
        ],

        "model__criterion": [
            "squared_error",
            "absolute_error",
            "poisson",
        ],

        "model__max_depth": [
            None, 5, 10, 15, 20, 30, 40,
        ],

        "model__min_samples_split": [
            2, 5, 10, 15, 20, 30,
        ],

        "model__min_samples_leaf": [
            1, 2, 4, 6, 10, 15, 20,
        ],

        "model__max_features": [
            1.0, "sqrt", "log2", 0.25, 0.5, 0.75,
        ],

        "model__bootstrap": [False],

        "model__ccp_alpha": [
            0.0, 0.0001, 0.0005, 0.001,
            0.005, 0.01, 0.02,
        ],
    },
]


# ============================================================
# DATA AND MODEL
# ============================================================

data = utils.load_data()

estimator = RandomForestRegressor(
    random_state=utils.RANDOM_STATE,
    n_jobs=1,
)


# ============================================================
# 1. KFOLD
# ============================================================

for configuration in CONFIGURATIONS:

    print("\n" + "=" * 80)
    print(f"KFOLD | {configuration}")
    print("=" * 80)

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
        experiment=experiment,
        model_name=MODEL_NAME,
        feature_configuration=configuration,
        validation="KFold",
    )


# ============================================================
# 2. LEAVE-ONE-COUMARIN-OUT (LOCO)
# ============================================================

for configuration in CONFIGURATIONS:

    print("\n" + "=" * 80)
    print(f"LOCO | {configuration}")
    print("=" * 80)

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
        experiment=experiment,
        model_name=MODEL_NAME,
        feature_configuration=configuration,
        validation="LOCO",
    )


# ============================================================
# 3. LEAVE-ONE-COUMARIN-GROUP-OUT (LOGO)
# ============================================================

for configuration in CONFIGURATIONS:

    print("\n" + "=" * 80)
    print(f"LOGO | {configuration}")
    print("=" * 80)

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
        experiment=experiment,
        model_name=MODEL_NAME,
        feature_configuration=configuration,
        validation="LOGO",
    )


# ============================================================
# 4. AUTHOR HOLDOUT (90/10)
# ============================================================

for configuration in CONFIGURATIONS:

    print("\n" + "=" * 80)
    print(f"AUTHOR HOLDOUT | {configuration}")
    print("=" * 80)

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
        experiment=experiment,
        model_name=MODEL_NAME,
        feature_configuration=configuration,
        validation="AuthorHoldout_90_10",
    )


# ============================================================
# 5. GROUP-SPECIFIC LOCO
# ============================================================

for group_name in COUMARIN_GROUPS:

    for configuration in CONFIGURATIONS:

        print("\n" + "=" * 80)
        print(f"GROUP-SPECIFIC LOCO | {group_name} | {configuration}")
        print("=" * 80)

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
            experiment=experiment,
            model_name=MODEL_NAME,
            feature_configuration=configuration,
            validation="GroupSpecific_LOCO",
        )


# ============================================================
# PLOTS
# ============================================================

utils.plot_model_results(MODEL_NAME)