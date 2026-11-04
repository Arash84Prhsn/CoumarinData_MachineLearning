import numpy as np

from sklearn.svm import SVR
import model_utils2 as utils


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

MODEL_NAME = "SVR"

N_ITER = 200
N_JOBS = -1
SCORING = "r2"


# ------------------------------------------------------------
# HYPERPARAMETER SEARCH SPACE
# ------------------------------------------------------------

PARAM_DISTRIBUTIONS = [

    # ---- Linear kernel ----
    # {
    #     "model__kernel": ["linear"],
    #     "model__C": np.logspace(-3, 3, 7),
    #     "model__epsilon": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    # },

    # ---- RBF kernel ----
    {
        "model__kernel": ["rbf"],
        "model__C": np.logspace(-3, 3, 7),
        "model__gamma": np.logspace(-4, 1, 6),
        "model__epsilon": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    },

    # ---- Sigmoid kernel ----
    {
        "model__kernel": ["sigmoid"],
        "model__C": np.logspace(-3, 3, 7),
        "model__gamma": np.logspace(-4, 1, 6),
        "model__coef0": [0.0, 0.5, 1.0],
        "model__epsilon": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    },

    # ---- Polynomial kernel ----
    {
        "model__kernel": ["poly"],
        "model__C": np.logspace(-3, 3, 7),
        "model__gamma": np.logspace(-4, -1, 4),
        "model__degree": [2, 3],
        "model__coef0": [0.0, 0.5, 1.0],
        "model__epsilon": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    },

]


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

data = utils.load_data()


# ------------------------------------------------------------
# RUN EXPERIMENTS
# ------------------------------------------------------------

for feature_configuration in utils.FEATURE_CONFIGURATIONS:

    for validation in ["KFold", "LOCO"]:

        estimator = SVR()

        experiment = utils.run_experiment(
            data=data,
            estimator=estimator,
            param_distributions=PARAM_DISTRIBUTIONS,
            feature_configuration=feature_configuration,
            validation=validation,
            model_name=MODEL_NAME,
            n_iter=N_ITER,
            scoring=SCORING,
            n_jobs=N_JOBS,
        )

        utils.save_experiment_results(
            experiment=experiment,
            model_name=MODEL_NAME,
            feature_configuration=feature_configuration,
            validation=validation,
        )


# ------------------------------------------------------------
# PLOT RESULTS
# ------------------------------------------------------------

utils.plot_model_results(MODEL_NAME)