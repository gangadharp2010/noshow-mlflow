"""Search the two settings with Optuna. Nothing is recorded.

    python3 -m noshow.search --trials 8

make_objective() returns the function Optuna calls once per trial. It picks
the trial's settings with suggest_settings and returns their cross validated
ROC AUC on the training rows.
"""
import argparse

import optuna

from .data import load_split
from .evaluate import cv_roc_auc
from .pipeline import build_pipeline

SEED = 11


def make_study():
    """A new, empty study that maximises the score. The sampler is seeded."""
    return optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )


def suggest_settings(trial):
    """The search space. Returns {"learning_rate": float, "max_leaf_nodes": int}."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 64),
    }


def make_objective():
    """The objective Optuna optimises. It loads the training rows once."""
    X_train, _, y_train, _ = load_split()

    def objective(trial):
        settings = suggest_settings(trial)
        return cv_roc_auc(build_pipeline(**settings), X_train, y_train)

    return objective


def main():
    parser = argparse.ArgumentParser(description="Search without recording.")
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = make_study()
    study.optimize(make_objective(), n_trials=args.trials)
    print("best cv_roc_auc %.4f with %s" % (study.best_value, study.best_params))


if __name__ == "__main__":
    main()
