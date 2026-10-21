import argparse
import os
from pathlib import Path

import mlflow
from noshow import search


ROOT = Path(__file__).resolve().parents[1]


def configure_tracking():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(
            f"sqlite:///{ROOT / 'mlflow.db'}"
        )

    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment("no-show")


def tracked(objective):
    def wrapped(trial):
        with mlflow.start_run(
            run_name=f"trial-{trial.number:03d}",
            nested=True,
        ):
            mlflow.set_tag("stage", "trial")

            # If the objective raises, let the exception propagate.
            # MLflow closes this child run as FAILED.
            value = objective(trial)

            if trial.params:
                mlflow.log_params(trial.params)

            mlflow.log_metric("cv_roc_auc", value)

            return value

    return wrapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()

    configure_tracking()

    with mlflow.start_run(run_name="noshow-search") as run:
        mlflow.set_tag("stage", "search")
        mlflow.log_param("trials", args.trials)

        study = search.make_study()

        study.optimize(
            tracked(search.make_objective()),
            n_trials=args.trials,
        )

        mlflow.log_metric(
            "best_cv_roc_auc",
            study.best_value,
        )

        print(
            f"best cv_roc_auc {study.best_value:.4f} "
            f"with {study.best_params}"
        )
        print(
            f"tracked as run {run.info.run_id} "
            f"with {len(study.trials)} child runs"
        )


if __name__ == "__main__":
    main()

