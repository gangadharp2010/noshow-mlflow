import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from noshow import train
from noshow.data import load_frame


ROOT = Path(__file__).resolve().parents[1]


def configure_tracking():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(
            f"sqlite:///{ROOT / 'mlflow.db'}"
        )

    # MLflow Projects already creates and selects the run/experiment.
    # Do not change the experiment when joining a Project run.
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment("no-show")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning-rate", type=float, default=0.1
    )
    parser.add_argument(
        "--max-leaf-nodes", type=int, default=31
    )
    args = parser.parse_args()

    configure_tracking()

    with mlflow.start_run(run_name="noshow-train") as run:
        mlflow.set_tag("stage", "training")

        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("max_leaf_nodes", args.max_leaf_nodes)

        pipeline, scores = train.fit(
            args.learning_rate,
            args.max_leaf_nodes,
        )

        for metric_name, value in scores.items():
            mlflow.log_metric(metric_name, value)

        # Preserve the original feature types, including int64 columns.
        X, _ = load_frame()
        input_example = X.head(3)

        predictions = pipeline.predict_proba(input_example)
        signature = infer_signature(input_example, predictions)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            pyfunc_predict_fn="predict_proba",
            input_example=input_example,
            signature=signature,
            pip_requirements=str(ROOT / "requirements.txt"),
            serialization_format="cloudpickle",
        )

        print(f"test_roc_auc {scores['test_roc_auc']:.4f}")
        print(f"tracked as run {run.info.run_id}")


if __name__ == "__main__":
    main()
