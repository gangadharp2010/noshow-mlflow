"""The model: one-hot categories in front of gradient boosted trees.

The numeric columns pass straight through. The trees handle the missing
days_since_last_visit themselves, so nothing is imputed.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MAX_ITER = 120


def build_pipeline(learning_rate, max_leaf_nodes):
    """An unfitted pipeline for one pair of settings."""
    preprocess = ColumnTransformer(
        [
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
            ("numbers", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        max_iter=MAX_ITER,
        early_stopping=False,
        random_state=0,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])
