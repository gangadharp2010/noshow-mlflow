"""Fit the model once and print its holdout scores. Nothing is recorded.

    python3 -m noshow.train
"""
from .data import load_split
from .evaluate import holdout_scores
from .pipeline import build_pipeline

DEFAULT_LEARNING_RATE = 0.1
DEFAULT_MAX_LEAF_NODES = 31


def fit(learning_rate, max_leaf_nodes):
    """Fit on the training rows. Returns (fitted pipeline, holdout scores)."""
    X_train, X_test, y_train, y_test = load_split()
    pipeline = build_pipeline(learning_rate, max_leaf_nodes).fit(X_train, y_train)
    return pipeline, holdout_scores(pipeline, X_test, y_test)


def main():
    _, scores = fit(DEFAULT_LEARNING_RATE, DEFAULT_MAX_LEAF_NODES)
    for name, value in scores.items():
        print("%-24s %.4f" % (name, value))


if __name__ == "__main__":
    main()
