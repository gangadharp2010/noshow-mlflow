"""The two ways this project scores a model."""
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

CV_FOLDS = 3


def holdout_scores(pipeline, X_test, y_test):
    """A fitted pipeline on the holdout rows. Returns a dict of two floats."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "test_average_precision": float(average_precision_score(y_test, proba)),
    }


def cv_roc_auc(pipeline, X_train, y_train):
    """Mean ROC AUC over three stratified folds of the training rows."""
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=7)
    scores = cross_val_score(pipeline, X_train, y_train, cv=folds, scoring="roc_auc")
    return float(scores.mean())
