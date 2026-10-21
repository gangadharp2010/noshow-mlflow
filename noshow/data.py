"""Loading the appointments table.

Every column the model sees is listed here once, so the rest of the project
never spells a column name twice.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "appointments.csv"

# Whole numbers. They load as int64.
INTEGER_FEATURES = ["age", "lead_days", "prior_visits", "prior_no_shows", "sms_reminders"]
# Loaded as float64. days_since_last_visit is empty for a first visit.
FLOAT_FEATURES = ["distance_km", "days_since_last_visit"]
CATEGORICAL_FEATURES = ["clinic", "booking_channel", "weekday", "slot", "insurance"]

NUMERIC_FEATURES = INTEGER_FEATURES + FLOAT_FEATURES
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "no_show"

RANDOM_STATE = 42
TEST_SIZE = 0.25


def load_frame():
    """Every appointment. Returns (X, y): X holds FEATURES in that order."""
    frame = pd.read_csv(DATA_PATH)
    return frame[FEATURES], frame[TARGET]


def load_split():
    """The fixed split. Returns X_train, X_test, y_train, y_test."""
    X, y = load_frame()
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
