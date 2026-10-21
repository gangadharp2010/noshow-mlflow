# No-show model

Predicts whether a patient will miss a booked clinic appointment, so the
booking team can send an extra reminder or double-book the slot.

## Layout

- `data/appointments.csv` is 6000 past appointments, one row each.
- `noshow/data.py` loads the table and makes the fixed train and holdout split.
- `noshow/pipeline.py` builds the model for one pair of settings.
- `noshow/evaluate.py` scores a model on the holdout rows or by cross validation.
- `noshow/train.py` fits one model and returns it with its holdout scores.
- `noshow/search.py` searches the two settings with Optuna.
- `requirements.txt` lists what the project needs.

## Running it

From this directory:

    python3 -m noshow.train
    python3 -m noshow.search --trials 8

Neither command records anything. Each prints its scores and forgets them.
