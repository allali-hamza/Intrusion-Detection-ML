import pandas as pd
from pathlib import Path

from config import TARGET_COL
from data_load import load_raw_data

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def preprocess_data(train_df, test_df):

    train_df[TARGET_COL] = train_df[TARGET_COL].apply(lambda x: 0 if x == "normal" else 1)
    test_df[TARGET_COL] = test_df[TARGET_COL].apply(lambda x: 0 if x == "normal" else 1)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # One Hot Encoding
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLS)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_COLS)

    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    return X_train, X_test, y_train, y_test


def save_processed(X_train, X_test, y_train, y_test, out_dir="data/processed"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)


if __name__ == "__main__":

    train_df, test_df = load_raw_data()

    X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)

    save_processed(X_train, X_test, y_train, y_test)

    print("Done.")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("y_train distribution:\n", y_train.value_counts())