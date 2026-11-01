import pandas as pd
from pathlib import Path

from config import TRAIN_PATH, TEST_PATH, COLUMNS, BASE_DIR



def load_raw_data():
    train_df = pd.read_csv(TRAIN_PATH, header=None, names=COLUMNS)
    test_df = pd.read_csv(TEST_PATH, header=None, names=COLUMNS)
    return train_df, test_df


def save_processed_data(train_df, test_df):
    output_dir = BASE_DIR / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


if __name__ == "__main__":
    train_df, test_df = load_raw_data()
    save_processed_data(train_df, test_df)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
