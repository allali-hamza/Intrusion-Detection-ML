import pandas as pd
from pathlib import Path
from config import TRAIN_PATH, TEST_PATH, COLUMNS


def load_raw_data():
    train_df = pd.read_csv(TRAIN_PATH, header=None, names=COLUMNS)
    test_df = pd.read_csv(TEST_PATH, header=None, names=COLUMNS)
    return train_df, test_df


def save_processed_data(train_df, test_df):
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)


if __name__ == "__main__":
    train_df, test_df = load_raw_data()
    save_processed_data(train_df, test_df)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)