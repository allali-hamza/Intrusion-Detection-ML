from src.train import train_random_forest
from src.evaluate import evaluate_model
import pandas as pd

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()


# print(X_train.shape)
# print(y_train.shape)

# print(X_test.shape)
# print(y_test.shape)

model = train_random_forest(X_train, y_train)
evaluate_model(model, X_test, y_test)