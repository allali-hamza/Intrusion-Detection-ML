from src.train import train_random_forest, train_decision_tree
from src.evaluate import evaluate_model
import pandas as pd

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

models = {
    "Decision Tree":       train_decision_tree(X_train, y_train),
    "Random Forest":       train_random_forest(X_train, y_train),
}

for name, model in models.items():
    evaluate_model(model, X_test, y_test, model_name=name, threshold=0.05)