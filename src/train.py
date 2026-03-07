from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model




def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model