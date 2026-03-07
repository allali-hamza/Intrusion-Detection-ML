from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, Y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, Y_train)
    return model