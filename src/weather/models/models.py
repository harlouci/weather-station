from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

models = {
    "RandomForest": {
        "model": RandomForestClassifier(),
        "param_grid": {"model__n_estimators": [5, 10], "model__max_depth": [None, 5, 10]},
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(),
        "param_grid": {"model__n_estimators": [50, 100], "model__learning_rate": [0.1, 0.01]},
    },
}
