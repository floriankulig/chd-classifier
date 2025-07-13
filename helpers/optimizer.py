# Funktion für Grid Search und Evaluierung
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore")


def optimize_hyperparameters(model_name, model, param_grid, X_train, y_train, cv=5):
    print(f"Optimiere Hyperparameter für {model_name}...")

    # Löse Solver/Penalty Konflikt bei Logistic Regression
    if model_name == "Logistic Regression":
        valid_combinations = []
        for penalty in param_grid["penalty"]:
            for solver in param_grid["solver"]:
                # liblinear unterstützt keine 'none' penalty
                if penalty is None and solver == "liblinear":
                    continue
                # liblinear unterstützt kein 'elasticnet'
                if penalty == "elasticnet" and solver != "saga":
                    continue
                # l1 funktioniert nur mit liblinear oder saga
                if penalty == "l1" and solver not in ["liblinear", "saga"]:
                    continue

                valid_combinations.append({"penalty": penalty, "solver": solver})

        # Erstelle ein neues param_grid mit den gültigen Kombinationen
        new_param_grid = {"C": param_grid["C"], "max_iter": param_grid["max_iter"]}
        for param in valid_combinations:
            for k, v in param.items():
                if k not in new_param_grid:
                    new_param_grid[k] = []
                if v not in new_param_grid[k]:
                    new_param_grid[k].append(v)

        param_grid = new_param_grid

    grid_search = GridSearchCV(
        model, param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"Beste Parameter für {model_name}: {grid_search.best_params_}")
    print(f"Bester F1-Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_
