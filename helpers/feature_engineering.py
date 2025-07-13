from sklearn import clone
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

from helpers.eval_model import evaluate_model


def test_feature_engineering_strategies(strategies, models, X_data, y_data):
    all_engineering_results = {}

    for model_name, model in models.items():
        if model is None:
            print(f"Modell {model_name} nicht gefunden, wird übersprungen")
            continue

        # Original-Modell als Baseline evaluieren (mit den ursprünglichen Features)
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X_data, y_data, test_size=0.25, random_state=42, stratify=y_data
        )
        scaler_orig = StandardScaler()
        X_train_orig_scaled = scaler_orig.fit_transform(X_train_orig)
        X_test_orig_scaled = scaler_orig.transform(X_test_orig)

        # Modell mit Originaldaten trainieren und evaluieren
        model_orig = clone(model)
        model_orig.fit(X_train_orig_scaled, y_train_orig)
        results_orig = evaluate_model(
            model_orig,
            X_train_orig_scaled,
            X_test_orig_scaled,
            y_train_orig,
            y_test_orig,
            X_data,
            y_data,
        )

        # Ergebnisse speichern
        if model_name not in all_engineering_results:
            all_engineering_results[model_name] = {}
        all_engineering_results[model_name]["Original"] = results_orig["f1"]

        # Feature Engineering Strategien anwenden
        for strategy_name, strategy in strategies.items():
            try:
                # Transformierte Features erstellen
                X_transformed = strategy["transform"](X_data)

                # Daten teilen
                X_train_trans, X_test_trans, y_train_trans, y_test_trans = (
                    train_test_split(
                        X_transformed,
                        y_data,
                        test_size=0.25,
                        random_state=42,
                        stratify=y_data,
                    )
                )

                # Daten skalieren
                scaler_trans = StandardScaler()
                X_train_trans_scaled = scaler_trans.fit_transform(X_train_trans)
                X_test_trans_scaled = scaler_trans.transform(X_test_trans)

                # Modell klonen und mit transformierten Features trainieren
                model_trans = clone(model)
                model_trans.fit(X_train_trans_scaled, y_train_trans)

                # Modell evaluieren
                results_trans = evaluate_model(
                    model_trans,
                    X_train_trans_scaled,
                    X_test_trans_scaled,
                    y_train_trans,
                    y_test_trans,
                    X_transformed,
                    y_data,
                )

                # Ergebnisse speichern
                all_engineering_results[model_name][strategy_name] = results_trans["f1"]

            except Exception as e:
                print(f"Fehler bei {strategy_name} für {model_name}: {e}")
                all_engineering_results[model_name][strategy_name] = None

    return all_engineering_results
