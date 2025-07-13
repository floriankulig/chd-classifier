from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# Funktion zur Evaluierung der Modelle
def evaluate_model(model, X_train, X_test, y_train, y_test, X_data, y_data):
    # Modell trainieren
    model.fit(X_train, y_train)

    # Vorhersagen treffen
    y_pred = model.predict(X_test)

    # Metriken berechnen
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Kreuzvalidierung
    cv_scores = cross_val_score(model, X_data, y_data, cv=5, scoring="accuracy")

    return {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "predictions": y_pred,
    }
