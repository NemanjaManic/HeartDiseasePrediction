import os
from datetime import datetime
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from data_preprocessing import load_and_preprocess_data
from models_config import get_models_and_params


def train_models():

    x_train, x_test, y_train, y_test = load_and_preprocess_data()
    models, param_grids = get_models_and_params()

    os.makedirs('results', exist_ok=True)
    putanja_izvestaja = os.path.join('results', 'train_results.txt')

    with open(putanja_izvestaja, 'a', encoding='utf-8') as f:

        vreme_pokretanja = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"\n==================================================\n")
        f.write(f"Training: {vreme_pokretanja}\n")
        f.write(f"==================================================\n")

        for name, model in models.items():
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='recall', n_jobs=-1)
            grid_search.fit(x_train, y_train)

            best_estimator = grid_search.best_estimator_
            best_estimator.fit(x_train, y_train)

            y_pred = best_estimator.predict(x_test)

            precision = metrics.precision_score(y_test, y_pred)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            sensitivity = metrics.recall_score(y_test, y_pred)
            f_score = metrics.f1_score(y_test, y_pred)

            izvestaj_modela = (
                f"--- Model: {name} ---\n"
                f"Najbolji parametri: {grid_search.best_params_}\n"
                f"precision: {precision:.2f}\n"
                f"accuracy: {accuracy:.2f}\n"
                f"sensitivity/recall: {sensitivity:.2f}\n"
                f"F score: {f_score:.2f}\n\n"
            )

            print(izvestaj_modela, end="")
            f.write(izvestaj_modela)

if __name__ == "__main__":
    train_models()