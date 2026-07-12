import os
import joblib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data_preprocessing import load_and_preprocess_data
from models_config import get_models_and_params

def save_and_evaluate_best_model():
    x_train, x_test, y_train, y_test = load_and_preprocess_data()
    models, param_grids = get_models_and_params()

    model = models['LogisticRegression']
    param_grid = param_grids['LogisticRegression']

    #ponovna pretraga parametara fokusirana na recall
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='recall', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    best_model.fit(x_train, y_train)

    print(f"Najbolji pronadjeni parametri: {grid_search.best_params_}")

    y_pred = best_model.predict(x_test)

    #cuvanje najboljeg modela
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'best_model.pkl')
    joblib.dump(best_model, model_path)

    matrica_konfuzije = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrica_konfuzije, display_labels=best_model.classes_)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, im_kw={"interpolation": "nearest"})

    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    save_and_evaluate_best_model()