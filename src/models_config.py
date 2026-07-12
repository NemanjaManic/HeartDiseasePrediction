from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def get_models_and_params():
    models = {
        'LogisticRegression': LogisticRegression(),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'SVC': SVC()
    }

    param_grids = {
        'LogisticRegression': {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear'],
            'class_weight': [None, 'balanced'],
            'max_iter': [1000, 5000]
        },
        'KNN': {
            'n_neighbors': [1, 2, 3, 4, 5],
            'metric': ['euclidean', 'manhattan']
        },
        'DecisionTree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, 15],
            'min_samples_split': [0.01, 0.05],
            'class_weight': [None, 'balanced']
        },
        'SVC': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear', 'poly','sigmoid']
        }
    }

    return models, param_grids