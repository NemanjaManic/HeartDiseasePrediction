import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data():
    df = pd.read_csv('../data/heart.csv')

    df.drop(df[df["RestingBP"] == 0].index, inplace=True)

    kategoricke_kolone = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    df = pd.get_dummies(df, columns=kategoricke_kolone, drop_first=True)

    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].mean())

    x = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

    scaler = StandardScaler()
    x_train_skalirano = scaler.fit_transform(x_train)
    x_test_skalirano = scaler.transform(x_test)

    return x_train_skalirano, x_test_skalirano, y_train, y_test