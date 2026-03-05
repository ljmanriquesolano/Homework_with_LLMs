import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def generar_caso_de_uso_entrenar_clasificador_iris():

    iris = load_iris()

    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    y = iris.target

    input_data = {}

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo = LogisticRegression(max_iter=200)
    modelo.fit(X_train_scaled, y_train)

    predicciones = modelo.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predicciones)

    output_data = accuracy

    return input_data, output_data
