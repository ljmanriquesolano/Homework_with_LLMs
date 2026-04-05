import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def generar_caso_de_uso_entrenar_clasificador():
    """
    Genera un caso de uso aleatorio para la función entrenar_clasificador.
    """

    # 1. Cargar dataset base
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 2. Convertir a DataFrame
    df = pd.DataFrame(X, columns=iris.feature_names)
    df["target"] = y

    # 3. Introducir aleatoriedad REAL
    # - Permutar filas
    df = df.sample(frac=1).reset_index(drop=True)

    # - Añadir ruido leve a features
    ruido = np.random.normal(0, 0.1, df.iloc[:, :-1].shape)
    df.iloc[:, :-1] = df.iloc[:, :-1] + ruido

    # - Eliminar aleatoriamente algunas filas (simula datasets distintos)
    n_drop = np.random.randint(0, 10)
    if n_drop > 0:
        df = df.iloc[:-n_drop]

    # 4. Construir INPUT
    input_data = {
        "df": df.copy(),
        "target_col": "target"
    }

    # -------------------------------------------------
    # 5. Calcular OUTPUT esperado (ground truth)
    # -------------------------------------------------

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train_scaled, y_train)

    y_pred = modelo.predict(X_test_scaled)
    accuracy = float(accuracy_score(y_test, y_pred))

    output_data = accuracy

    return input_data, output_data


# --- Bloque de prueba ---
if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_entrenar_clasificador()

    print("--- INPUT ---")
    print(input_data["df"].head())
    print(f"Filas: {len(input_data['df'])}")

    print("\n--- OUTPUT ---")
    print(f"Accuracy: {output_data:.4f}")
