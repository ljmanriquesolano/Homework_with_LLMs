import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def generar_caso_de_uso_entrenar_modelo_viviendas():
    """    Genera un caso de uso aleatorio para el problema de predicción de viviendas,    incluyendo un DataFrame sintético y los resultados esperados del entrenamiento.    """

    # 1. Parámetros aleatorios para la estructura
    n_rows = np.random.randint(50, 201)
    n_features = np.random.randint(3, 9)
    target_name = "precio"

    # 2. Generación de features aleatorias (X) usando NumPy
    # Creamos nombres genéricos para las columnas
    cols = [f"feature_{i}" for i in range(n_features)]
    X_values = np.random.rand(n_rows, n_features) * 100
    df = pd.DataFrame(X_values, columns=cols)

    # 3. Generación de un Target coherente (Combinación lineal + Ruido)
    # Creamos coeficientes aleatorios para cada feature
    weights = np.random.uniform(100, 1000, size=n_features)
    bias = 50000
    # y = Xw + b + ruido
    ruido = np.random.normal(0, 5000, size=n_rows)
    df[target_name] = X_values.dot(weights) + bias + ruido

    # --- Lógica de procesamiento para generar el OUTPUT ---

    # Separar X e y
    X = df.drop(columns=[target_name])
    y = df[target_name]

    # Split (80/20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Escalado (Fit solo en train para evitar data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenamiento del modelo Ridge
    modelo = Ridge()
    modelo.fit(X_train_scaled, y_train)

    # Predicción y cálculo de métrica
    y_pred = modelo.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)

    # --- Construcción del retorno ---

    input_data = {
        "df": df,
        "target": target_name
    }

    output_data = (mse, modelo)

    return input_data, output_data

# Bloque de prueba
if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_entrenar_modelo_viviendas()

    df_generado = input_data["df"]
    mse_resultado, modelo_resultado = output_data

    print("--- Caso de Uso Generado ---")
    print(f"Filas del DataFrame: {len(df_generado)}")
    print(f"Columnas detectadas: {list(df_generado.columns)}")
    print(f"MSE calculado en test: {mse_resultado:.4f}")
    print(f"Tipo de modelo: {type(modelo_resultado)}")

    # Verificación rápida de coeficientes
    print(f"Coeficientes del modelo: {modelo_resultado.coef_}")
