import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def generar_caso_de_uso_entrenar_y_predecir():
    """    Genera un dataset aleatorio (input) y calcula el accuracy     correspondiente (output) mediante un modelo de Regresión Logística.    """
    # --- 1. Generación de datos aleatorios ---
    n_filas = np.random.randint(50, 201)

    data = {
        'Edad': np.random.randint(18, 81, n_filas),
        'Mensualidad': np.around(np.random.uniform(20, 200, n_filas), 2),
        'Meses_Contrato': np.random.randint(1, 61, n_filas),
        'Plan': np.random.choice(['Básico', 'Premium'], n_filas)
    }

    # Generar Churn con una ligera dependencia lógica para que no sea puro azar
    # Si la mensualidad es alta y el contrato corto, hay más probabilidad de Churn
    prob_base = (data['Mensualidad'] / 200) * 0.5 + (1 - data['Meses_Contrato'] / 60) * 0.3
    data['Churn'] = [1 if p > np.random.rand() else 0 for p in prob_base]

    df = pd.DataFrame(data)

    # Introducir valores nulos aleatorios (en aprox. 10% de las filas)
    for col in df.columns:
        mask = np.random.random(n_filas) < 0.1
        df.loc[mask, col] = np.nan

    # Guardamos una copia para el input antes de procesar
    input_dict = {"df": df.copy()}

    # --- 2. Lógica de procesamiento (Cálculo del Output) ---

    # A. Limpieza: Eliminar nulos
    df_clean = df.dropna()

    # B. Transformación: One-hot encoding para 'Plan'
    # Usamos drop_first=True para evitar la trampa de la multicolinealidad
    df_proc = pd.get_dummies(df_clean, columns=['Plan'], drop_first=True)

    # C. Separación de X e y
    X = df_proc.drop('Churn', axis=1)
    y = df_proc['Churn']

    # D. División de datos (random_state fijo para reproducibilidad del test interno)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # E. Entrenamiento
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)

    # F. Evaluación
    predicciones = modelo.predict(X_test)
    accuracy = float(accuracy_score(y_test, predicciones))

    return input_dict, accuracy

# --- Bloque de prueba ---
if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_entrenar_y_predecir()

    print("--- INPUT (Primeras 5 filas del DataFrame) ---")
    print(input_data["df"].head())
    print(f"\nTotal de filas generadas: {len(input_data['df'])}")
    print(f"Valores nulos totales: {input_data['df'].isnull().sum().sum()}")

    print("\n" + "-"*45)
    print(f"--- OUTPUT (Accuracy del modelo) ---")
    print(f"Accuracy: {output_data:.4f}")
