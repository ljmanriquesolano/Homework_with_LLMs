import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def generar_caso_de_uso_entrenar_modelo_prestamos():
    """    Genera un caso de uso aleatorio para el reto de predicción de préstamos.    Devuelve un diccionario de entrada (df) y el resultado esperado (modelo, accuracy, df_procesado).    """
    # --- 1. Generación de Datos Aleatorios ---
    n_rows = np.random.randint(80, 301)

    # Creamos características base
    edad = np.random.randint(18, 71, n_rows)
    ingresos = np.random.uniform(1500, 10000, n_rows)
    monto = np.random.uniform(500, 5000, n_rows)
    historial = np.random.choice(['Bueno', 'Regular', 'Malo'], n_rows)
    empleo = np.random.choice(['Fijo', 'Temporal', 'Independiente'], n_rows)

    # Lógica para el target (Estado_Prestamo) para que no sea puro ruido
    # El éxito depende de ingresos altos, bajo monto y buen historial
    score = (ingresos / 10000) - (monto / 5000) + (historial == 'Bueno').astype(int)
    prob = 1 / (1 + np.exp(-score)) # Sigmoide para probabilidad
    estado_prestamo = (prob > 0.5).astype(int)

    # Asegurar que existan ambas clases
    if len(np.unique(estado_prestamo)) < 2:
        estado_prestamo[0], estado_prestamo[1] = 0, 1

    df = pd.DataFrame({
        'Edad': edad,
        'Ingresos_Mensuales': ingresos,
        'Monto_Prestamo': monto,
        'Historial_Crediticio': historial,
        'Empleo': empleo,
        'Estado_Prestamo': estado_prestamo
    })

    # Introducción de valores nulos aleatorios (aprox 10% de los datos)
    for col in df.columns[:-1]: # No tocamos el target
        mask = np.random.random(n_rows) < 0.1
        df.loc[mask, col] = np.nan

    # Copia para el input antes de procesar
    input_dict = {"df": df.copy()}

    # --- 2. Lógica de Procesamiento (Réplica para Output) ---
    df_proc = df.copy()

    # A. Imputación
    num_cols = ['Edad', 'Ingresos_Mensuales', 'Monto_Prestamo']
    cat_cols = ['Historial_Crediticio', 'Empleo']

    for col in num_cols:
        df_proc[col] = df_proc[col].fillna(df_proc[col].median())

    for col in cat_cols:
        # La moda devuelve una Serie, tomamos el primer elemento [0]
        df_proc[col] = df_proc[col].fillna(df_proc[col].mode()[0])

    # B. Feature Engineering
    # Evitamos división por cero asegurando un mínimo en ingresos
    df_proc['Cuota_Ingreso'] = df_proc['Monto_Prestamo'] / df_proc['Ingresos_Mensuales'].replace(0, 1)

    # Actualizamos lista de numéricas para incluir la nueva variable
    num_cols_final = num_cols + ['Cuota_Ingreso']

    # C. Separar X e y
    X = df_proc.drop(columns=['Estado_Prestamo'])
    y = df_proc['Estado_Prestamo']

    # D. Preprocesamiento (Encoding y Scaling)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols_final),
            ('cat', OneHotEncoder(sparse_output=False), cat_cols)
        ]
    )

    # Obtenemos el DataFrame procesado final para el retorno
    X_transformed = preprocessor.fit_transform(X)

    # Generar nombres de columnas para el DF procesado
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    final_cols = num_cols_final + list(cat_names)
    dataframe_procesado = pd.DataFrame(X_transformed, columns=final_cols)
    # Añadimos el target al final para completarlo
    dataframe_procesado['Estado_Prestamo'] = y.values

    # E. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42
    )

    # F. Entrenar modelo
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    # G. Evaluar
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # H. Construir Output
    output_tuple = (modelo, accuracy, dataframe_procesado)

    return input_dict, output_tuple

# --- Bloque de Prueba ---
if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_entrenar_modelo_prestamos()

    print(f"--- Caso de Uso Generado ---")
    print(f"Tamaño del dataset original: {input_data['df'].shape}")
    print(f"Accuracy del modelo: {output_data[1]:.4f}")
    print(f"Columnas en el DF procesado: {list(output_data[2].columns)}")
    print("\nPrimeras filas del DataFrame procesado:")
    print(output_data[2].head())
