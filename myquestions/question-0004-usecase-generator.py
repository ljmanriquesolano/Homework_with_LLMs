import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

def generar_caso_de_uso_escalar_dataframe():

    n_rows = random.randint(5, 15)
    n_cols = random.randint(2, 5)

    data = np.random.randn(n_rows, n_cols)

    columnas = [f"col_{i}" for i in range(n_cols)]

    df = pd.DataFrame(data, columns=columnas)

    input_data = {
        "df": df.copy()
    }

    scaler = StandardScaler()
    output_data = scaler.fit_transform(df)

    return input_data, output_data
