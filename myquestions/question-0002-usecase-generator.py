import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_calcular_media_columna():

    n_rows = random.randint(5, 15)

    data = np.random.randn(n_rows)
    mask = np.random.choice([True, False], size=n_rows, p=[0.2, 0.8])
    data[mask] = np.nan

    df = pd.DataFrame({"valores": data})

    input_data = {
        "df": df.copy(),
        "columna": "valores"
    }

    output_data = df["valores"].mean()

    return input_data, output_data
