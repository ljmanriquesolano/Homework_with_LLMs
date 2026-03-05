import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_contar_valores_unicos():

    n_rows = random.randint(5, 20)

    valores = np.random.randint(0, 10, size=n_rows)

    df = pd.DataFrame({"numeros": valores})

    input_data = {
        "df": df.copy(),
        "columna": "numeros"
    }

    output_data = df["numeros"].nunique()

    return input_data, output_data
