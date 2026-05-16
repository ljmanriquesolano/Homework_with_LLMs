import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preparar_datos_clasificacion(df, target_col):
    # Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # Imputar valores faltantes
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y
