import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def limpiar_outliers(df, target_col):
    # Separar variables
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Imputar NaNs con la media
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )

    # Calcular z-scores
    z_scores = ((X_imputed - X_imputed.mean()) / X_imputed.std()).abs()

    # Filtrar filas sin outliers
    mask = (z_scores < 3).all(axis=1)

    # Convertir a numpy arrays
    X_clean = X_imputed[mask].to_numpy()
    y_clean = y[mask].to_numpy()

    return X_clean, y_clean
