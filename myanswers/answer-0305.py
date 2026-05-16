import pandas as pd
import numpy as np

def analizar_retencion_por_dispositivo(df):
    # Eliminar rebotes
    df_clean = df[df["tiempo_minutos"] >= 1].copy()

    # Crear columna booleana
    df_clean["es_retenido"] = df_clean["tiempo_minutos"] > 10

    # Agrupar y calcular métricas
    resultado = (
        df_clean
        .groupby("dispositivo")
        .agg(
            sesiones_validas=("tiempo_minutos", "count"),
            tasa_retencion=("es_retenido", lambda x: round(x.mean(), 4)),
            tiempo_promedio=("tiempo_minutos", lambda x: round(x.mean(), 2)),
        )
        .sort_values("tasa_retencion", ascending=False)
        .reset_index()
    )

    return resultado
