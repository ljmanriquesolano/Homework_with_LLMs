import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def segmentar_clientes_kmeans(datos, num_clusters):
    modelo = KMeans(n_clusters=num_clusters, random_state=42)

    etiquetas = modelo.fit_predict(datos)
    centroides = modelo.cluster_centers_

    return etiquetas, centroides
