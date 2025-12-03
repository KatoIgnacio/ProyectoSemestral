import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def cargar_datos(ruta_csv: str) -> pd.DataFrame:

    df = pd.read_csv(ruta_csv)
    print(f"[INFO] Datos cargados desde {ruta_csv} con shape {df.shape}")
    return df


def preprocesar_datos(df: pd.DataFrame, columnas_numericas: list):
    # Verificacion de columnas
    columnas_faltantes = [c for c in columnas_numericas if c not in df.columns]
    if columnas_faltantes:
        print(f"[WARN] Las siguientes columnas no se encontraron en el DataFrame: {columnas_faltantes}")

    df_num = df[columnas_numericas]
    filas_iniciales = df_num.shape[0]
    X = df_num.dropna()
    filas_finales = X.shape[0]

    print(f"[INFO] Filas antes de dropna: {filas_iniciales}, despues: {filas_finales}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[INFO] Estandarización aplicada a las columnas numéricas seleccionadas")

    return X, X_scaled, scaler


def aplicar_pca(X_scaled, n_componentes: int = 2):

    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X_scaled)

    varianza_explicada = pca.explained_variance_ratio_
    varianza_acumulada = varianza_explicada.cumsum()

    print("[INFO] PCA aplicado")
    print(f"[INFO] Varianza explicada por componente: {varianza_explicada}")
    print(f"[INFO] Varianza explicada acumulada: {varianza_acumulada}")

    return X_pca, pca


def entrenar_kmeans(X_features, k: int = 3):

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    etiquetas = kmeans.fit_predict(X_features)

    print("[INFO] K-Means entrenado")
    print(f"[INFO] Numero de clusters: {k}")
    print(f"[INFO] Inercia del modelo (distancia interna total): {kmeans.inertia_}")
    print(f"[INFO] Distribucion de elementos por cluster: {pd.Series(etiquetas).value_counts().to_dict()}")

    return kmeans, etiquetas


def ejecutar_pipeline(
    ruta_csv: str,
    columnas_numericas: list,
    n_componentes_pca: int = 2,
    k_clusters: int = 3,
):

    print("[INFO] Iniciando pipeline completo PCA + K-Means para datos estudiantiles")

    df = cargar_datos(ruta_csv)
    X, X_scaled, scaler = preprocesar_datos(df, columnas_numericas)
    X_pca, pca = aplicar_pca(X_scaled, n_componentes=n_componentes_pca)
    kmeans, etiquetas = entrenar_kmeans(X_pca, k=k_clusters)

    print("[INFO] Pipeline completado correctamente")

    resultados = {
        "df": df,
        "X": X,
        "X_scaled": X_scaled,
        "X_pca": X_pca,
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "etiquetas": etiquetas,
    }
    return resultados


if __name__ == "__main__":
    # Ejemplo para pruebas rapidas
    ruta = "datos_estudiantes.csv"
    columnas = [
        "tiempo_moodle",
        "accesos",
        "participacion_foros",
        "porcentaje_tareas",
        "nota_final",
    ]

    resultados = ejecutar_pipeline(
        ruta_csv=ruta,
        columnas_numericas=columnas,
        n_componentes_pca=2,
        k_clusters=3,
    )

    X_pca = resultados["X_pca"]
    etiquetas = resultados["etiquetas"]

    print("Shape PCA:", X_pca.shape)
    print("Clusters generados:", set(etiquetas))