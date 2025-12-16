import streamlit as st
import pandas as pd

from modelo import (
    preprocesar_datos,
    aplicar_pca,
    entrenar_kmeans,
)


def main():
    st.set_page_config(page_title="CatStudy", layout="wide")

    st.sidebar.image("logo.png")
    st.sidebar.title("CatStudy")

    archivo = st.sidebar.file_uploader("Archivo CSV", type=["csv"])

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:

        st.title("˗ˏˋ ♡ ˎˊ˗ CatStudy ˗ˏˋ ♡ ˎˊ˗")
        st.subheader("Agrupamiento de perfiles estudiantiles segun habitos de aprendizaje")

        st.write(
            "Esta aplicacion permite explorar perfiles estudiantiles usando PCA y K-Means."
        )
        st.write(
            "Sube un archivo CSV con datos numericos de estudiantes "
        )

        if archivo is None:
            st.info("🐀 ・゜゜・. 🐈 ..... Esperando que subas un archivo CSV en la barra lateral.")
            return

        # Lectura del archivo CSV
        try:
            df = pd.read_csv(archivo)
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
            return

        st.subheader("Vista previa del dataset")
        st.dataframe(df.head())

        columnas_numericas = df.select_dtypes(include="number").columns.tolist()

        if len(columnas_numericas) == 0:
            st.error("El archivo no tiene columnas numericas. Revisa el CSV.")
            return

        st.markdown("Columnas numericas detectadas:")
        st.write(columnas_numericas)

        columnas_seleccionadas = st.sidebar.multiselect(
            "Columnas numericas para el analisis",
            columnas_numericas,
            default=columnas_numericas,
        )

        if len(columnas_seleccionadas) == 0:
            st.warning("Selecciona al menos una columna numerica en la barra lateral.")
            return

        k = st.sidebar.slider(
            "Numero de clusters (k)",
            min_value=2,
            max_value=8,
            value=3,
            step=1,
        )

        st.sidebar.markdown("---")
        st.sidebar.caption(
            "El modelo aplica PCA (2 componentes) para visualizacion "
            "y luego K-Means para agrupar estudiantes."
        )

        try:
            X, X_scaled, scaler = preprocesar_datos(df, columnas_seleccionadas)
            X_pca, pca = aplicar_pca(X_scaled, n_componentes=2)
            modelo, etiquetas = entrenar_kmeans(X_pca, k=k)
        except Exception as e:
            st.error(f"Error al ejecutar el modelo: {e}")
            return

        df_result = df.loc[X.index].copy()
        df_result["PCA1"] = X_pca[:, 0]
        df_result["PCA2"] = X_pca[:, 1]
        # cluster como numero, no como string
        df_result["Cluster"] = etiquetas

        st.subheader("Resumen del procesamiento")
        st.caption(
            f"Filas del dataset original: {df.shape[0]} | "
            f"Filas usadas en el modelo (sin NaN en columnas seleccionadas): {X.shape[0]}"
        )
        st.caption(
            f"Inercia del modelo K-Means (distancia interna total): {modelo.inertia_:.2f}"
        )

        st.subheader("Grafico de dispersion (PCA1 y PCA2)")
        st.markdown(
            "Los resultados se visualizan en un plano formado por las dos primeras componentes principales "
            "(PCA1 y PCA2). Cada punto representa un estudiante y el color indica el cluster asignado."
        )
        st.scatter_chart(df_result, x="PCA1", y="PCA2", color="Cluster")

        # resumen de variables numericas por cluster
        resumen = df_result.groupby("Cluster")[columnas_seleccionadas].mean()

        # etiquetas interpretables por cluster basadas en promedios
        promedio_total = resumen.mean().mean()
        labels_clusters = {}
        descripciones_clusters = {}

        for cluster_id in resumen.index:
            fila_cluster = resumen.loc[cluster_id]
            promedio_cluster = fila_cluster.mean()

            if promedio_cluster >= promedio_total * 1.10:
                etiqueta = "Estudiantes de alto compromiso"
                descripcion_simple = (
                    "son estudiantes que usan mucho la plataforma, "
                    "participan con frecuencia y suelen obtener buenas notas."
                )
            elif promedio_cluster <= promedio_total * 0.90:
                etiqueta = "Estudiantes en riesgo academico"
                descripcion_simple = (
                    "son estudiantes con menor uso de la plataforma, "
                    "menos tareas entregadas o notas mas bajas. "
                    "Requieren acompanamiento y apoyo adicional."
                )
            else:
                etiqueta = "Estudiantes intermedios"
                descripcion_simple = (
                    "son estudiantes con un uso y rendimiento medios: "
                    "cumplen la mayoria de las actividades y se mantienen "
                    "cerca del promedio del curso."
                )

            labels_clusters[cluster_id] = etiqueta
            descripciones_clusters[cluster_id] = descripcion_simple

        # --- Tabla resumen de clusters ---
        st.subheader("Tabla resumen de clusters")
        st.caption(
            "Resumen general de cada cluster, indicando su etiqueta interpretativa "
            "y la cantidad de estudiantes asignados."
        )

        # conteo de estudiantes por cluster
        conteo_clusters = df_result["Cluster"].value_counts().sort_index()

        # DataFrame con Cluster, Etiqueta y Cantidad de estudiantes
        tabla_clusters = pd.DataFrame(
            {
                "Cluster": conteo_clusters.index,
                "Etiqueta": [
                    labels_clusters.get(i, f"Cluster {i}") for i in conteo_clusters.index
                ],
                "Cantidad_estudiantes": conteo_clusters.values,
            }
        )

        st.dataframe(tabla_clusters)

        st.subheader("Tabla de caracteristicas promedio por cluster")
        st.caption(
            "La tabla muestra, para cada cluster, el promedio de las variables seleccionadas, "
            "lo que permite describir el perfil tipico de cada grupo."
        )
        st.dataframe(resumen.style.format("{:.2f}"))

        # --- Nueva seccion: inferencia para un estudiante nuevo ---
        st.subheader("Ingresar un nuevo estudiante para inferencia")

        # Formulario para capturar los datos de un estudiante hipotetico
        with st.form("nuevo_estudiante"):
            st.write("Complete los valores para un estudiante hipotetico:")

            # Diccionario donde se guardan los valores ingresados
            valores_nuevo = {}
            for col in columnas_seleccionadas:
                # usamos como valor por defecto el promedio de la columna
                valor_defecto = float(df_result[col].mean())
                valores_nuevo[col] = st.number_input(
                    f"{col}",
                    value=valor_defecto,
                    key=f"nuevo_{col}",
                )

            submit_nuevo = st.form_submit_button("Calcular cluster")

        if submit_nuevo:
            # Crear DataFrame con una sola fila
            df_nuevo = pd.DataFrame([valores_nuevo])

            # Usar el mismo scaler y pca entrenados
            X_nuevo = df_nuevo[columnas_seleccionadas].values
            X_nuevo_scaled = scaler.transform(X_nuevo)
            X_nuevo_pca = pca.transform(X_nuevo_scaled)
            cluster_nuevo = int(modelo.predict(X_nuevo_pca)[0])

            # Mensaje principal pensado para todo publico
            if cluster_nuevo in labels_clusters:
                etiqueta = labels_clusters[cluster_nuevo]
                st.success(
                    f"El nuevo estudiante se parece al grupo: {etiqueta} "
                    f"(cluster {cluster_nuevo})."
                )
            else:
                st.success(
                    f"El nuevo estudiante fue asignado al cluster {cluster_nuevo}."
                )

            # ---- Interpretacion automatica del cluster ----
            if cluster_nuevo in labels_clusters:
                fila = resumen.loc[cluster_nuevo]

                st.markdown(
                    f"### Cluster {cluster_nuevo} — {labels_clusters[cluster_nuevo]}"
                )

                # Explicacion en lenguaje sencillo
                if cluster_nuevo in descripciones_clusters:
                    st.info(
                        "En palabras simples, esto significa que "
                        f"{descripciones_clusters[cluster_nuevo]}"
                    )

                st.write("Este cluster se caracteriza por los siguientes promedios:")

                detalles = []
                for col in columnas_seleccionadas:
                    detalles.append(f"- **{col}:** {fila[col]:.2f}")

                st.markdown("\n".join(detalles))
            else:
                st.warning(
                    "No se pudo encontrar informacion del cluster en el resumen."
                )

            st.write("Coordenadas PCA del nuevo estudiante:")
            st.write(
                {
                    "PCA1": float(X_nuevo_pca[0, 0]),
                    "PCA2": float(X_nuevo_pca[0, 1]),
                }
            )

if __name__ == "__main__":
    main()