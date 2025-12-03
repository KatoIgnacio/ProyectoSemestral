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

        st.subheader("Grafico PCA por cluster")
        st.markdown(
            "Cada punto representa un estudiante proyectado en dos componentes principales (PCA1 y PCA2). "
            "Los colores indican el cluster asignado por K-Means."
        )
        st.scatter_chart(df_result, x="PCA1", y="PCA2", color="Cluster")

        st.subheader("Cantidad de estudiantes por cluster")
        conteo = df_result["Cluster"].value_counts().sort_index()
        st.bar_chart(conteo)

        st.subheader("Resumen por cluster (promedios)")
        resumen = df_result.groupby("Cluster")[columnas_seleccionadas].mean()
        st.dataframe(resumen.style.format("{:.2f}"))

        # etiquetas interpretables por cluster basadas en promedios
        promedio_total = resumen.mean().mean()
        labels_clusters = {}
        for cluster_id in resumen.index:
            fila_cluster = resumen.loc[cluster_id]
            promedio_cluster = fila_cluster.mean()

            if promedio_cluster >= promedio_total * 1.10:
                etiqueta = "Estudiantes de alto compromiso"
            elif promedio_cluster <= promedio_total * 0.90:
                etiqueta = "Estudiantes en riesgo academico"
            else:
                etiqueta = "Estudiantes intermedios"

            labels_clusters[int(cluster_id)] = etiqueta

        # --- Nueva seccion: inferencia para un estudiante nuevo ---
        st.subheader("Ingresar un nuevo estudiante para inferencia")

        st.markdown(
            "Completa los valores de un estudiante hipotetico para ver "
            "a que cluster seria asignado segun el modelo entrenado."
        )

        with st.form("form_nuevo_estudiante"):
            valores_nuevo = {}

            st.write("Ingresa las variables numericas seleccionadas:")

            for col in columnas_seleccionadas:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())

                valores_nuevo[col] = st.number_input(
                    f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
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

            st.success(f"El nuevo estudiante fue asignado al cluster {cluster_nuevo}")

            # ---- Interpretacion automatica del cluster ----
            if cluster_nuevo in labels_clusters:
                fila = resumen.loc[cluster_nuevo]

                st.markdown(
                    f"### Cluster {cluster_nuevo} — {labels_clusters[cluster_nuevo]}"
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

        st.subheader("Tabla de ejemplo con cluster asignado")
        st.dataframe(df_result.head(20))


if __name__ == "__main__":
    main()