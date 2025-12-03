Integrantes: Kato Bello Martínez, Carlos Sepúlveda Navarrete.

CatStudy 
Aplicación interactiva desarrollada en Python con Streamlit 
que permite agrupar perfiles estudiantiles utilizando PCA (Análisis de Componentes Principales) y K-Means.
El usuario puede cargar un archivo CSV, seleccionar variables numéricas y visualizar clusters en un gráfico PCA 2D.

Requisitos: Python 3.10 o superior (proyecto desarrollado en Python 3.13)
Librerías: pandas, scikit-learn, streamlit

Dependencias:
pip install pandas scikit-learn streamlit

Ejecutar el proyecto:
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pandas scikit-learn streamlit
streamlit run app.py

Estructura simple del proyecto:
Proyecto/
│── app.py                # Interfaz Streamlit
│── modelo.py             # Logica de PCA y K-Means
│── datos_estudiantes.csv # Dataset de ejemplo
│── logo.png              # Logo de la app
│── README.md             # Este archivo
