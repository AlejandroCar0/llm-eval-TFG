import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = f"{EXECUTION_PATH}/../metrics"
def dataframe_from_csv(csv_path: str) -> pd.DataFrame: 
    with open(f"{csv_path}", "r") as f:
        data = f.readlines()

    headers = data[0].strip().split(";")
    data_lines = data[1:]
    data_lines = [datita.strip().split(";") for datita in data_lines]
    rows = []
    dataframe = pd.DataFrame()
    for i, name in enumerate(headers):
        dataframe[name] = [float(row[i]) for row in data_lines]
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"],unit = "s")
    return dataframe

def grafico_lineas(dataframe: pd.DataFrame, metric :str):
    fig, ax = plt.subplots()
    ax.plot(dataframe["timestamp"], dataframe[metric],'go--', linewidth=2, markersize=12, color = "red")
    labels = [f"{float(item):.2f}%" for item in ax.get_yticks()]
    ax.set_yticklabels(labels)

    #Titulos y etiquetas
    ax.set_title(f"------{metric}------",fontsize=20, fontweight='bold', color='#333333', pad=20)
    ax.set_xlabel('Tiempo (min)', fontsize=14, fontweight='bold', color='#555555', labelpad=15)
    ax.set_ylabel(f"{metric}", fontsize=14, fontweight='bold', color='#555555', labelpad=15)

    # Estetico
    ax.tick_params(axis='x', rotation=45)
    
    
    ax.grid(True, linestyle='--', alpha=0.6)

    # Fondo blanco con borde gris claro
    ax.set_facecolor("#000000")
    fig.patch.set_facecolor("#00C3FFFF")

    # Mejorar la apariencia de los ticks
    ax.tick_params(axis='both', which='major', labelsize=12, colors="#000000")

    return fig

st.title("Visualización de resultados de benchmarks")

# Carga del CSV (pon la ruta correcta a tu archivo)

dataframe = dataframe_from_csv(f"{METRICS_PATH}/prometheus_metrics.csv")


st.subheader("Previsualizacion del dataframe:")

st.write("Datos cargados:")
st.dataframe(dataframe)

#resumir datos
dataframe = dataframe.iloc[::2, :]
# Puedes mostrar gráficos, por ejemplo:
columnas = dataframe.select_dtypes(include = ['number']).columns.to_list()


for columna in columnas:
    st.pyplot(grafico_lineas(dataframe,columna))
"""
st.write(columnas)
graficos = ["Barras","linea","area"]

graphic = st.selectbox("Select graphic type:", graficos)
eje_x = "timestamp"
col_scatter_y = st.selectbox("Select metric to show", columnas, index=1)
if col_scatter_y:
    st.pyplot(grafico_lineas(dataframe,eje_x,ejey=col_scatter_y,titulo="Prueba"))
"""
st.markdown("---")
