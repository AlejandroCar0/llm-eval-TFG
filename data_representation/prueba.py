import streamlit as st
import pandas as pd
import os
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, DatetimeTickFormatter
from bokeh.layouts import column
from bokeh.palettes import Category10

#This line activates the wide mode for the web page
st.set_page_config(layout="wide")

EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = f"{EXECUTION_PATH}/../metrics"


def get_dataframe(csv_path: str) -> pd.DataFrame:

    raw_data = pd.read_csv(csv_path,sep = ";", dtype=str)
    df = raw_data.apply(pd.to_numeric, errors = "coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit = "s")


    return df


def line_graphic(df: pd.DataFrame, columns, title):
    df_plot = df.copy()
    df_plot["timestamp_str"] = df_plot["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    source = ColumnDataSource(df_plot)

    p = figure(
        x_axis_type="datetime",
        title=f"{title}",
        width=900,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="above"
    )

    colors = Category10[10]

    for i, col in enumerate(columns):
        p.line(
            x='timestamp',
            y=col,
            source=source,
            line_width=2,
            color=colors[i % len(colors)],
            legend_label=col
        )
    hover = HoverTool(tooltips=[
            ("Time", "@timestamp_str"),
            *[(col, f"@{col}") for col in columns]
        ],
        mode = "vline"
        )
    
    p.add_tools(hover)

    p.xaxis.formatter = DatetimeTickFormatter(
        seconds=["%H:%M:%S"],
        minsec=["%H:%M:%S"],
        minutes=["%H:%M:%S"],
        hourmin=["%H:%M:%S"],
        hours=["%H:%M:%S"],
        days=["%H:%M:%S"],
        months=["%H:%M:%S"],
        years=["%H:%M:%S"],
    )

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.xaxis.axis_label = "Timestamp"
    p.yaxis.axis_label = "Value"

    return p


def deploy_three_chart(charts: list):

    cols = st.columns(3)
    for i in range(min(len(charts), len(cols))):
        cols[i].bokeh_chart(charts[i], use_container_width=True)

#Empieza el codigo principal:

st.title("Visualización de resultados de benchmarks")


# Carga del CSV (pon la ruta correcta a tu archivo)

dataframe = get_dataframe(f"{METRICS_PATH}/prometheus_metrics.csv")


st.subheader("Previsualizacion del dataframe:")

st.write("Datos cargados:")
st.dataframe(dataframe)


p1 = line_graphic(dataframe,["cpu","memory"], "CPU&Memory")
p2 = line_graphic(dataframe,["cpu","gpu_utilization"], "GPU&CPU")
p3 = line_graphic(dataframe,["gpu_utilization","memory"], "GPU&Memory")

deploy_three_chart([p1,p2,p3])
deploy_three_chart([p1,p2])

#st.markdown("---")
