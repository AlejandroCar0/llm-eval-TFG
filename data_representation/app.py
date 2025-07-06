import streamlit as st
import pandas as pd
import os
import random
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, DatetimeTickFormatter
from bokeh.layouts import column
from bokeh.palettes import Category10

st.set_page_config(layout="wide")


def line_graphic(df: pd.DataFrame, columns, title, random_color: bool = False):
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
        if not random_color:
            p.line(
                x='timestamp',
                y=col,
                source=source,
                line_width=2,
                color=colors[i % len(colors)],
                legend_label=col
            )
        else :
            numero = random.randint(0, 9)
            p.line(
                x='timestamp',
                y=col,
                source=source,
                line_width=2,
                color=colors[numero],
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


def wait_for_file(file_name: str, file_type: str):
    file = st.file_uploader(f"Upload your {file_name}", type = file_type)

    if not file:
        st.markdown(f"### Waiting for {file_name}")
        st.stop()
    else:
        st.success(f"File {file.name} uploaded!")
    
    return file

def print_txt_file(file):
    content = file.read().decode("utf-8")
    st.text_area("General information from the sut", content, height=300)

def get_dataframe(file: str) -> pd.DataFrame:
    data = pd.read_csv(file,delimiter=";")
    for col in data.columns:
        if col != "Model":
            data[col].apply(pd.to_numeric, errors = "coerce")

    data["timestamp"] = pd.to_datetime(data["timestamp"], unit = "s")

    return data


ollama_file = wait_for_file("ollama_metrics_file", file_type = "csv")
ollama_score_file = wait_for_file("ollama_score_file", file_type = "csv")
prometheus_metrics_file = wait_for_file("Prometheus_metrics_file", file_type = "csv")
general_info_file = wait_for_file("General_info", file_type = "txt")
print_txt_file(general_info_file)


ollama_df = get_dataframe(ollama_file)
prometheus_df = get_dataframe(prometheus_metrics_file)
score_df = pd.read_csv(ollama_score_file, delimiter = ";")

ollama_df['total_duration'] = (ollama_df['total_duration'] / 1e9).round(2)
ollama_df['load_duration'] = (ollama_df['load_duration'] / 1e9).round(2)
ollama_df['prompt_eval_duration'] = (ollama_df['prompt_eval_duration'] / 1e9).round(2)
ollama_df['eval_duration'] = (ollama_df['eval_duration'] / 1e9).round(2)


st.dataframe(ollama_df)
st.dataframe(prometheus_df)

fusion_df = pd.merge_asof(ollama_df,prometheus_df, on="timestamp", direction="backward")

st.dataframe(fusion_df)
models = []
models = set(ollama_df['model'])

columns = [
    ['total_duration', 
     'load_duration',
     'prompt_eval_duration',
     'eval_duration',
     ],
     [
         'prompt_eval_count',
         'eval_count' 
     ],
     [
         'cpu'
     ],
     [
         'memory'
     ],
     [
         'disk_read_bytes',
         'disk_written_bytes',
         'disk_reads_completed',
         'disk_writes_completed',
         'disk_busy_time',
         'disk_used_bytes'
     ]
]
if "gpu_utilization" in fusion_df:
    columns.append(['gpu_utilization'])
    columns.append(['gpu_memory_total'])
    columns.append(['gpu_power_usage'])
    columns.append(['gpu_temperature'])
    columns.append(['gpu_encoder_util','gpu_decoder_util'])

for column in columns:
    st.markdown(f"## {','.join(column)} INFORMATION")
    i = 0
    charts = []
    for model in models:
        charts.append(line_graphic(fusion_df[fusion_df['model'] == model], column, title= f"{model} stats for: {','.join(column)}"))
        i +=1
        if i % 3 == 0:
            deploy_three_chart(charts)
            i = 0
            charts = []


st.markdown("### Models marks:")

p = figure(x_range=score_df['Model'], height=400, title="Score per Modelo",
           tools="pan,wheel_zoom,box_zoom,reset,save", toolbar_location="above")
p.vbar(x='Model', top='Score', width=0.6, source=score_df, color='darkorange')

# Etiquetas y estética
p.xgrid.grid_line_color = None
p.y_range.start = 0
p.xaxis.axis_label = "Model"
p.yaxis.axis_label = "Score"

st.bokeh_chart(p, use_container_width=True)

st.markdown("### OVERALL STATS")

i = 0
charts = []
columns.pop(0)
columns.pop(0)
for column in columns:
    charts.append(line_graphic(prometheus_df, column, title= "", random_color = True))
    i +=1
    if i % 3 == 0:
        deploy_three_chart(charts)
        i = 0
        charts = []