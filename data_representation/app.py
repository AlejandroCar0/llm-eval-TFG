import streamlit as st
import pandas as pd
import os
import random
import argparse
import sys
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, DatetimeTickFormatter
from bokeh.layouts import column
from bokeh.palettes import Category10

st.set_page_config(layout="wide")

# Parse command line arguments
parser = argparse.ArgumentParser(description='LLM Evaluation Dashboard')
parser.add_argument('--directory', '-d', type=str, help='Directory containing the metrics files')
args = parser.parse_args()

# Global variable to store the directory path
METRICS_DIRECTORY = args.directory


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
    if METRICS_DIRECTORY:
        # Auto-load from directory
        return load_file_from_directory(file_name, file_type)
    else:
        # Interactive file upload
        file = st.file_uploader(f"Upload your {file_name}", type = file_type)

        if not file:
            st.markdown(f"### Waiting for {file_name}")
            st.stop()
        else:
            st.success(f"File {file.name} uploaded!")
        
        return file

def load_file_from_directory(file_name: str, file_type: str):
    """Load file from the specified directory based on file name pattern"""
    if not os.path.exists(METRICS_DIRECTORY):
        st.error(f"Directory {METRICS_DIRECTORY} does not exist!")
        st.stop()
    
    # Define file mapping based on expected file names
    file_patterns = {
        "ollama_metrics_file": ["ollama_metrics.csv", "ollama_metrics*.csv"],
        "ollama_score_file": ["models_score.csv", "models_puntuation.csv", "*score*.csv"],
        "Prometheus_metrics_file": ["prometheus_metrics.csv", "prometheus_metrics*.csv"],
        "General_info": ["general_info.txt", "general_info*.txt"]
    }
    
    patterns = file_patterns.get(file_name, [f"{file_name}.{file_type}"])
    
    for pattern in patterns:
        if "*" in pattern:
            import glob
            files = glob.glob(os.path.join(METRICS_DIRECTORY, pattern))
            if files:
                file_path = files[0]  # Take the first match
                break
        else:
            file_path = os.path.join(METRICS_DIRECTORY, pattern)
            if os.path.exists(file_path):
                break
    else:
        st.error(f"Could not find file matching patterns {patterns} in directory {METRICS_DIRECTORY}")
        st.stop()
    
    st.success(f"Loaded {file_name} from: {file_path}")
    return file_path

def print_txt_file(file):
    if isinstance(file, str):
        # File path - read from disk
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        # File object - read content
        content = file.read().decode("utf-8")
    st.text_area("General information from the sut", content, height=300)

def get_dataframe(file) -> pd.DataFrame:
    if isinstance(file, str):
        # File path - read from disk
        data = pd.read_csv(file, delimiter=";")
    else:
        # File object - read content
        data = pd.read_csv(file, delimiter=";")
    
    for col in data.columns:
        if col != "model":
           data[col] = data[col].apply(pd.to_numeric, errors = "coerce")

    data["timestamp"] = pd.to_datetime(data["timestamp"], unit = "s")

    return data


ollama_file = wait_for_file("ollama_metrics_file", file_type = "csv")
ollama_score_file = wait_for_file("ollama_score_file", file_type = "csv")
prometheus_metrics_file = wait_for_file("Prometheus_metrics_file", file_type = "csv")
general_info_file = wait_for_file("General_info", file_type = "txt")

# Display mode information
if METRICS_DIRECTORY:
    st.header(f"📁 Loading files from directory: {METRICS_DIRECTORY}")
else:
    st.header("📤 Interactive file upload mode")

print_txt_file(general_info_file)


ollama_df = get_dataframe(ollama_file)
prometheus_df = get_dataframe(prometheus_metrics_file)

# Handle score_df loading for both file objects and file paths
if isinstance(ollama_score_file, str):
    score_df = pd.read_csv(ollama_score_file, delimiter = ";")
else:
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
models = ollama_df['model'].unique()

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
    if len(charts) > 0:
        deploy_three_chart(charts)


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
if len(charts) > 0:
    deploy_three_chart(charts)