import streamlit as st
import pandas as pd
import os
import argparse
import logging
import glob
from typing import Optional
from bokeh.models import LabelSet, BoxAnnotation
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category10, Viridis256
import re

TITLE = "LLM Profiler Dashboard"

st.set_page_config(layout="wide", page_title=TITLE, page_icon="🚀")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description=TITLE)
parser.add_argument('--directory', '-d', type=str, help='Directory containing the metrics files')
args = parser.parse_args()

# Global variable to store the directory path
METRICS_DIRECTORY = args.directory

# File patterns for automatic loading
FILE_PATTERNS = {
    "ollama_metrics": ["ollama_metrics.csv", "ollama_metrics*.csv"],
    "ollama_score": ["models_score.csv", "*score*.csv"],
    "prometheus_metrics": ["prometheus_metrics.csv", "prometheus_metrics*.csv"],
    "general_info": ["general_info.txt", "general_info*.txt"]
}

def find_file_in_directory(directory: str, file_key: str) -> Optional[str]:
    """Find file in directory based on file patterns"""
    if not directory or not os.path.exists(directory):
        return None
    
    patterns = FILE_PATTERNS.get(file_key, [])
    
    for pattern in patterns:
        if "*" in pattern:
            files = glob.glob(os.path.join(directory, pattern))
            if files:
                return files[0] # Take the first match
        else:
            file_path = os.path.join(directory, pattern)
            if os.path.exists(file_path):
                return file_path
    
    return None

def load_files_from_directory(directory: str) -> dict:
    """Load all required files from the specified directory"""
    files = {}
    missing_files = []
    found_files = []
    
    for file_key in FILE_PATTERNS.keys():
        file_path = find_file_in_directory(directory, file_key)
        if file_path:
            files[file_key] = file_path
            found_files.append(f"{file_key}: {os.path.basename(file_path)}")
        else:
            missing_files.append(file_key)
    
    if found_files:
        st.success(f"✅ Found {len(found_files)} files")
    
    if missing_files:
        st.error(f"❌ Missing files: {', '.join(missing_files)}")
        return None
    
    return files

@st.cache_data
def load_dataframe(file_path: str) -> Optional[pd.DataFrame]:
    """Load and cache dataframe from file path"""
    try:
        data = pd.read_csv(file_path, delimiter=";")
        
        # Validate required columns
        if "timestamp" not in data.columns:
            st.error(f"Missing required 'timestamp' column in {file_path}")
            return None
        
        # Convert numeric columns safely
        for col in data.columns:
            if col not in ["model", "timestamp"]:
                data[col] = pd.to_numeric(data[col], errors="coerce")
        
        # Handle timestamp conversion with error checking
        try:
            data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s")
        except Exception as e:
            st.warning(f"Timestamp conversion failed, trying alternative formats: {e}")
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            
        return data
        
    except Exception as e:
        st.error(f"Error loading dataframe from {file_path}: {e}")
        logger.error(f"Error loading dataframe from {file_path}: {e}", exc_info=True)
        return None

def display_general_info(file_path: str):
    """Display general information from text file in a compact, formatted way"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse key-value pairs from the content
        info = {}
        for line in content.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                info[key.strip()] = value.strip()
        
        # Create a compact display with fundamental information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Operating System
            os_name = info.get('PRETTY_NAME', info.get('NAME', 'Unknown OS'))
            st.metric("🖥️ Operating System", os_name)
            
            # CPU
            cpu_model = info.get('CPU_MODEL', 'Unknown CPU')
            # Shorten CPU name if too long
            if len(cpu_model) > 35:
                cpu_model = cpu_model[:32] + "..."
            st.metric("⚡ CPU", cpu_model)
        
        with col2:
            # Memory
            memory = info.get('TOTAL_MEMORY', 'Unknown')
            st.metric("🧠 Memory", memory)
            
            # IP Address
            ip_address = info.get('IP_ADDRESS', 'Unknown')
            st.metric("🌐 IP Address", ip_address)
        
        with col3:
            # GPU
            gpu_info = info.get('GPU_INFO', 'No GPU info')
            st.metric("🎮 GPU", gpu_info)
        
        # Show full details in an expandable section if needed
        with st.expander("📋 Full System Details", expanded=False):
            st.text(content)
            
    except Exception as e:
        st.error(f"Error reading general info file: {e}")

@st.cache_data
def process_model_metrics(ollama_df: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    """Process and combine model metrics for analysis"""
    # Convert nanoseconds to seconds for duration columns
    duration_cols = ['total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration']
    ollama_processed = ollama_df.copy()
    
    for col in duration_cols:
        if col in ollama_processed.columns:
            ollama_processed[col] = ollama_processed[col] / 1e9 # Convert to seconds
    
    # Calculate derived metrics
    ollama_processed['response_time'] = ollama_processed['total_duration'] - ollama_processed['load_duration']
    ollama_processed['tokens_per_second'] = ollama_processed['eval_count'] / ollama_processed['eval_duration']
    ollama_processed['prompt_tokens_per_second'] = ollama_processed['prompt_eval_count'] / ollama_processed['prompt_eval_duration']
    
    # Aggregate by model
    model_stats = ollama_processed.groupby('model').agg({
        'total_duration': ['mean', 'std', 'count'],
        'response_time': ['mean', 'std'],
        'eval_duration': ['mean', 'std'],
        'load_duration': ['mean'],
        'tokens_per_second': ['mean', 'std'],
        'prompt_tokens_per_second': ['mean'],
        'eval_count': ['mean', 'sum'],
        'prompt_eval_count': ['mean']
    }).round(3)
    
    # Flatten column names
    model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns]
    model_stats = model_stats.reset_index()
    
    # Merge with scores
    if 'Model' in score_df.columns and 'Score' in score_df.columns:
        model_stats = model_stats.merge(score_df, left_on='model', right_on='Model', how='left')
    
    return model_stats

def create_performance_overview_chart(model_stats: pd.DataFrame):
    """Create the main performance overview scatter plot with family-based grouping"""
    if model_stats.empty:
        st.warning("No model statistics available for performance overview")
        return None
        
    if 'Score' not in model_stats.columns:
        st.warning("No score data available for performance overview")
        return None
    
    # Check for required columns
    required_cols = ['response_time_mean', 'tokens_per_second_mean', 'eval_count_sum', 'response_time_std']
    missing_cols = [col for col in required_cols if col not in model_stats.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return None
    
    # Filter out rows with NaN values
    clean_stats = model_stats.dropna(subset=['Score', 'response_time_mean', 'tokens_per_second_mean'])
    
    if clean_stats.empty:
        st.warning("No valid data points for performance overview")
        return None
    
    # Sort models by family and size
    models_list = clean_stats['model'].tolist()
    sorted_models = sort_models_by_family_and_size(models_list)
    
    # Get model family groupings and colors
    family_to_models, family_colors = get_model_families_and_colors(sorted_models)
    
    # Prepare data
    source_data = {
        'model': clean_stats['model'].tolist(),
        'avg_response_time': clean_stats['response_time_mean'].tolist(),
        'score': clean_stats['Score'].tolist(),
        'tokens_per_sec': clean_stats['tokens_per_second_mean'].tolist(),
        'total_tokens': clean_stats['eval_count_sum'].tolist(),
        'consistency': (1 / (clean_stats['response_time_std'] + 0.001)).tolist()
    }
    
    # Add family information to tooltips
    model_info_list = []
    colors_list = []
    for model in source_data['model']:
        family, size = parse_model_info(model)
        model_info_list.append(f"{model} ({family}, {size}B)")
        colors_list.append(get_model_color_with_shade(model, family_to_models, family_colors))
    
    source_data['model_info'] = model_info_list
    source_data['colors'] = colors_list
    
    source = ColumnDataSource(source_data)
    
    # Create figure
    p = figure(
        title="🎯 Model Performance Overview: Quality vs Speed (Grouped by Family)",
        x_axis_label="Average Response Time (seconds)",
        y_axis_label="Quality Score",
        width=800,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # Create scatter plot with family-based colors
    p.scatter(
        x='avg_response_time', 
        y='score', 
        size=15,
        color='colors',
        alpha=0.8,
        line_color='white',
        line_width=2,
        source=source
    )
    
    # Enhanced hover tool with family information
    hover = HoverTool(tooltips=[
        ("Model", "@model_info"),
        ("Quality Score", "@score{0.00}"),
        ("Avg Response Time", "@avg_response_time{0.00}s"),
        ("Tokens/Second", "@tokens_per_sec{0.0}"),
        ("Total Tokens", "@total_tokens{0,0}")
    ])
    p.add_tools(hover)
    
    # Add labels for each point
    labels = LabelSet(x='avg_response_time', y='score', text='model', 
                     x_offset=5, y_offset=5, source=source, 
                     text_font_size='9pt', text_color='black')
    p.add_layout(labels)
    
    return p

def create_response_time_distribution(ollama_df: pd.DataFrame):
    """Create box plot showing response time distribution by model with family grouping"""
    # Convert to seconds
    ollama_processed = ollama_df.copy()
    ollama_processed['response_time'] = (ollama_processed['total_duration'] - ollama_processed['load_duration']) / 1e9
    
    models = ollama_processed['model'].unique()
    sorted_models = sort_models_by_family_and_size(models)
    
    # Get family groupings and colors
    family_to_models, family_colors = get_model_families_and_colors(sorted_models)
    
    p = figure(
        title="📊 Response Time Distribution by Model (Grouped by Family)",
        x_range=sorted_models,
        y_axis_label="Response Time (seconds)",
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # Prepare data for box plots
    box_data = []
    
    for model in sorted_models:
        model_data = ollama_processed[ollama_processed['model'] == model]['response_time']
        if len(model_data) == 0:
            continue
            
        q1 = model_data.quantile(0.25)
        q2 = model_data.median()
        q3 = model_data.quantile(0.75)
        iqr = q3 - q1
        upper = min(model_data.max(), q3 + 1.5 * iqr)
        lower = max(model_data.min(), q1 - 1.5 * iqr)
        
        # Get color for this model
        color = get_model_color_with_shade(model, family_to_models, family_colors)
        
        # Find outliers
        outliers = model_data[(model_data > upper) | (model_data < lower)]
        
        box_data.append({
            'model': model,
            'q1': q1,
            'q2': q2,
            'q3': q3,
            'upper': upper,
            'lower': lower,
            'color': color,
            'outliers': outliers.tolist()
        })
    
    # Create box plots using quad for better control
    hover_data = []
    
    for data in box_data:
        model = data['model']
        model_index = sorted_models.index(model)  # Get numeric index for positioning
        
        # Prepare hover data
        family, size = parse_model_info(model)
        hover_info = {
            'x': model_index,
            'y': (data['q1'] + data['q3']) / 2,  # Middle of the box for hover target
            'model': model,
            'family': family,
            'size': f"{size}B",
            'median': f"{data['q2']:.3f}s",
            'q1': f"{data['q1']:.3f}s",
            'q3': f"{data['q3']:.3f}s",
            'iqr': f"{data['q3'] - data['q1']:.3f}s",
            'min': f"{data['lower']:.3f}s",
            'max': f"{data['upper']:.3f}s",
            'outliers_count': len(data['outliers'])
        }
        hover_data.append(hover_info)
        
        # Box (interquartile range) - use quad for precise control
        p.quad(top=data['q3'], bottom=data['q1'], 
               left=model_index-0.3, right=model_index+0.3,
               fill_color=data['color'], fill_alpha=0.7, line_color='black')
        
        # Median line - horizontal line across the box
        p.segment(x0=model_index-0.3, y0=data['q2'], 
                 x1=model_index+0.3, y1=data['q2'], 
                 line_width=3, line_color='white')
        
        # Whiskers - vertical lines
        p.segment(x0=model_index, y0=data['upper'], 
                 x1=model_index, y1=data['q3'], 
                 line_color='black', line_width=2)
        p.segment(x0=model_index, y0=data['lower'], 
                 x1=model_index, y1=data['q1'], 
                 line_color='black', line_width=2)
        
        # Whisker caps - horizontal lines at the ends
        p.segment(x0=model_index-0.1, y0=data['upper'], 
                 x1=model_index+0.1, y1=data['upper'], 
                 line_width=2, line_color='black')
        p.segment(x0=model_index-0.1, y0=data['lower'], 
                 x1=model_index+0.1, y1=data['lower'], 
                 line_width=2, line_color='black')
        
        # Outliers
        if data['outliers']:
            p.circle(x=[model_index] * len(data['outliers']), y=data['outliers'], 
                    size=6, color=data['color'], alpha=0.6)
    
    # Add invisible circles for hover functionality
    if hover_data:
        hover_source = ColumnDataSource(pd.DataFrame(hover_data))
        hover_circles = p.circle(x='x', y='y', size=20, alpha=0, source=hover_source)
        
        # Add hover tool with detailed statistics
        hover = HoverTool(
            renderers=[hover_circles],
            tooltips=[
                ("Model", "@model (@family, @size)"),
                ("Median", "@median"),
                ("Q1 (25%)", "@q1"),
                ("Q3 (75%)", "@q3"),
                ("IQR", "@iqr"),
                ("Min", "@min"),
                ("Max", "@max"),
                ("Outliers", "@outliers_count")
            ]
        )
        p.add_tools(hover)
    
    p.xaxis.major_label_orientation = 45
    return p

def create_resource_timeline(prometheus_df: pd.DataFrame, ollama_df: pd.DataFrame):
    """Create resource utilization timeline"""
    p = figure(
        title="📈 Resource Utilization Over Time",
        x_axis_type='datetime',
        y_axis_label="Utilization (%)",
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    if 'gpu_utilization' in prometheus_df.columns:
        p.line(prometheus_df['timestamp'], prometheus_df['gpu_utilization'], 
               legend_label="GPU Utilization", line_color='red', line_width=2)
    
    if 'cpu' in prometheus_df.columns:
        p.line(prometheus_df['timestamp'], prometheus_df['cpu'], 
               legend_label="CPU Utilization", line_color='blue', line_width=2)
    
    # Add model execution periods as shaded regions
    if not ollama_df.empty and 'timestamp' in ollama_df.columns:
        models = ollama_df['model'].unique()
        family_to_models, family_colors = get_model_families_and_colors(models)
        
        for i, row in ollama_df.iterrows():
            start_time = row['timestamp']
            duration = row['total_duration'] / 1e9  # Convert to seconds
            end_time = start_time + pd.Timedelta(seconds=duration)
            
            # Get family-based color for this model
            model_color = get_model_color_with_shade(row['model'], family_to_models, family_colors)
            
            # Add shaded box for model execution period
            box = BoxAnnotation(
                left=start_time.timestamp() * 1000,  # Convert to milliseconds for Bokeh
                right=end_time.timestamp() * 1000,
                fill_alpha=0.2,
                fill_color=model_color,
                line_color=model_color,
                line_alpha=0.5
            )
            p.add_layout(box)
    
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    return p

def create_tokens_per_second_chart(model_stats: pd.DataFrame):
    """Create tokens per second performance chart with family grouping"""
    if model_stats.empty:
        return None
    
    models = model_stats['model'].tolist()
    sorted_models = sort_models_by_family_and_size(models)
    
    # Reorder data according to sorted models
    sorted_stats = model_stats.set_index('model').loc[sorted_models].reset_index()
    tokens_per_sec = sorted_stats['tokens_per_second_mean'].tolist()
    
    # Get family groupings and colors
    family_to_models, family_colors = get_model_families_and_colors(sorted_models)
    
    p = figure(
        title="🚀 Model Throughput: Tokens per Second (Grouped by Family)",
        x_range=sorted_models,
        y_axis_label="Tokens per Second",
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # Assign colors to each model
    colors = []
    for model in sorted_models:
        colors.append(get_model_color_with_shade(model, family_to_models, family_colors))
    
    bars = p.vbar(x=sorted_models, top=tokens_per_sec, width=0.6, 
                  color=colors, alpha=0.8)
    
    # Add value labels on bars
    source = ColumnDataSource(dict(x=sorted_models, y=tokens_per_sec, 
                                  labels=[f"{val:.1f}" for val in tokens_per_sec]))
    labels = LabelSet(x='x', y='y', text='labels', x_offset=-10, y_offset=5,
                     source=source, text_font_size='10pt')
    p.add_layout(labels)
    
    p.xaxis.major_label_orientation = 45
    
    return p

def parse_model_info(model_name: str) -> tuple[str, float]:
    """Parse model name to extract family and parameter size.
    
    Returns:
        tuple: (family_name, parameter_size_in_billions)

    """
    model_name = model_name.lower().strip()
    
    # Common patterns for parameter sizes
    size_patterns = [
        r'(\d+\.?\d*)b\b',  # e.g., "7b", "13b", "0.5b"
        r'(\d+\.?\d*)-?billion',  # e.g., "7-billion", "13billion"
        r'(\d+\.?\d*)t\b',  # e.g., "1t" (trillion parameters, convert to billions)
    ]
    
    parameter_size = None
    for pattern in size_patterns:
        match = re.search(pattern, model_name)
        if match:
            size = float(match.group(1))
            if 't' in pattern:  # trillion parameters
                size *= 1000  # convert to billions
            parameter_size = size
            break
    
    # Extract family name (everything before the colon or size indicator)
    family_patterns = [
        r'^([^:]+):',  # e.g., "gemma3:12b" -> "gemma3"
        r'^([a-zA-Z-]+)',  # e.g., "orca-mini" -> "orca-mini"
    ]
    
    family_name = model_name
    for pattern in family_patterns:
        match = re.search(pattern, model_name)
        if match:
            family_name = match.group(1)
            break
    
    # Handle special cases and estimate sizes for known models
    size_estimates = {
        'orca-mini': 3.0,  # Typically around 3B
        'orca': 13.0,  # Standard Orca is usually 13B
        'vicuna': 7.0,  # Common Vicuna size
        'alpaca': 7.0,  # Common Alpaca size
        'codellama': 7.0,  # Default CodeLlama size
        'llama': 7.0,  # Default Llama size
    }
    
    if parameter_size is None:
        # Try to estimate based on family name
        for family_key, estimated_size in size_estimates.items():
            if family_key in family_name:
                parameter_size = estimated_size
                break
        
        # If still no size, default to 1.0B
        if parameter_size is None:
            parameter_size = 1.0
    
    return family_name, parameter_size

def get_model_families_and_colors(models: list) -> tuple[dict, dict]:
    """Get family groupings and assign colors to model families.
    
    Returns:
        tuple: (family_to_models_dict, family_to_color_dict)

    """
    # Parse all models
    model_info = {}
    families = set()
    
    for model in models:
        family, size = parse_model_info(model)
        model_info[model] = (family, size)
        families.add(family)
    
    # Assign base colors to families
    family_colors = {}
    base_colors = Category10[10] if len(families) <= 10 else Viridis256[::int(256/len(families))]
    
    for i, family in enumerate(sorted(families)):
        family_colors[family] = base_colors[i % len(base_colors)]
    
    # Group models by family
    family_to_models = {}
    for model in models:
        family, size = model_info[model]
        if family not in family_to_models:
            family_to_models[family] = []
        family_to_models[family].append((model, size))
    
    # Sort models within each family by parameter size
    for family in family_to_models:
        family_to_models[family].sort(key=lambda x: x[1])  # Sort by parameter size
    
    return family_to_models, family_colors

def sort_models_by_family_and_size(models: list) -> list:
    """Sort models by family name and then by parameter size within each family.
    """
    model_info = [(model, *parse_model_info(model)) for model in models]
    # Sort by family name first, then by parameter size
    model_info.sort(key=lambda x: (x[1], x[2]))
    return [model[0] for model in model_info]

def get_model_color_with_shade(model: str, family_to_models: dict, family_colors: dict) -> str:
    """Get a color for a specific model based on its family and position within the family.
    Models in the same family get different shades of the same base color.
    """
    family, size = parse_model_info(model)
    base_color = family_colors.get(family, "#888888")
    
    if family not in family_to_models:
        return base_color
    
    # Get models in this family sorted by size
    family_models = family_to_models[family]
    model_index = next((i for i, (m, s) in enumerate(family_models) if m == model), 0)
    
    # Create shades by adjusting the brightness
    num_models = len(family_models)
    if num_models == 1:
        return base_color
    
    # Convert hex to RGB for manipulation
    base_color = base_color.lstrip('#')
    r, g, b = tuple(int(base_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Create darker/lighter shades
    # Smallest model gets lightest shade, largest gets darkest
    shade_factor = 0.3 + 0.7 * (model_index / (num_models - 1))  # Range from 0.3 to 1.0
    
    r = int(r * shade_factor)
    g = int(g * shade_factor)
    b = int(b * shade_factor)
    
    return f"#{r:02x}{g:02x}{b:02x}"

def create_memory_usage_chart(prometheus_df: pd.DataFrame, ollama_df: pd.DataFrame):
    """Create memory usage timeline showing both system RAM and GPU memory"""
    p = figure(
        title="💾 Memory Usage Over Time",
        x_axis_type='datetime',
        y_axis_label="Memory Usage (%)",
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # System RAM usage (percentage)
    if 'memory' in prometheus_df.columns:
        p.line(prometheus_df['timestamp'], prometheus_df['memory'], 
               legend_label="System RAM (%)", line_color='blue', line_width=2)
    
    # GPU Memory usage (convert to percentage)
    if 'gpu_memory_used' in prometheus_df.columns and 'gpu_memory_total' in prometheus_df.columns:
        # Calculate GPU memory percentage, handling division by zero
        gpu_memory_pct = prometheus_df.apply(
            lambda row: (row['gpu_memory_used'] / row['gpu_memory_total']) * 100 
            if row['gpu_memory_total'] > 0 else 0, axis=1
        )
        p.line(prometheus_df['timestamp'], gpu_memory_pct, 
               legend_label="GPU Memory (%)", line_color='red', line_width=2)
        
        # Add GPU memory in GB as additional info
        gpu_memory_gb = prometheus_df['gpu_memory_used'] / (1024**3)
        max_gpu_gb = prometheus_df['gpu_memory_total'].max() / (1024**3)
        
        # Create hover tool with additional GPU memory info
        hover_source = ColumnDataSource(data=dict(
            timestamp=prometheus_df['timestamp'],
            ram_pct=prometheus_df['memory'] if 'memory' in prometheus_df.columns else [0] * len(prometheus_df),
            gpu_pct=gpu_memory_pct,
            gpu_gb=gpu_memory_gb,
            gpu_total_gb=[max_gpu_gb] * len(prometheus_df)
        ))
        
        # Add invisible circles for detailed hover information
        hover_circles = p.circle(x='timestamp', y='gpu_pct', size=8, alpha=0, source=hover_source)
        
        hover = HoverTool(
            renderers=[hover_circles],
            tooltips=[
                ("Time", "@timestamp{%F %T}"),
                ("System RAM", "@ram_pct{0.1f}%"),
                ("GPU Memory", "@gpu_pct{0.1f}%"),
                ("GPU Memory", "@gpu_gb{0.1f} / @gpu_total_gb{0.1f} GB")
            ],
            formatters={'@timestamp': 'datetime'}
        )
        p.add_tools(hover)
    
    # Add model execution periods as shaded regions
    if not ollama_df.empty and 'timestamp' in ollama_df.columns:
        models = ollama_df['model'].unique()
        family_to_models, family_colors = get_model_families_and_colors(models)
        
        for i, row in ollama_df.iterrows():
            start_time = row['timestamp']
            duration = row['total_duration'] / 1e9  # Convert to seconds
            end_time = start_time + pd.Timedelta(seconds=duration)
            
            # Get family-based color for this model
            model_color = get_model_color_with_shade(row['model'], family_to_models, family_colors)
            
            # Add shaded box for model execution period
            box = BoxAnnotation(
                left=start_time.timestamp() * 1000,  # Convert to milliseconds for Bokeh
                right=end_time.timestamp() * 1000,
                fill_alpha=0.15,
                fill_color=model_color,
                line_color=model_color,
                line_alpha=0.3
            )
            p.add_layout(box)
    
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    return p

def main():
    """Main application function"""
    st.title(f"🚀 {TITLE}")
    
    # Initialize session state for directory
    if 'current_directory' not in st.session_state:
        st.session_state.current_directory = METRICS_DIRECTORY
    
    # Directory selection section - compact layout
    col1, col2, col3 = st.columns([2, 3, 1])
    
    with col1:
        st.markdown("**📁 Data Directory:**")
    
    with col2:
        new_directory = st.text_input("Directory path:", 
                                    value=st.session_state.current_directory or "",
                                    placeholder="/path/to/metrics/directory",
                                    label_visibility="collapsed")
    
    with col3:
        if st.button("Load", type="primary"):
            if new_directory and os.path.exists(new_directory):
                st.session_state.current_directory = new_directory
                st.rerun()
            elif new_directory:
                st.error(f"Directory does not exist: {new_directory}")
            else:
                st.error("Please enter a valid directory path")
    
    # If no directory is set, stop here
    if not st.session_state.current_directory:
        st.warning("Please select a directory containing the metrics files.")
        st.stop()
    
    # Load files from directory
    with st.spinner("Loading data files..."):
        files = load_files_from_directory(st.session_state.current_directory)
    
    if not files:
        st.error("Failed to load required files. Please check the directory and file formats.")
        st.stop()
    
    # Display file loading status in one compact line
    file_names = [os.path.basename(path) for path in files.values()]
    st.info(f"📋 **Loaded files:** {' • '.join(file_names)}")
    
    # Display general information
    st.header("📊 System Information")
    display_general_info(files['general_info'])
    
    # Load dataframes
    st.header("📈 Data Loading")
    
    try:
        with st.spinner("Processing data files..."):
            ollama_df = load_dataframe(files['ollama_metrics'])
            prometheus_df = load_dataframe(files['prometheus_metrics'])
            score_df = pd.read_csv(files['ollama_score'], delimiter=";")
        
        if ollama_df is not None and prometheus_df is not None:
            st.success("✅ All data files loaded successfully!")
            
            # Display basic data info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Ollama Records", len(ollama_df))
                if 'model' in ollama_df.columns:
                    st.metric("Models", ollama_df['model'].nunique())
            
            with col2:
                st.metric("Prometheus Records", len(prometheus_df))
                if 'timestamp' in prometheus_df.columns:
                    time_range = prometheus_df['timestamp'].max() - prometheus_df['timestamp'].min()
                    # Convert to human readable format
                    total_seconds = int(time_range.total_seconds())
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    seconds = total_seconds % 60
                    
                    if hours > 0:
                        time_display = f"{hours}h {minutes}m {seconds}s"
                    elif minutes > 0:
                        time_display = f"{minutes}m {seconds}s"
                    else:
                        time_display = f"{seconds}s"
                    
                    st.metric("Time Range", time_display)
            
            with col3:
                st.metric("Score Records", len(score_df))
                if 'Score' in score_df.columns:
                    st.metric("Avg Score", f"{score_df['Score'].mean():.2f}")
            
            # Analysis sections
            st.header("📊 Model Performance Analysis")
            
            # Process model metrics
            with st.spinner("Processing model metrics..."):
                model_stats = process_model_metrics(ollama_df, score_df)
            
            # 1. Performance Overview (Quality vs Speed)
            st.subheader("🎯 Performance Overview: Quality vs Speed")
            st.markdown("*This chart shows the critical trade-off between model quality (score) and response speed. Bubble color indicates throughput (tokens/second).*")
            
            perf_chart = create_performance_overview_chart(model_stats)
            if perf_chart:
                st.bokeh_chart(perf_chart, use_container_width=True)
            
            # 2. Response Time Distribution
            st.subheader("📊 Response Time Consistency")
            st.markdown("*Box plots showing response time distribution for each model. Smaller boxes indicate more consistent performance.*")
            
            dist_chart = create_response_time_distribution(ollama_df)
            if dist_chart:
                st.bokeh_chart(dist_chart, use_container_width=True)
            
            # 3. Tokens per Second Performance
            st.subheader("🚀 Model Throughput")
            st.markdown("*Higher tokens/second means faster text generation - critical for production deployments.*")
            
            tokens_chart = create_tokens_per_second_chart(model_stats)
            if tokens_chart:
                st.bokeh_chart(tokens_chart, use_container_width=True)
            
            # 4. Resource Utilization Timeline
            st.subheader("📈 System Resource Usage")
            st.markdown("*Timeline showing GPU and CPU utilization during model execution. Vertical lines indicate model runs.*")
            
            resource_chart = create_resource_timeline(prometheus_df, ollama_df)
            if resource_chart:
                st.bokeh_chart(resource_chart, use_container_width=True)
            
            # 5. Memory Usage Over Time
            st.subheader("💾 Memory Usage Over Time")
            st.markdown("*Tracks system RAM and GPU memory usage over time, with model execution periods.*")
            
            memory_chart = create_memory_usage_chart(prometheus_df, ollama_df)
            if memory_chart:
                st.bokeh_chart(memory_chart, use_container_width=True)
            
            # 6. Model Statistics Summary
            st.subheader("� Model Performance Summary")
            
            # Create a summary table with family information
            if not model_stats.empty:
                # Add family and size information
                model_families = []
                model_sizes = []
                for model in model_stats['model']:
                    family, size = parse_model_info(model)
                    model_families.append(family)
                    model_sizes.append(f"{size}B")
                
                summary_df = pd.DataFrame({
                    'Model': model_stats['model'],
                    'Family': model_families,
                    'Size': model_sizes,
                    'Quality Score': model_stats.get('Score', [0] * len(model_stats)),
                    'Avg Response Time (s)': model_stats['response_time_mean'],
                    'Tokens/Second': model_stats['tokens_per_second_mean'],
                    'Total Requests': model_stats['total_duration_count'],
                    'Consistency (1/std)': (1 / (model_stats['response_time_std'] + 0.001)).round(2)
                })
                
                # Sort by family and size
                sorted_models = sort_models_by_family_and_size(model_stats['model'].tolist())
                summary_df = summary_df.set_index('Model').loc[sorted_models].reset_index()
                
                # Format the dataframe for better display
                summary_df['Quality Score'] = summary_df['Quality Score'].round(2)
                summary_df['Avg Response Time (s)'] = summary_df['Avg Response Time (s)'].round(2)
                summary_df['Tokens/Second'] = summary_df['Tokens/Second'].round(1)
                
                st.dataframe(summary_df, use_container_width=True)
                
                # Key insights
                st.subheader("� Key Insights")
                
                if len(summary_df) > 0:
                    best_quality = summary_df.loc[summary_df['Quality Score'].idxmax()]
                    fastest = summary_df.loc[summary_df['Tokens/Second'].idxmax()]
                    most_consistent = summary_df.loc[summary_df['Consistency (1/std)'].idxmax()]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "🏆 Highest Quality",
                            best_quality['Model'],
                            f"Score: {best_quality['Quality Score']}"
                        )
                    
                    with col2:
                        st.metric(
                            "⚡ Fastest Throughput",
                            fastest['Model'],
                            f"{fastest['Tokens/Second']:.1f} tok/s"
                        )
                    
                    with col3:
                        st.metric(
                            "🎯 Most Consistent",
                            most_consistent['Model'],
                            f"Consistency: {most_consistent['Consistency (1/std)']:.1f}"
                        )
            
            # Display data preview
            with st.expander("🔍 Data Preview", expanded=False):
                tab1, tab2, tab3 = st.tabs(["Ollama Data", "Prometheus Data", "Score Data"])
                
                with tab1:
                    st.dataframe(ollama_df.head(), use_container_width=True)
                
                with tab2:
                    st.dataframe(prometheus_df.head(), use_container_width=True)
                
                with tab3:
                    st.dataframe(score_df, use_container_width=True)
        
        else:
            st.error("Failed to load data files. Please check the file formats and content.")
    
    except Exception as e:
        st.error(f"An error occurred while processing data: {e}")
        logger.error(f"Data processing error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
