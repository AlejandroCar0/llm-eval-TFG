LLM Profiler
🚀 A tool for characterizing Large Language Models (LLMs) with real-time monitoring and resource analysis using Ollama, Prometheus, and Streamlit.

📋 Table of Contents
Overview
Features
Prerequisites
Installation
Quick Start
Usage
Dashboard
Project Structure
Contributing
License
🎯 Overview
This project provides a robust framework for: - Evaluating LLM performance on remote machines - Monitoring system resources (CPU, GPU, memory) during inference - Analyzing model responses and scoring - Visualizing results through an interactive dashboard

✨ Features
Remote LLM Evaluation: Execute evaluations on remote machines via SSH
Real-time Monitoring: Track system resources using Prometheus
GPU Metrics: Monitor GPU utilization and memory usage
Interactive Dashboard: Streamlit-based visualization of results
Multiple Model Support: Test various LLM models through Ollama
Comprehensive Logging: Detailed execution logs and debugging
🔧 Prerequisites
Python 3.8+
SSH access to target machine(s)
Prometheus (for metrics collection)
Ollama (for LLM inference)
🚀 Installation
1. Install Prometheus
For macOS (ARM64):
curl -LO https://github.com/prometheus/prometheus/releases/download/v3.3.1/prometheus-3.3.1.darwin-arm64.tar.gz
tar -xvf prometheus-3.3.1.darwin-arm64.tar.gz
sudo mv ./prometheus-3.3.1.darwin-arm64/prometheus /usr/local/bin/
sudo mv ./prometheus-3.3.1.darwin-arm64/promtool /usr/local/bin/
For Linux (AMD64):
curl -LO https://github.com/prometheus/prometheus/releases/download/v3.3.1/prometheus-3.3.1.linux-amd64.tar.gz
tar -xvf prometheus-3.3.1.linux-amd64.tar.gz
sudo mv ./prometheus-3.3.1.linux-amd64/prometheus /usr/local/bin/
sudo mv ./prometheus-3.3.1.linux-amd64/promtool /usr/local/bin/
Verify Installation:
prometheus --version
Alternative Installation (Local Path):
If you prefer not to install globally:

export PATH=$HOME/prometheus/:$PATH
2. Setup Python Environment
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
🎮 Quick Start
Configure your experiment: bash python3 llm_llama_eval.py --help

Run evaluation: bash python3 llm_llama_eval.py --host <remote-host> --user <username> --key <private-key-path>

View results: bash streamlit run data_representation/dashboard.py

📖 Usage
Command Line Options
Main Evaluation Script (llm_llama_eval.py)
python3 llm_llama_eval.py [OPTIONS]
Option	Required	Description
-i, --ip-address TEXT	✅ Required	IP address of the host where the test will be executed
-u, --user TEXT		Name of the user to connect to target destination
-p, --password TEXT		Password of the user (alternative to private key)
-pk, --private-key TEXT		Path to private key (.pem format) for SSH authentication
-ov, --ollama-version TEXT		Ollama version to install (format: "vx.x.x")
-nv, --node-version TEXT		Node exporter version to install (format: "vx.x.x")
-ro, --reinstall-ollama		Force reinstallation of Ollama even if already installed
Example usage:

# Using private key authentication
python3 llm_llama_eval.py -i 192.168.1.100 -u ubuntu -pk ./my-key.pem

# Using password authentication
python3 llm_llama_eval.py -i 192.168.1.100 -u ubuntu -p mypassword

# With specific versions
python3 llm_llama_eval.py -i 192.168.1.100 -u ubuntu -pk ./key.pem -ov v0.1.32 -nv v1.5.0
Dashboard Script (dashboard.py)
python3 data_representation/dashboard.py [OPTIONS]
Option	Description
-d, --directory DIRECTORY	Directory containing the metrics files for automatic loading
-h, --help	Show help message and exit
Configuration Files
prometheus/prometheus.yml: Prometheus configuration
ollama/prompts.txt: Test prompts for evaluation
ollama/answers.txt: Answers for test prompts (when applicable)
prometheus/querys.txt: Prometheus queries for metrics collection
ollama/model_list.txt: List of models to evaluate
📊 Dashboard
The interactive dashboard provides:

Model Performance Metrics: Response times, accuracy scores
Resource Utilization: CPU, memory, GPU usage over time
Comparative Analysis: Side-by-side model comparisons
Export Capabilities: Download results in various formats
Launching the Dashboard
You have two options to run the dashboard:

Option 1: Manual File Upload (Interactive Mode)
streamlit run data_representation/dashboard.py
Then access the dashboard at http://localhost:8501 and manually upload your experiment files in this order: 1. Ollama metrics 2. Model scores
3. Prometheus metrics 4. General information

Option 2: Automatic File Loading (Directory Mode)
# Point to a directory containing your experiment results
streamlit run data_representation/dashboard.py -- --directory /path/to/experiment/results

# Example with actual experiment directory
streamlit run data_representation/dashboard.py -- --directory ./experiment_results/2025-07-11-10-30-15
Note: When using the --directory option, the dashboard will automatically detect and load files matching these patterns: - ollama_metrics.csv or ollama_metrics*.csv - models_score.csv or *score*.csv - prometheus_metrics.csv or prometheus_metrics*.csv - general_info.txt or general_info*.txt

📁 Project Structure
llm-eval-TFG/
├── data_representation/     # Dashboard and visualization
│   └── dashboard.py
├── experiment_results/      # Stored experiment outputs
├── gpu_exporter/           # GPU metrics collection
├── logger/                 # Logging utilities
├── metrics/                # Collected metrics storage
├── ollama/                 # Ollama integration
├── prometheus/             # Prometheus configuration
├── versions/               # Version management
├── llm_llama_eval.py      # Main evaluation script
├── requirements.txt        # Python dependencies
└── README.md              # This file
🤝 Contributing
Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Prometheus for metrics collection
Ollama for LLM inference
Streamlit for dashboard framework
For more information or support, please open an issue in the repository.