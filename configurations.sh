#!/bin/bash
WORK_DIR=$(realpath $(dirname $0))
password=$1
ollama_version=$2
prometheus_version=$3
reinstall_ollama=$4
architecture=$(uname -m)
extension=""

if [ $architecture == "x86_64" ]
then
    extension='amd'
else
    extension='arm'
fi

echo $password | sudo -S apt update -y
#echo $password | sudo -S apt upgrade -y 
echo $password | sudo -S apt install -y p7zip-full \
git \
netcat-traditional \
python3 \
python3-venv \
python3-pip \
curl
echo $password | sudo -S apt clean && rm -rf /var/lib/apt/list/*

if [ extension == "arm" ]
then
    echo $password | sudo -S pip3 install -U jetson-stats
fi

#Download and configure ollama
OLLAMA_PATH=$WORK_DIR/ollama

# Check if Ollama is already installed and if we should reinstall
if [ -d "$OLLAMA_PATH/bin" ] && [ -f "$OLLAMA_PATH/bin/ollama" ] && [ "$reinstall_ollama" != "true" ]; then
    echo "Ollama is already installed and reinstall flag is not set. Skipping Ollama installation."
else
    if [ -d "$OLLAMA_PATH" ] && [ "$reinstall_ollama" = "true" ]; then
        echo "Removing existing Ollama installation for reinstallation..."
        rm -rf $OLLAMA_PATH
    fi
    
    echo "Installing Ollama version $ollama_version..."
    mkdir -p $OLLAMA_PATH
    curl -L https://github.com/ollama/ollama/releases/download/${ollama_version}/ollama-linux-${extension}64.tgz -o $OLLAMA_PATH/ollama.tgz
    tar xzf $OLLAMA_PATH/ollama.tgz -C $OLLAMA_PATH
    ls $OLLAMA_PATH | grep -v -E "(^bin$|^lib$)" | xargs -I{} rm -rf $OLLAMA_PATH/{}
    echo "Ollama installation completed."
fi

#Download and configure the nodeExporter for prometheus
EXPORTER_PATH=$WORK_DIR/node_exporter
mkdir -p $EXPORTER_PATH
curl -o $EXPORTER_PATH/nodeExporter.tar.gz -L https://github.com/prometheus/node_exporter/releases/download/${prometheus_version}/node_exporter-${prometheus_version#v}.linux-${extension}64.tar.gz
tar xf $EXPORTER_PATH/nodeExporter.tar.gz -C $EXPORTER_PATH
mv $EXPORTER_PATH/node_exporter-*/* $EXPORTER_PATH
ls $EXPORTER_PATH | grep -v -e "^node_exporter$" | xargs -I{} rm -rf $EXPORTER_PATH/{}
nohup $EXPORTER_PATH/node_exporter &>/dev/null &
