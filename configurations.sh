#!/bin/bash
execution_dir=$(realpath $(dirname $0))
target_name=ollama-linux-amd64
password=$1

export WORK_DIR=$execution_dir

echo $password | sudo -S apt update
echo $password | sudo -S apt upgrade
echo $password | sudo -S apt install -y p7zip-full \
git \
netcat-traditional \
python3 \
python3-venv \
python3-pip \
curl
echo $password | sudo -S apt clean && rm -rf /var/lib/apt/list/*

#Download and configure ollama
mkdir $WORK_DIR/ollama
export OLLAMA_PATH=$WORK_DIR/ollama
curl -L https://github.com/ollama/ollama/releases/download/v0.5.11/ollama-linux-amd64.tgz -o "$OLLAMA_PATH/$target_name.tgz"
gunzip $OLLAMA_PATH/$target_name.tgz 
tar xf $OLLAMA_PATH/$target_name.tar -C $OLLAMA_PATH
ls  $OLLAMA_PATH | grep -v -e "bin" -e "lib" | xargs rm -rf

#Download and configure the nodeExporter for prometheus

curl -L https://github.com/prometheus/node_exporter/releases/download/v1.9.1/node_exporter-1.9.1.linux-amd64.tar.gz -o "$WORK_DIR/nodeExporter.tar.gz"
tar xf $WORK_DIR/nodeExporter.tar.gz -C $WORK_DIR/node_exporter
rm -rf $WORK_DIR/nodeExporter.tar.gz
