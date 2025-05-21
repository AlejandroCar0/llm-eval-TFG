#!/bin/bash
WORK_DIR=$(realpath $(dirname $0))
password=$1
architecture=$(uname -m)
extension=""

if [ $architecture == "x86_64" ]
then
    extension='amd'
else
    extension='arm'
fi

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
OLLAMA_PATH=$WORK_DIR/ollama
mkdir -p $OLLAMA_PATH
curl -L https://github.com/ollama/ollama/releases/download/v0.5.11/ollama-linux-${extension}64.tgz -o $OLLAMA_PATH/ollama.tgz
tar xzf $OLLAMA_PATH/ollama.tgz -C $OLLAMA_PATH
ls $OLLAMA_PATH | grep -v -E "(^bin$|^lib$)" | xargs -I{} rm -rf $OLLAMA_PATH/{}

#Download and configure the nodeExporter for prometheus
EXPORTER_PATH=$WORK_DIR/node_exporter
mkdir -p $EXPORTER_PATH
curl -o $EXPORTER_PATH/nodeExporter.tar.gz -L https://github.com/prometheus/node_exporter/releases/download/v1.9.1/node_exporter-1.9.1.linux-amd64.tar.gz
tar xf $EXPORTER_PATH/nodeExporter.tar.gz -C $EXPORTER_PATH
mv $EXPORTER_PATH/node_exporter-*/* $EXPORTER_PATH
ls $EXPORTER_PATH | grep -v -e "^node_exporter$" | xargs -I{} rm -rf $EXPORTER_PATH/{}
