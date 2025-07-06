# First you will need to install prometheus on your computer to take the analysis of the remote machine

## DOWNLOAD prometheus app

### for MACOS OR ARM ARchitectures

curl -LO https://github.com/prometheus/prometheus/releases/download/v3.3.1/prometheus-3.3.1.darwin-arm64.tar.gz

### for Linux OR AMD architectures

curl -LO https://github.com/prometheus/prometheus/releases/download/v3.3.1/prometheus-3.3.1.linux-amd64.tar.gz

## Extract the files:

tar -xvf prometheus-*.tar.gz 

## Move the binaries "prometheus" and "promtool" to the /usr/local/bin or any place in the path

sudo mv ./prometheus-3.3.1.darwin-arm64/prometheus /usr/local/bin/
sudo mv ./prometheus-3.3.1.darwin-arm64/promtool /usr/local/bin/

if prometheus is now on your /usr/local/bin you can do any comand, for example:
prometheus --version will work

##### Alternative moving prometheus to the path
Take the path where prometheus has been installed, example : /home/pepe/prometheus/

export PATH=$HOME/prometheus/:$PATH


## Install de requirements.txt in your virual-env to run the project

Use pip install -r requirements.txt


## Run the llm_llama_eval.py

Use python3 llm_llama_eval.py --help and follow the instructions


## Once you have ran your experiment you can watch for the results using streamlit

user streamlit run data_representation/app.py

Then follow the URL that streamlit gives to you, and upload the files in this order:
- Ollama metrics
- Models scores
- Prometheus metrics
- General inforamtion