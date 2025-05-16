# First you will need to install prometheus on your computer to take the analysis of the remote machine

## DOWNLOAD prometheus app

### for MACOS OR ARM ARchitectures

curl -LO https://github.com/prometheus/prometheus/releases/download/v3.3.1/prometheus-3.3.1.darwin-arm64.tar.gz

### for Linux OR AMD architectures

curl -LO https://github.com/prometheus/prometheus/releases/download/v3.3.1/prometheus-3.3.1.linux-amd64.tar.gz

## Extract the files:

tar -xvf prometheus-*.tar.gz 

## Move the binaries "prometheus" and "promtool" to the /usr/local/bin 

sudo mv ./prometheus-3.3.1.darwin-arm64/prometheus /usr/local/bin/
sudo mv ./prometheus-3.3.1.darwin-arm64/promtool /usr/local/bin/

if prometheus is now on your /usr/local/bin you can do any comand, for example:
prometheus --version will work