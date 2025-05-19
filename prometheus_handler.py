import os
import requests
EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
class PrometheusHandler():
    def __init__(self, ip_address: str, port: str = "9090", time_interval = "1"):
        self.ip_address = ip_address
        self.port = port
        self.time_interval = time_interval
        #Configurates the prometheus configuration file
        self.configuration_file = "" \
        "global:\n" \
        f"  scrape_interval: {self.time_interval}s\n" \
        "scrape_configs:\n" \
        "  - job_name: 'remote'\n" \
        "    static_configs:\n" \
        f"     - targets: [\"{self.ip_address}:{self.port}\"]"
        #Write the configuration in the configuration file
        os.system(f"touch {EXECUTION_PATH}/prometheus.yml")
        with open(f"{EXECUTION_PATH}/prometheus.yml", "w") as f:
            f.write(self.configuration_file)
        os.system("prometheus --config.file=./prometheus.yml >/dev/null 2>&1 &")

    def __str__(self):
        to_string = f"PrometheusHandler configuration:\n"\
            f"RemoteHost: [{self.ip_address}:{self.port}]\n"\
            f"TimeInterval: {self.time_interval}\n"\
            f"Configuration file:\n\n{self.configuration_file}"
        return to_string
    
    def get_metrics(self):
        url = "http://127.0.0.1:9090/metrics"
        response = requests.get(url)
        print(response.text)

def main():
    p = PrometheusHandler(ip_address = "127.0.0.1")
    p.get_metrics()
main()