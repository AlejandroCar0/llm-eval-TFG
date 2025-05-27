import os
import requests
import json
EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
class Prometheus():
    def __init__(self, remote_ip_address: str, remote_port: str = "9100", time_interval = "1"):
        self.remote_ip_address = remote_ip_address
        self.remote_port = remote_port
        self.time_interval = time_interval
        self.url = f"http://127.0.0.1:9090/api/v1/query"
        #Configurates the prometheus configuration file
        self.configuration_file = "" \
        "global:\n" \
        f"  scrape_interval: {self.time_interval}s\n" \
        "scrape_configs:\n" \
        "  - job_name: 'remote'\n" \
        "    static_configs:\n" \
        f"     - targets: [\"{self.remote_ip_address}:{self.remote_port}\"]"

        #Write the configuration in the configuration file
        os.system(f"touch {EXECUTION_PATH}/prometheus.yml")
        with open(f"{EXECUTION_PATH}/prometheus.yml", "w") as f:
            f.write(self.configuration_file)

        #os.system(f"prometheus --config.file={EXECUTION_PATH}/prometheus.yml >/dev/null 2>&1 &")

    def __str__(self):
        to_string = f"PrometheusHandler configuration:\n"\
            f"RemoteHost: [{self.remote_ip_address}:{self.remote_port}]\n"\
            f"TimeInterval: {self.time_interval}\n"\
            f"Configuration file:\n\n{self.configuration_file}"
        return to_string
    
    def query(self, query: str):
        payload = {"query" : query}
        response = requests.get(self.url, params=payload)
        data = response.json()
        data = data.get("data",{}).get("result",[])[0].get("value",[])
        #manejar errores en la obtencion del dato
        return data
        

def main():
    p = Prometheus(remote_ip_address = "172.25.202.253")
    print(p.query("100 * (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))"))
    print(p.query("100 - (avg(rate(node_cpu_seconds_total{mode=\"idle\"}[1m])) * 100)"))
    #print(json.dumps(p.counter("node_memory_MemAvailable_bytes /1024/1024/1024"),indent=4))