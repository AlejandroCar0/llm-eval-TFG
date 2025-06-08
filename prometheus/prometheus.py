import os
import requests
import json
EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
class Prometheus():
    def __init__(self, remote_ip_address: str, node_exporter_port: str = "9100",gpu_exporter_port: str = 9115, time_interval = "1"):
        self.remote_ip_address = remote_ip_address
        self.node_exporter_port = node_exporter_port
        self.gpu_exporter_port = gpu_exporter_port
        self.time_interval = time_interval
        self.url = f"http://127.0.0.1:9090/api/v1/query"
        #Configurates the prometheus configuration file
        self.configuration_file = "" \
        "global:\n" \
        f"  scrape_interval: {self.time_interval}s\n" \
        "scrape_configs:\n" \
        "  - job_name: 'node_exporter'\n" \
        "    static_configs:\n" \
        f"     - targets: [\"{self.remote_ip_address}:{self.node_exporter_port}\"]\n" \
        f"  - job_name: 'gpu_exporter'\n" \
        "    static_configs:\n" \
        f"     - targets: [\"{self.remote_ip_address}:{self.gpu_exporter_port}\"]\n" \

        #Write the configuration in the configuration file
        os.system(f"touch {EXECUTION_PATH}/prometheus.yml")
        with open(f"{EXECUTION_PATH}/prometheus.yml", "w") as f:
            f.write(self.configuration_file)

        os.system(f"prometheus --config.file={EXECUTION_PATH}/prometheus.yml >/dev/null 2>&1 &")

    def __str__(self):
        to_string = f"PrometheusHandler configuration:\n"\
            f"RemoteHost: [{self.remote_ip_address}:{self.node_exporter_port}]\n"\
            f"TimeInterval: {self.time_interval}\n"\
            f"Configuration file:\n\n{self.configuration_file}"
        return to_string
    
    def query(self, query: str):
        payload = {"query" : query}
        response = requests.get(self.url, params=payload)
        data = response.json()
        data = data.get("data",{}).get("result",[])[0].get("value",[])
        #manejar errores en la obtencion del dato
        return data[1]
        

def main():
    p = Prometheus(remote_ip_address = "192.168.1.21")
    print(p.query("100 * (1 - (node_memory_MemAvailable_bytes{job=\"node_exporter\"} / node_memory_MemTotal_bytes{job=\"node_exporter\"}))"))
    print(p.query("100 - (avg(rate(node_cpu_seconds_total{job=\"node_exporter\", mode=\"idle\"}[1m])) * 100)"))
    print(p.query(""))
    #print(json.dumps(p.counter("node_memory_MemAvailable_bytes /1024/1024/1024"),indent=4))
