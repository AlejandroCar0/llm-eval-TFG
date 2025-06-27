import os
import requests
import json
from logger.log import logger
EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
class Prometheus():
    def __init__(self, remote_ip_address: str, node_exporter_port: str = "9100",gpu_exporter_port: str = 9115, time_interval = "1"):
        self.log = logger.getChild(__file__)
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

    def __str__(self):
        to_string = f"Prometheus configuration:\n"\
            f"RemoteHost: [{self.remote_ip_address}:{self.node_exporter_port}]\n"\
            f"TimeInterval: {self.time_interval}\n"\
            f"Configuration file:\n\n{self.configuration_file}"
        return to_string
    
    def query(self, query: str):
        payload = {"query" : query}
        try:
            response = requests.get(self.url, params=payload)
            
            if response.status_code  != 200:
                raise Exception(f"Error in http request status code: {response.status_code}")
            
            data = response.json()
            data = data.get("data",{}).get("result",[])

            if not data:
                raise Exception("Not data found in the response")
            
            data = data[0].get("value",[])

            if  not data or len(data) <= 1 or not data[1]:   
                raise Exception("Data was empty")
            
            return data[1]

        
        except (requests.exceptions.RequestException) as e:
            raise Exception(f'{e}')