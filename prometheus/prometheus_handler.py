import threading
from prometheus.prometheus import Prometheus
import subprocess
import os
import time
import requests
from logger.log import logger


EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = f"{EXECUTION_PATH}/../metrics"
#Hay que mirar en la maquina destino no en esta.


class PrometheusHandler():

    def __init__(self, remote_ip_address: str, gpu_available: bool):
        self.log = logger.getChild(__file__)
        self.remote_ip_address = remote_ip_address
        self.gpu_available = gpu_available

        self.export_metrics = "timestamp;cpu;memory;disk_read_bytes;disk_written_bytes;disk_reads_completed;disk_writes_completed;disk_busy_time;disk_used_bytes;"
        if self.gpu_available:
            self.export_metrics +="gpu_utilization;gpu_memory_used;gpu_memory_total;gpu_power_usage;gpu_temperature;gpu_encoder_util;gpu_decoder_util\n"
        else :
            self.export_metrics += "\n" 
        
        self._start_prometheus()

        self.stop_signal = threading.Event()
        self.collector = threading.Thread(target = self._process_metrics)
        self.collector.daemon = True
        
        os.system(f"mkdir -p {METRICS_PATH}") #deberia hacerse al princpio de llm-eval que prepare todas las rutas
        with open(f"{METRICS_PATH}/prometheus_metrics.csv", "w") as f:
            f.write(f"{self.export_metrics}")

    def _start_prometheus(self):
        self.log.debug_color(f"Starting prometheus...")
        self.prometheus = Prometheus(self.remote_ip_address)

        process = subprocess.Popen(
            ["prometheus",
             f'--config.file={EXECUTION_PATH}/prometheus.yml'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        try:
            self._wait_for_prometheus(60)
            self.log.debug_color(f"Prometheus started!")

        except Exception as e:
            raise

    def _wait_for_prometheus(self, timeout: int = 60):
        url = f'http://127.0.0.1:9090/-/ready'
        for i in range(timeout):
            try:
                response = requests.get(url, timeout=1)

                if response.status_code == 200:
                    break;
            
            except requests.RequestException:
                pass

            time.sleep(1)

        else:
            raise Exception(f"Faile to get up Prometheus in {timeout} seconds")

    def _file_to_list(self, path: str) -> list:
        result = []
        with open(f"{path}", "r") as f:
            result = [line.rstrip("\n") for line in f]

        return result

    def _read_querys(self):
        querys = []

        self.log.debug_color("Reading querys....")

        querys = self._file_to_list(f"{EXECUTION_PATH}/querys.txt")
        if self.gpu_available:
            querys.extend(self._file_to_list(f"{EXECUTION_PATH}/gpu_querys.txt"))

        self.log.debug_color(f"Querys read!")
        
        return querys

    def _collect_metrics(self, querys : list[str]):
        with open(f"{METRICS_PATH}/prometheus_metrics.csv", "a") as f:
            values = []
            values.append(str(time.time()))
            for query in querys:
                try:
                    self.log.debug_color(f"Doing query: \[{query}]")
                    data = str(self.prometheus.query(query))
                    values.append(f"{data}")
                    self.log.debug_color(f"Query done!")
                except Exception as e:
                    self.log.warning_color(f'Error in query: {query}: {e}')
                    values.append("Error")

            f.write(";".join(values) + "\n")
    
    def _process_metrics(self):
        querys = self._read_querys()

        while not self.stop_signal.is_set():
            self._collect_metrics(querys)
            time.sleep(2)
    
    def start_collection(self):
        self.log.debug_color("Starting metrics collection")
        self.collector.start()
    
    def stop_collection(self):
        self.log.debug_color("Stopping metrics collection")

        self.stop_signal.set()
        self.collector.join()

        self.log.debug_color("Metrics collection stopped")