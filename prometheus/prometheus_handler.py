import threading
from prometheus.prometheus import Prometheus
import threading
import os
import time
import torch
from logger.log import logger


EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = f"{EXECUTION_PATH}/../metrics"
#Hay que mirar en la maquina destino no en esta.


class PrometheusHandler():

    def __init__(self, remote_ip_address: str, gpu_available: bool):
        self.log = logger.getChild(__file__)
        self.export_metrics = "timestamp;cpu;memory;disk_read_bytes;disk_written_bytes;disk_reads_completed;disk_writes_completed;disk_busy_time;disk_used_bytes;"
        self.gpu_available = gpu_available

        if self.gpu_available:
            self.export_metrics +="gpu_utilization;gpu_memory_used;gpu_memory_total;gpu_power_usage;gpu_temperature;gpu_encoder_util;gpu_decoder_util\n"
        else :
            self.export_metrics += "\n" 
        self.log.debug_color(f"Starting prometheus...")
        self.prometheus = Prometheus(remote_ip_address)
        time.sleep(20)

        self.log.debug_color(f"Prometheus started!")

        self.stop_signal = threading.Event()
        self.collector = threading.Thread(target = self._process_metrics)
        self.collector.daemon = True
        os.system(f"mkdir -p {METRICS_PATH}") #deberia hacerse al princpio de llm-eval que prepare todas las rutas
        with open(f"{METRICS_PATH}/prometheus_metrics.csv", "w") as f:
            f.write(f"{self.export_metrics}")

    def _file_to_list(self, path: str) -> list:
        result = []
        with open(f"{path}", "r") as f:
            result = [line.rstrip("\n") for line in f]

        return result

    def _read_querys(self):
        querys = []

        self.log.debug_color("Reading querys....")

        querys = self._file_to_list(f"{EXECUTION_PATH}/querys.txt")
        """
        with open(f"{EXECUTION_PATH}/querys.txt", "r") as f:
            querys = [query.rstrip("\n") for query in f]
         """
        if self.gpu_available:
            querys.extend(self._file_to_list(f"{EXECUTION_PATH}/gpu_querys.txt"))

        self.log.debug_color(f"Querys read!")
        
        return querys

    def _collect_metrics(self, querys : list[str]):
        with open(f"{METRICS_PATH}/prometheus_metrics.csv", "a") as f:
            values = []
            values.append(str(time.time()))
            for query in querys:
                self.log.debug_color(f"Doing query: \[{query}]")
                data = str(self.prometheus.query(query))
                values.append(f"{data}")
                self.log.debug_color(f"Query done!")
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