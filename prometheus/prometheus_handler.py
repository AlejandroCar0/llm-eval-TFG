import threading
from prometheus.prometheus import Prometheus
import threading
import os
import time
from logger.log import logger

EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = f"{EXECUTION_PATH}/../metrics"
EXPORT_METRICS = "cpu;memory;gpu_utilization;gpu_memory_used;gpu_memory_total;gpu_power_usage;gpu_temperature;gpu_encoder_util;gpu_decoder_util\n"
class PrometheusHandler():

    def __init__(self, remote_ip_address: str):
        self.log = logger.getChild(__file__)
        self.log.debug_color(f"Starting prometheus...")

        self.prometheus = Prometheus(remote_ip_address)
        time.sleep(10)

        self.log.debug_color(f"Prometheus started!")

        self.stop_signal = threading.Event()
        self.collector = threading.Thread(target = self._process_metrics)
        os.system(f"mkdir -p {METRICS_PATH}")
        with open(f"{METRICS_PATH}/prometheus_metrics.txt", "w") as f:
            f.write(f"{EXPORT_METRICS}")

    def _read_querys(self):
        querys = []

        self.log.debug_color("Reading querys....")

        with open(f"{EXECUTION_PATH}/querys.txt", "r") as f:
            querys = [query.rstrip("\n") for query in f]

        self.log.debug_color("Querys read!")
        
        return querys

    def _collect_metrics(self, querys : list[str]):
        with open(f"{METRICS_PATH}/prometheus_metrics.txt", "a") as f:

            for query in querys:
                self.log.debug_color(f"Doing query: \[{query}]")
                data = self.prometheus.query(query)
                f.write(f"{data};")
                self.log.debug_color(f"Query done!")

            f.write("\n")
    
    def _process_metrics(self):
        querys = self.read_querys()

        while not self.stop_signal.is_set():
            self.collect_metrics(querys)
            time.sleep(2)
    
    def start_collection(self):
        self.log.debug_color("Starting metrics collection")
        self.collector.start()
    
    def stop_collection(self):
        self.log.debug_color("Stopping metrics collection")

        self.stop_signal.set()
        self.collector.join()

        self.log.debug_color("Metrics collection stopped")