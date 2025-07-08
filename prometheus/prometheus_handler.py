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
    """
    Handler principal para la gestión completa del sistema de monitorización con Prometheus.
    
    Orquesta el ciclo de vida completo del monitoreo: inicialización del servidor Prometheus,
    configuración de targets remotos, recolección continua de métricas en background,
    y persistencia de datos en formato CSV con sincronización temporal.
    
    Attributes:
        remote_ip_address (str): IP del sistema bajo prueba
        gpu_available (bool): Indica si hay GPU disponible para monitoreo
        export_metrics (str): Header CSV con las métricas a exportar
        prometheus (Prometheus): Instancia de la clase base para consultas
        stop_signal (threading.Event): Señal para detener la recolección
        collector (threading.Thread): Thread daemon para recolección continua
    """

    def __init__(self, remote_ip_address: str, gpu_available: bool):
        """
        Inicializa el handler de Prometheus con configuración adaptativa.
        
        Configura automáticamente las métricas según la disponibilidad de GPU,
        inicia el servidor Prometheus, prepara archivos de salida y configura
        el thread de recolección en modo daemon.
        
        Args:
            remote_ip_address (str): Dirección IP del sistema bajo prueba
            gpu_available (bool): True si hay GPU NVIDIA disponible
        """
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
        """
        Inicia el servidor Prometheus como proceso independiente.
        
        Lanza Prometheus con la configuración generada dinámicamente y
        implementa un mecanismo de health check con timeout para verificar
        que el servicio esté completamente operativo antes de continuar.
        
        Raises:
            Exception: Si Prometheus no se inicia correctamente en el tiempo límite
        """
        self.log.debug_color("Starting prometheus...")
        self.prometheus = Prometheus(self.remote_ip_address)

        process = subprocess.Popen(
            ["prometheus",
             f'--config.file={EXECUTION_PATH}/prometheus.yml'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        try:
            self._wait_for_prometheus(60)
            self.log.debug_color("Prometheus started!")

        except Exception:
            raise

    def _wait_for_prometheus(self, timeout: int = 60):
        """
        Implementa health check con polling para verificar disponibilidad de Prometheus.
        
        Realiza polling continuo al endpoint de readiness de Prometheus hasta que
        responda correctamente o se alcance el timeout. Incluye un período de
        gracia adicional para garantizar estabilidad del servicio.
        
        Args:
            timeout (int, optional): Tiempo máximo de espera en segundos. Por defecto 60
            
        Raises:
            Exception: Si Prometheus no está listo en el tiempo especificado
        """
        url = 'http://127.0.0.1:9090/-/ready'
        for i in range(timeout):
            try:
                response = requests.get(url, timeout=1)

                if response.status_code == 200:
                    time.sleep(5)
                    break
            
            except requests.RequestException:
                pass

            time.sleep(1)

        else:
            raise Exception(f"Faile to get up Prometheus in {timeout} seconds")

    def _file_to_list(self, path: str) -> list:
        """
        Utility para leer archivos de texto y convertirlos en lista.
        
        Args:
            path (str): Ruta del archivo a leer
            
        Returns:
            list: Lista con las líneas del archivo sin caracteres de nueva línea
        """
        result = []
        with open(f"{path}", "r") as f:
            result = [line.rstrip("\n") for line in f]

        return result

    def _read_querys(self):
        """
        Carga las consultas PromQL desde archivos de configuración.
        
        Lee las consultas base del sistema desde querys.txt y añade
        las consultas específicas de GPU si hay hardware disponible.
        Esta carga diferida permite configuración adaptativa según el hardware.
        
        Returns:
            list: Lista de consultas PromQL a ejecutar
        """
        querys = []

        self.log.debug_color("Reading querys....")

        querys = self._file_to_list(f"{EXECUTION_PATH}/querys.txt")
        if self.gpu_available:
            querys.extend(self._file_to_list(f"{EXECUTION_PATH}/gpu_querys.txt"))

        self.log.debug_color("Querys read!")
        
        return querys

    def _collect_metrics(self, querys : list[str]):
        """
        Ejecuta todas las consultas y persiste los resultados en CSV.
        
        Itera sobre todas las consultas PromQL, ejecuta cada una con manejo
        robusto de errores, y escribe los resultados en formato CSV con
        timestamp sincronizado para correlación temporal.
        
        Args:
            querys (list[str]): Lista de consultas PromQL a ejecutar
        """
        with open(f"{METRICS_PATH}/prometheus_metrics.csv", "a") as f:
            values = []
            values.append(str(time.time()))
            for query in querys:
                try:
                    self.log.debug_color(f"Doing query: \[{query}]")
                    data = str(self.prometheus.query(query))
                    values.append(f"{data}")
                    self.log.debug_color("Query done!")
                except Exception as e:
                    self.log.warning_color(f'Error in query: {query}: {e}')
                    values.append("Error")

            f.write(";".join(values) + "\n")
    
    def _process_metrics(self):
        """
        Bucle principal de recolección continua ejecutado en thread daemon.
        
        Ejecuta recolección de métricas cada 2 segundos hasta recibir señal
        de parada. Mantiene sincronización temporal constante para correlación
        con eventos de evaluación de LLMs.
        """
        querys = self._read_querys()

        while not self.stop_signal.is_set():
            self._collect_metrics(querys)
            time.sleep(2)
    
    def start_collection(self):
        """
        Inicia la recolección de métricas en background.
        
        Lanza el thread de recolección que ejecutará continuamente
        hasta recibir señal de parada.
        """
        self.log.debug_color("Starting metrics collection")
        self.collector.start()
    
    def stop_collection(self):
        """
        Detiene la recolección de métricas de forma ordenada.
        
        Envía señal de parada al thread de recolección y espera
        su finalización completa antes de continuar. Garantiza
        cleanup limpio de recursos.
        """
        self.log.debug_color("Stopping metrics collection")

        self.stop_signal.set()
        self.collector.join()

        self.log.debug_color("Metrics collection stopped")