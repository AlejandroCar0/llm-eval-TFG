import os
import requests
from logger.log import logger
EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))

class Prometheus():
    """
    Clase base para la comunicación con la API de Prometheus.
    
    Gestiona la configuración dinámica del servidor Prometheus y proporciona
    una interfaz simplificada para realizar consultas PromQL contra métricas
    de sistema y GPU recolectadas desde exporters remotos.
    
    Attributes:
        remote_ip_address (str): Dirección IP del sistema bajo prueba
        node_exporter_port (str): Puerto del node_exporter (por defecto 9100)
        gpu_exporter_port (str): Puerto del gpu_exporter (por defecto 9115)
        time_interval (str): Intervalo de scraping en segundos
        url (str): URL de la API de consultas de Prometheus
        configuration_file (str): Configuración YAML generada dinámicamente
    """
    def __init__(self, remote_ip_address: str, node_exporter_port: str = "9100",gpu_exporter_port: str = 9115, time_interval = "1"):
        """
        Inicializa la instancia de Prometheus con configuración dinámica.
        
        Genera automáticamente el archivo de configuración YAML para Prometheus
        con los targets especificados para node_exporter y gpu_exporter.
        
        Args:
            remote_ip_address (str): IP del sistema bajo prueba donde están los exporters
            node_exporter_port (str, optional): Puerto del node_exporter. Por defecto "9100"
            gpu_exporter_port (str, optional): Puerto del gpu_exporter. Por defecto 9115
            time_interval (str, optional): Intervalo de scraping. Por defecto "1"
        """
        self.log = logger.getChild(__file__)
        self.remote_ip_address = remote_ip_address
        self.node_exporter_port = node_exporter_port
        self.gpu_exporter_port = gpu_exporter_port
        self.time_interval = time_interval
        self.url = "http://127.0.0.1:9090/api/v1/query"
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
        """
        Representación en string de la configuración de Prometheus.
        
        Returns:
            str: Información detallada de la configuración incluyendo
                 host remoto, intervalo de tiempo y configuración completa
        """
        to_string = f"Prometheus configuration:\n"\
            f"RemoteHost: [{self.remote_ip_address}:{self.node_exporter_port}]\n"\
            f"TimeInterval: {self.time_interval}\n"\
            f"Configuration file:\n\n{self.configuration_file}"
        return to_string
    
    def query(self, query: str):
        """
        Ejecuta una consulta PromQL contra la API de Prometheus.
        
        Realiza validación estricta de la respuesta HTTP y estructura de datos
        para garantizar que se obtenga un valor válido de la métrica consultada.
        
        Args:
            query (str): Consulta PromQL a ejecutar
            
        Returns:
            str: Valor de la métrica consultada
            
        Raises:
            Exception: Si hay errores HTTP, datos vacíos o estructura inválida
        """
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