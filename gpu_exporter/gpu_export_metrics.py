"""
GPU Exporter para Prometheus - Multiplataforma

Exportador personalizado que proporciona métricas detalladas de GPU NVIDIA
para el ecosistema Prometheus. Soporta tanto arquitecturas x86_64 como ARM64 (Jetson)
mediante detección automática de plataforma y APIs específicas.

Características principales:
- Detección automática de hardware (Jetson vs x86_64)
- Métricas unificadas independientes de la plataforma
- Servidor HTTP integrado en puerto 9115
- Compatibilidad completa con Prometheus
- Proceso daemon identificable

Métricas exportadas:
- gpu_utilization: Porcentaje de uso de GPU
- gpu_memory_used_bytes: Memoria GPU utilizada en bytes
- gpu_memory_total_bytes: Memoria GPU total en bytes  
- gpu_power_usage_mw: Consumo de potencia en milliwatts
- gpu_temperature_celsius: Temperatura en grados Celsius
- gpu_encoder_utilization: Utilización del encoder de video
- gpu_decoder_utilization: Utilización del decoder de video
"""

import time
import platform
import os
from prometheus_client import start_http_server, Gauge
import setproctitle

setproctitle.setproctitle("gpu_exporter")

def is_jetson():
    """
    Detecta si el sistema actual es una plataforma NVIDIA Jetson.
    
    Utiliza dos métodos de detección:
    1. Verificación de archivo específico del sistema Tegra
    2. Detección de arquitectura ARM64 que es característica de Jetson
    
    Returns:
        bool: True si es plataforma Jetson, False para otras plataformas
    """
    return os.path.isfile("/etc/nv_tegra_release") or 'aarch64' in platform.machine()

# Define unified metrics
gpu_utilization = Gauge('gpu_utilization', 'GPU usage percentage')
gpu_memory_used = Gauge('gpu_memory_used_bytes', 'Used GPU memory in bytes')
gpu_memory_total = Gauge('gpu_memory_total_bytes', 'Total GPU memory in bytes')
gpu_power_usage = Gauge('gpu_power_usage_mw', 'GPU power usage in milliwatts')
gpu_temperature = Gauge('gpu_temperature_celsius', 'GPU temperature in Celsius')
gpu_encoder_util = Gauge('gpu_encoder_utilization', 'GPU encoder utilization percentage')
gpu_decoder_util = Gauge('gpu_decoder_utilization', 'GPU decoder utilization percentage')

def run_jetson_exporter():
    """
    Ejecuta el exportador de métricas específico para plataformas Jetson.
    
    Utiliza la librería jetson-stats (jtop) para acceder a métricas del sistema
    Tegra. Maneja las diferencias arquitectónicas de Jetson como memoria
    unificada y APIs específicas del hardware.
    
    Convierte unidades cuando es necesario para mantener consistencia:
    - Potencia: Watts a milliwatts
    - Memoria: MB a bytes
    
    Raises:
        ImportError: Si jetson-stats no está instalado
    """
    try:
        from jtop import jtop
    except ImportError:
        print("Please install jetson-stats: pip install jetson-stats")
        return

    with jtop() as jetson:
        while jetson.ok():
            stats = jetson.stats
            gpu_utilization.set(stats.get('GPU', 0.0))
            gpu_memory_used.set(stats.get('RAM_USED', 0.0) * 1024 * 1024)
            gpu_memory_total.set(stats.get('RAM', 0.0) * 1024 * 1024)
            gpu_power_usage.set(stats.get('GPU_POWER', 0.0) * 1000)  # W to mW
            gpu_temperature.set(stats.get('GPU_TEMP', 0.0))
            time.sleep(1)

def run_nvml_exporter():
    """
    Ejecuta el exportador de métricas utilizando la API NVML para GPUs estándar.
    
    Utiliza NVIDIA Management Library para acceso directo a métricas de GPU
    en sistemas x86_64 y otras arquitecturas que no sean Jetson. Proporciona
    acceso de bajo nivel a todas las capacidades de monitoreo de la GPU.
    
    Métricas capturadas:
    - Utilización de GPU y memoria
    - Información de memoria (usada/total)
    - Consumo de potencia en milliwatts
    - Temperatura del núcleo GPU
    - Utilización de encoder/decoder de video
    
    Implementa cleanup garantizado con finally para liberar recursos NVML.
    
    Raises:
        ImportError: Si pynvml no está instalado
    """
    try:
        from pynvml import (
            nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates,
            nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature,
            nvmlDeviceGetEncoderUtilization, nvmlDeviceGetDecoderUtilization,
            nvmlShutdown, NVML_TEMPERATURE_GPU
        )
    except ImportError:
        print("Please install pynvml: pip install nvidia-ml-py3")
        return

    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    try:
        while True:
            util = nvmlDeviceGetUtilizationRates(h)
            mem = nvmlDeviceGetMemoryInfo(h)
            power = nvmlDeviceGetPowerUsage(h)
            temp = nvmlDeviceGetTemperature(h, NVML_TEMPERATURE_GPU)
            enc_util = nvmlDeviceGetEncoderUtilization(h)
            dec_util = nvmlDeviceGetDecoderUtilization(h)

            gpu_utilization.set(util.gpu)
            gpu_memory_used.set(mem.used)
            gpu_memory_total.set(mem.total)
            gpu_power_usage.set(power)
            gpu_temperature.set(temp)
            gpu_encoder_util.set(enc_util[0])
            gpu_decoder_util.set(dec_util[0])

            time.sleep(1)
    finally:
        nvmlShutdown()

if __name__ == "__main__":
    """
    Punto de entrada principal del GPU exporter.
    
    Inicia servidor HTTP de Prometheus en puerto 9115 y selecciona
    automáticamente el exportador apropiado según la plataforma detectada:
    - Jetson: Utiliza jetson-stats para acceso optimizado a Tegra
    - Otras: Utiliza NVML para acceso estándar a GPU NVIDIA
    
    El proceso se ejecuta como daemon con nombre identificable para
    facilitar gestión desde scripts externos.
    """
    start_http_server(9115)
    if is_jetson():
        print("Detected Jetson platform. Using jetson-stats.")
        run_jetson_exporter()
    else:
        print("Detected x86_64 or other platform. Using NVML.")
        run_nvml_exporter()