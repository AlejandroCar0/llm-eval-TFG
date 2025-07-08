"""
Detector de GPU NVIDIA - Multiplataforma

Script de detección robusta de GPU NVIDIA que utiliza múltiples métodos
de verificación para maximizar compatibilidad across diferentes sistemas
y configuraciones de drivers.

Métodos de detección implementados:
1. nvidia-smi: Comando estándar disponible en la mayoría de instalaciones
2. NVML API: Acceso directo a drivers para verificación de bajo nivel  
3. tegrastats: Específico para plataformas Jetson/Tegra ARM64

El script utiliza patrón de verificación en cascada donde cualquier método
exitoso indica presencia de GPU. Diseñado para ser ejecutado remotamente
via SSH como parte del sistema de evaluación de LLMs.

Exit codes:
    0: GPU detectada por al menos uno de los métodos
    1: Ningún método detectó GPU disponible
"""

import sys
import pynvml
import subprocess

def check_nvidia_smi():
    """
    Verifica disponibilidad de GPU mediante el comando nvidia-smi.
    
    Ejecuta nvidia-smi para detectar presencia de drivers NVIDIA y GPU.
    Método estándar de detección compatible con la mayoría de instalaciones.
    
    Returns:
        int: 0 si nvidia-smi ejecuta correctamente, 1 en caso contrario
    """
    try:
        sub = subprocess.run(["nvidia-smi"])

        return sub.returncode

    except (subprocess.CalledProcessError, FileNotFoundError):

        return 1

def check_tegrastats():
    """
    Verifica disponibilidad de GPU en plataformas Jetson/Tegra mediante tegrastats.
    
    Específico para dispositivos NVIDIA Jetson que utilizan arquitectura ARM64.
    tegrastats es la herramienta estándar para monitoreo en estos sistemas.
    Maneja el caso especial donde tegrastats se ejecuta indefinidamente cuando
    encuentra hardware compatible.
    
    Returns:
        int: 0 si tegrastats inicia correctamente o timeout (indica GPU disponible),
             1 si falla la ejecución
    """
    try:
        subprocess.run(
            ["tegrastats"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,  # Lanza excepción si falla
            timeout=5
        )
    
        return 0
    
    except (subprocess.CalledProcessError, FileNotFoundError):
        
        return 1
    
    except(subprocess.TimeoutExpired):

        return 0 # We assume that the command has started and gpu is available

def check_nvml():
    """
    Verifica disponibilidad de GPU mediante la API NVML de bajo nivel.
    
    Utiliza Python NVML bindings para acceso directo a los drivers NVIDIA.
    Método más confiable que detecta específicamente la presencia de GPUs
    y drivers funcionales, no solo la instalación de herramientas.
    
    Returns:
        int: 0 si NVML detecta GPU disponible, 1 si no hay GPU o drivers
    """

    try:
        pynvml.nvmlInit()
        pynvml.nvmlDeviceGetCount()  # Lanza excepción si no hay GPU
        pynvml.nvmlShutdown()

        return 0 
    
    except pynvml.NVMLError:

        return 1  # No hay GPU o no están los drivers
    
def checker():
    """
    Orquesta la detección de GPU ejecutando múltiples métodos de verificación.
    
    Implementa patrón de verificación en cascada probando diferentes métodos
    de detección en orden de confiabilidad. Si cualquier método detecta GPU,
    termina con éxito. Solo falla si todos los métodos fallan.
    
    Esta función controla el exit code del script completo.
    
    Exit Codes:
        0: GPU detectada por al menos uno de los métodos
        1: Ningún método detectó GPU disponible
    """
    checks = [
        check_nvidia_smi,
        check_nvml,
        check_tegrastats
    ]

    for check in checks:
        if check() == 0:
            sys.exit(0)
        
    sys.exit(1)

checker()
