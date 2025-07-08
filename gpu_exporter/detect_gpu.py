import sys
import pynvml
import subprocess

def check_nvidia_smi():
    try:
        sub = subprocess.run(["nvidia-smi"])

        return sub.returncode

    except (subprocess.CalledProcessError, FileNotFoundError):

        return 1

def check_tegrastats():
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

    try:
        pynvml.nvmlInit()
        pynvml.nvmlDeviceGetCount()  # Lanza excepción si no hay GPU
        pynvml.nvmlShutdown()

        return 0 
    
    except pynvml.NVMLError:

        return 1  # No hay GPU o no están los drivers
    
def checker():
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
