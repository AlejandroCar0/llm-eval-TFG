import sys
import pynvml

try:
    pynvml.nvmlInit()
    pynvml.nvmlDeviceGetCount()  # Lanza excepción si no hay GPU
    pynvml.nvmlShutdown()
    sys.exit(0)  # OK
except pynvml.NVMLError:
    sys.exit(1)  # No hay GPU o no están los drivers