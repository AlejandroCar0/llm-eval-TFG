import time
import platform
import os
from prometheus_client import start_http_server, Gauge

def is_jetson():
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
    start_http_server(9115)
    if is_jetson():
        print("Detected Jetson platform. Using jetson-stats.")
        run_jetson_exporter()
    else:
        print("Detected x86_64 or other platform. Using NVML.")
        run_nvml_exporter()