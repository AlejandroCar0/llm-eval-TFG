import logging
import logging.handlers
from rich.logging import RichHandler
import os

EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))

class Log(logging.Logger):
    
    def info_color(self, msg: str, *args, **kwargs):
        super().info(f"[i blue bold]{msg}",extra = {"markup" : True}, stacklevel=2)
    
    def warning_color(self, msg: str, *args, **kwargs):
        super().warning(f"[i yellow bold]{msg}", extra= {"markup" : True}, stacklevel=2)
    
    def exception_color(self, msg: str, *args, **kwargs):
        super().exception(f"[bold red] \[!] {msg}", extra= {"markup" : True}, stacklevel=2)
    def debug_color(self, msg: str, *args, **kwargs): 
            super().debug(f"[i bold green]{msg}", extra= {"markup" : True}, stacklevel=2)


file_handler = logging.FileHandler(f"{EXECUTION_PATH}/logs.txt", mode = "w", encoding= "utf-8")
FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level = "NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(), file_handler]
)
logging.setLoggerClass(Log)
logger = logging.getLogger("llm-eval")