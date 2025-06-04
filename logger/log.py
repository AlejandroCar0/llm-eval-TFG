import logging
from rich.logging import RichHandler

class Log(logging.Logger):
    
    def info_color(self, msg: str, *args, **kwargs):
        super().info(f"[i blue bold]{msg}",extra = {"markup" : True}, stacklevel=2)
    
    def warning_color(self, msg: str, *args, **kwargs):
        super().warning(f"[i yellow bold]{msg}", extra= {"markup" : True}, stacklevel=2)
    
    def exception_color(self, msg: str, *args, **kwargs):
        super().exception(f"[bold red] \[!] {msg}", extra= {"markup" : True}, stacklevel=2)
    def debug_color(self, msg: str, *args, **kwargs): 
            super().debug(f"[i bold green]{msg}", extra= {"markup" : True}, stacklevel=2)


FORMAT = "%(asctime)s - %(message)s"
logging.basicConfig(
    level = "NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
logging.setLoggerClass(Log)
logger = logging.getLogger("llm-eval")