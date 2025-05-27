import logging
from rich.logging import RichHandler
FORMAT = "%(message)s"
logging.basicConfig(
    level = "NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

class Log():
    def __init__(self):
        self.log = logging.getLogger("rich")
    
    def info(self, msg: str):
        self.log.info(f"[i green bold]{msg}",extra = {"markup" : True})
    
    def warning(self, msg: str):
        self.log.warning(f"[i yellow bold]{msg}", extra= {"markup" : True})
    
    def exception(self, msg: str):
        self.log.exception(f"[bold red] \[!] {msg}", extra= {"markup" : True})
    