import click 
import re
import os
import paramiko
import traceback
import json
import logging
import paramiko.ssh_exception
from rich.logging import RichHandler
from ollama_handler import OllamaHandler
import requests
WORKING_PATH = f"llm-eval"
OLLAMA_PATH = f"{WORKING_PATH}/ollama"
EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
FORMAT = "%(message)s"
logging.basicConfig(
    level = "NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(tracebacks_suppress=[paramiko])]
)
log = logging.getLogger("rich")

def pull_models(models: list, handler: OllamaHandler) -> None:

    for model in models:
        log.info(f"[i green bold]Pulling model: {model}",extra = {"markup" : True})
        handler.pull_model(model)
        log.info(f"[i green bold]Model\[{model}] downloaded",extra = {"markup" : True})
    
def process_models(ip_address: str):
    handler = OllamaHandler(ip_address)
    models = read_models()

    #pull models
    pull_models(models, handler)
        
    #process prompts
    response = handler.multiple_prompts("orca-mini", [{"role" : "user", "content" : "Hello"}, {"role" : "user", "content" : "Who are you?"}]) # ver como funciona bien esta vaina
    response = handler.single_prompt("orca-mini","What number do you obtain from the sum off 2 and 3?")
    data = response.json()
    print(json.dumps(data, indent = 4))

    response = handler.single_prompt("orca-mini","Tell me the last number you obtain from the previous question I made to you?")
    data = response.json()
    print(json.dumps(data, indent = 4))
    handler.list_models()
    


def run_command(ssh: paramiko.SSHClient, command: str) -> tuple[paramiko.ChannelFile, paramiko.ChannelFile, paramiko.ChannelFile]:
    #TODO
    log.info(f"[i green bold] Running command: {command}", extra = {"markup" : True})
    stdin,stdout,stderr = ssh.exec_command(command)
    #wait for the end of the command on the remote machine
    stdout.channel.recv_exit_status()
    log.info(f"[i green bold] \[Command: {command}] Executed", extra = {"markup" : True})
    return (stdin,stdout,stderr)

def read_models() -> list:
    with open(f"{EXECUTION_PATH}/modelList.txt","r") as modelList:
        models = [line.rstrip("\n") for line in modelList]

    return models

def connection_establishment(user: str, password: str, ip_address: str, private_key: str) -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()

    log.info(f"[i blue] \[+] Connecting with the server... [/]", extra = {"markup" : True})
    ssh.connect(ip_address, username=user, password=password, key_filename=private_key)
    log.info(f"[i green u] \[+] Conexion established[/]", extra = {"markup" : True})

    return ssh

def environment_configuration(ssh: paramiko.SSHClient, password: str) -> None:
    log.info(f"[i blue] \[+] Setting the environment[/]", extra = {"markup" : True})

    #primero tenemos que definir la ruta donde vamos a trabajar en el server remoto en este caso va a ser ${HOME}/llm-eval/
    run_command(ssh, f"mkdir -p {WORKING_PATH}")

    #procedemos a copiar el archivo de configuracion para establecer las librerias y cosas necesarias en el servidor
    with ssh.open_sftp() as sftp:
        sftp.put(localpath=f"{EXECUTION_PATH}/configurations.sh", remotepath=f"{WORKING_PATH}/configurations.sh")
    
    #Ejecutamos el script configurations.sh en el servidor
    run_command(ssh, f"chmod 755 {WORKING_PATH}/configurations.sh")
    run_command(ssh, f"{WORKING_PATH}/configurations.sh {password}")

    log.info(f"[i green u]  \[+] Configured environment [/]", extra = {"markup" : True})

#Funcion llamada por el callback para validar/procesar la dir ip
def validarIp(ctx,param,valor: str) -> str:
    valor = valor.lower() # Parseamos el tipo de valo
    pattern = "(^((25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.){3}((25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?))$|localhost)"

    if not re.match(pattern,valor):
        raise click.BadParameter("El formato de la ip debe de ser el siguiente [0-255].[0-255].[0-255].[0-255] OR localhost")
    
    return valor

@click.command()
@click.option("--user", "-u", help="Name of the user to connect in target destination",default = "root")
@click.option("--password", "-p", help="Password of the user to connect in target destination", default = "")
@click.option("--ip-address", "-i", required=True, callback=validarIp, help="Ip-Address of the host where the test it's going to be executed")
@click.option("--private-key", "-pk", help="Path to private key in .pem format for ssh authentication", default=f"{os.getenv('HOME')}/.ssh/id_rsa.pub")
def procesarLLM(ip_address: str, private_key: str, user: str, password: str):
     #Creacion de un cliente ssh
    #Si no se especifica la siguiente linea no funciona nada
    #Cargar claves con ssh-keyscan
    try:
        ssh = connection_establishment(user, password, ip_address, private_key)
        environment_configuration(ssh, password)
        #-----Instalando LLMS-------
        run_command(ssh, f"OLLAMA_HOST={ip_address} {OLLAMA_PATH}/bin/ollama serve >/dev/null 2>&1 &") # poner el OLLAMA_HOST
        #Usar api de ollama para el texto
        process_models(ip_address)

    except (paramiko.AuthenticationException, paramiko.BadHostKeyException, paramiko.SSHException) as e:
        log.exception(f"[bold red] \[!] ERROR: {e}[/]", extra = {"markup" : True})

    except Exception as e:
        log.exception(f"[bold red] \[!] ERROR: {e}[/]", extra = {"markup" : True})
    
    finally:
        ssh.close()


if __name__ == '__main__':
    procesarLLM()