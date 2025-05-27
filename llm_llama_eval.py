import click 
import re
import os
import paramiko
from log import Log
from ollama import Ollama
from prometheus import Prometheus
WORKING_PATH = f"llm-eval"
OLLAMA_PATH = f"{WORKING_PATH}/ollama"
EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
log = Log()
#FORMAT = "%(message)s"
#logging.basicConfig(
#    level = "NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(tracebacks_suppress=[paramiko])]
#)

def pull_models(models: list, handler: Ollama) -> None:

    for model in models:
        log.info(f"Pulling model: {model}")
        handler.pull_model(model)
        log.info(f"Model\[{model}] downloaded")

def read_models() -> list:
    with open(f"{EXECUTION_PATH}/modelList.txt","r") as modelList:
        models = [line.rstrip("\n") for line in modelList]

    return models

def read_prompts() -> list:
    prompts = []
    with open(f"{EXECUTION_PATH}/prompts.txt","r") as promptsList:
        for line in promptsList:
            line = line.rsplit("\;")
            prompts.append(line)

    return prompts

def process_prompt(prompts: list, model: str, handler: Ollama) -> None:
    if len(prompts) == 1:
        response =  handler.single_prompt(model, prompts[0])
    else:
        messages = []
        for prompt in prompts:
            prompt = {"role" : "user", "content" : f"{prompt}"}
            messages.append(prompt)
            response = handler.chat_prompt(model,messages)
            data = response.json()
            messages.append(data.get("message"))
            #Subdividir todo esto en funciones para dejarlo clean
        print(messages)

def process_models(ip_address: str):
    handler = Ollama(ip_address)
    models = read_models()

    #pull models
    pull_models(models, handler)
        
    #process prompts
    prompts = read_prompts()
    for prompt in prompts:
        process_prompt(prompt, "phi3", handler)
    handler.list_models()
    

def run_command(ssh: paramiko.SSHClient, command: str) -> tuple[paramiko.ChannelFile, paramiko.ChannelFile, paramiko.ChannelFile]:
    #TODO
    log.info(f"Running command: {command}")
    stdin,stdout,stderr = ssh.exec_command(command)
    #wait for the end of the command on the remote machine
    stdout.channel.recv_exit_status()
    log.info(f"\[Command: {command}] Executed")

    return (stdin,stdout,stderr)

def connection_establishment(user: str, password: str, ip_address: str, private_key_file: str) -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    #If is the first time connecting to a new server, will automatically save the public key into the .ssh/known_hosts file
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_system_host_keys()
    try:
        log.info(f"\[+] Connecting with the server... [/]")

        pkey = paramiko.RSAKey.from_private_key_file(private_key_file)
        ssh.connect(ip_address, username = user, pkey = pkey)
        log.info(f"\[+] connection established[/]")
    except paramiko.AuthenticationException as e:
        log.warning(f"\[!] connection failed with key located in {private_key_file}")
        log.warning(f"\[+] Retrying with the password provided")
        ssh.connect(ip_address, username = user, password = password)
        log.info(f"\[+] connection established[/]")

    return ssh

def environment_configuration(ssh: paramiko.SSHClient, password: str, ollama_version: str, node_version: str) -> None:
    log.info(f"\[+] Setting the environment[/]")

    #primero tenemos que definir la ruta donde vamos a trabajar en el server remoto en este caso va a ser ${HOME}/llm-eval/
    run_command(ssh, f"mkdir -p {WORKING_PATH}")

    #procedemos a copiar el archivo de configuracion para establecer las librerias y cosas necesarias en el servidor
    with ssh.open_sftp() as sftp:
        sftp.put(localpath=f"{EXECUTION_PATH}/configurations.sh", remotepath=f"{WORKING_PATH}/configurations.sh")
    
    #Ejecutamos el script configurations.sh en el servidor
    run_command(ssh, f"chmod 755 {WORKING_PATH}/configurations.sh")
    run_command(ssh, f"{WORKING_PATH}/configurations.sh {password} {ollama_version} {node_version}")

    log.info(f"\[+] Configured environment [/]")

#Funcion llamada por el callback para validar/procesar la dir ip
def validarIp(ctx,param,valor: str) -> str:
    valor = valor.lower() # Parseamos el tipo de valo
    pattern = "(^((25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.){3}((25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?))$|localhost)"

    if not re.match(pattern,valor):
        raise click.BadParameter("El formato de la ip debe de ser el siguiente [0-255].[0-255].[0-255].[0-255] OR localhost")
    
    return valor

def validate_ollama_version(ctx,param,valor: str) -> str:
    ollama_versions = ''
    with open(f"{EXECUTION_PATH}/versions/ollama_versions.txt","r") as versions:
        ollama_versions = [version.rstrip(f"\n") for version in versions]

    if valor not in ollama_versions:
        raise click.BadParameter(f"Error, version should be one of the followings: {ollama_versions}")
    
    return valor

def validate_node_exporter_version(ctx,param,valor: str) -> str:
    node_exporter_versions = ''
    with open(f"{EXECUTION_PATH}/versions/node_exporter_versions.txt","r") as versions:
        node_exporter_versions = [version.rstrip("\n") for version in versions]
    if valor not in node_exporter_versions:
        raise click.BadParameter(f"Error, version should be one of the followings: {node_exporter_versions}")
    return valor

@click.command()
@click.option("--user", "-u", help="Name of the user to connect in target destination",default = "root")
@click.option("--password", "-p", help="Password of the user to connect in target destination", default = "")
@click.option("--ip-address", "-i", required=True, callback=validarIp, help="Ip-Address of the host where the test it's going to be executed")
@click.option("--ollama-version", "-ov", callback=validate_ollama_version, help="ollama version to install in the SUT you must put the \"vx.x.x\"", default = "v0.7.0")
@click.option("--node-version", "-nv", callback=validate_node_exporter_version, help="node_exporter version to install in the SUT you must put the \"vx.x.x\"", default = "v1.9.1")
@click.option("--private-key", "-pk", help="Path to private key in .pem format for ssh authentication", default=f"{os.getenv('HOME')}/.ssh/id_rsa")
def procesarLLM(ip_address: str, private_key: str, user: str, password: str, ollama_version: str, node_version: str):
     #Creacion de un cliente ssh
    #Si no se especifica la siguiente linea no funciona nada
    #Cargar claves con ssh-keyscan
    try:
        ssh = connection_establishment(user, password, ip_address, private_key)
        environment_configuration(ssh, password, ollama_version, node_version)
        #----Iniciando prometheus---
        #crear mi objeto de prometheus
        #prometheus = Prometheus(remote_ip_address = ip_address)
        #iniciar prometheus en el SUT
        #Iniciar prometheus en mi maquina local
        #-----Instalando LLMS-------
        #run_command(ssh, f"OLLAMA_HOST={ip_address} {OLLAMA_PATH}/bin/ollama serve >/dev/null 2>&1 &") # poner el OLLAMA_HOST
        #Usar api de ollama para el texto
        #process_models(ip_address)

    except (paramiko.AuthenticationException, paramiko.BadHostKeyException, paramiko.SSHException) as e:
        log.exception(f"ERROR: {e}[/]")

    except Exception as e:
        log.exception(f"ERROR: {e}[/]")
    
    finally:
        ssh.close()


if __name__ == '__main__':
    procesarLLM()