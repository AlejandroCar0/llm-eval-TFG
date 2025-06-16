import click 
import re
import os
import paramiko
from logger.log import logger
from ollama.ollama_handler import OllamaHandler
from prometheus.prometheus import Prometheus
from prometheus.prometheus_handler import PrometheusHandler
import threading
WORKING_PATH = f"llm-eval"
OLLAMA_PATH = f"{WORKING_PATH}/ollama"
EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))


    

def run_command(ssh: paramiko.SSHClient, command: str) -> tuple[paramiko.ChannelFile, paramiko.ChannelFile, paramiko.ChannelFile]:
    #TODO
    logger.debug_color(f"Running command: {command}")
    stdin,stdout,stderr = ssh.exec_command(command, timeout=180)
    #wait for the end of the command on the remote machine
    status_code = stdout.channel.recv_exit_status()
    logger.debug_color(f"\[Command: {command}] Executed with status code = {status_code}")

    return (stdin,stdout,stderr)

def connection_establishment(user: str, password: str, ip_address: str, private_key_file: str) -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    #If is the first time connecting to a new server, will automatically save the public key into the .ssh/known_hosts file
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_system_host_keys()
    try:
        logger.debug_color(f"\[+] Connecting with the server... [/]")

        pkey = paramiko.RSAKey.from_private_key_file(private_key_file)
        ssh.connect(ip_address, username = user, pkey = pkey)
        logger.debug_color(f"\[+] connection established[/]")
    except paramiko.AuthenticationException as e:
        logger.warning(f"\[!] connection failed with key located in {private_key_file}")
        logger.warning(f"\[+] Retrying with the password provided")
        ssh.connect(ip_address, username = user, password = password)
        logger.debug_color(f"\[+] connection established[/]")

    return ssh

def environment_configuration(ssh: paramiko.SSHClient, password: str, ollama_version: str, node_version: str) -> None:
    logger.debug_color(f"\[+] Setting the environment[/]")

    #primero tenemos que definir la ruta donde vamos a trabajar en el server remoto en este caso va a ser ${HOME}/llm-eval/
    run_command(ssh, f"mkdir -p {WORKING_PATH}/gpu_exporter")

    #procedemos a copiar el archivo de configuracion para establecer las librerias y cosas necesarias en el servidor
    with ssh.open_sftp() as sftp:
        sftp.put(localpath=f"{EXECUTION_PATH}/configurations.sh", remotepath=f"{WORKING_PATH}/configurations.sh")
        sftp.put(localpath=f"{EXECUTION_PATH}/gpu_exporter/requirements.txt", remotepath=f"{WORKING_PATH}/gpu_exporter/requirements.txt")
        sftp.put(localpath=f"{EXECUTION_PATH}/gpu_exporter/gpu_export_metrics.py", remotepath=f"{WORKING_PATH}/gpu_exporter/gpu_export_metrics.py")
    #Ejecutamos el script configurations.sh en el servidor
    run_command(ssh, f"chmod 755 {WORKING_PATH}/configurations.sh")
    run_command(ssh, f"{WORKING_PATH}/configurations.sh {password} {ollama_version} {node_version}")

    #arrancamos el script de exportacion
    run_command(ssh, f"chmod 777 {WORKING_PATH}/gpu_exporter/*")
    run_command(ssh,f"python3 -m venv {WORKING_PATH}/venv")
    run_command(ssh,f"{WORKING_PATH}/venv/bin/pip3 install -r {WORKING_PATH}/gpu_exporter/requirements.txt")
    run_command(ssh, f"{WORKING_PATH}/venv/bin/python3 {WORKING_PATH}/gpu_exporter/gpu_export_metrics.py >> pepe.txt 2>&1 &")

    logger.debug_color(f"\[+] Configured environment [/]")

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
stop = threading.Event()
def collect_prometheus_metrics(prometheus):
    with open(f"{EXECUTION_PATH}/metrics/prometheus_metrics.txt","a") as f:
        while not stop.is_set():
            with open(f"{EXECUTION_PATH}/prometheus/querys.txt","r") as f2:
                querys = [query.rstrip("\n") for query in f2]
                results = []
                data = ""
                for query in querys:
                    result = prometheus.query(query)
                    results.append(result)
                    data += f"{result};"
                data +="\n"
                f.write(data)
                #memory = prometheus.query("100 * (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))")
                #cpu = prometheus.query("100 - (avg(rate(node_cpu_seconds_total{mode=\"idle\"}[1m])) * 100)")
                #pepito = prometheus.query("gpu_power_usage_mw{job=\"gpu_exporter\"}")
                #data = f"{memory};{cpu};{pepito}\n"
                #f.write(data)   
                os.system("sleep 2")


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
        run_command(ssh, f"OLLAMA_HOST={ip_address} {OLLAMA_PATH}/bin/ollama serve > /dev/null 2>&1 &") # poner el OLLAMA_HOST
        prometheus = PrometheusHandler(ip_address)
        os.system("sleep 20")
        #-----Instalando LLMS-------
        ollama = OllamaHandler(ip_address)
        prometheus.start_collection()
        #Usar api de ollama para el texto
        ollama.process_models()
        prometheus.stop_collection()
       
    except (paramiko.AuthenticationException, paramiko.BadHostKeyException, paramiko.SSHException) as e:
        logger.exception_color(e)

    except Exception as e:
        logger.exception_color(e)
        prometheus.stop_collection()
    
    finally:
        ssh.close()


if __name__ == '__main__':
    procesarLLM()