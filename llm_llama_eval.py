import click 
import re
import os
import paramiko
import time
import datetime
import shutil
import platform
import paramiko.ssh_exception
from logger.log import logger
from ollama.ollama_handler import OllamaHandler
from prometheus.prometheus import Prometheus
from prometheus.prometheus_handler import PrometheusHandler
import threading
WORKING_PATH = f"llm-eval"
OLLAMA_PATH = f"{WORKING_PATH}/ollama"
EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
START_TIME = datetime.datetime.fromtimestamp(time.time())
    

def run_command(ssh: paramiko.SSHClient, command: str) -> tuple[paramiko.ChannelFile, paramiko.ChannelFile, paramiko.ChannelFile]:
    #TODO
    logger.debug_color(f"Running command: {command}")
    stdin,stdout,stderr = ssh.exec_command(command, timeout=180)
    #wait for the end of the command on the remote machine
    status_code = stdout.channel.recv_exit_status()
    logger.debug_color(f"\[Command: {command}] Executed with status code = {status_code}")

    return (stdin,stdout,stderr)

def connect_with_private_key(ssh: paramiko.SSHClient, user: str, ip_address: str, private_key_file: str) -> None:
    try:
        logger.debug_color(f"Connecting with the server using file: {private_key_file}")
        pkey = paramiko.RSAKey.from_private_key_file(private_key_file)
        ssh.connect(ip_address, username = user, pkey = pkey)

    except paramiko.AuthenticationException as e:
        logger.warning_color(f'Error on the authentication using private key: {e}')
        raise
    
    except paramiko.SSHException as e:
        logger.warning_color(f'Error in the ssh protocol using private key: {e}')
        raise
    
    except Exception as e:
        logger.warning_color(f'Error when trying to connect using private key: {e}')
        raise


def connect_with_password(ssh: paramiko.SSHClient, user: str, password: str, ip_address: str):
    try:
        logger.debug_color(f'Connecting with the server using password ****')
        ssh.connect(ip_address, username = user, password = password)

    except paramiko.AuthenticationException as e:
        logger.warning_color(f'Error on the authentication using password: {e}')
        raise

    except paramiko.SSHException as e:
        logger.warning_color(f'Error in the ssh protocol using password: {e}')
        raise
    
    except Exception as e:
        logger.warning_color(f'Error when trying to connect using password: {e}')
        raise 
    
def connection_establishment(user: str, password: str, ip_address: str, private_key_file: str) -> paramiko.SSHClient:
    logger.debug_color("Starting connection....")

    ssh = paramiko.SSHClient()
    #If is the first time connecting to a new server, will automatically save the public key into the .ssh/known_hosts file
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_system_host_keys()

    try:
        connect_with_private_key(ssh, user = user, ip_address = ip_address, private_key_file = private_key_file)

    except (paramiko.AuthenticationException, paramiko.SSHException, Exception):
        logger.warning_color(f'Retrying...')

        try:
            if not password:
                raise Exception(f"Impossible to connect with SSH to {ip_address}")
            
            connect_with_password(ssh = ssh, user = user, password = password, ip_address = ip_address)
            

        except (paramiko.AuthenticationException, paramiko.SSHException, Exception):
            raise Exception(f"Impossible to connect with SSH to {ip_address}")
    
    logger.debug_color("Connection established!")

    return ssh

def is_gpu_available(ssh):

    logger.info_color(f"Checking if GPU is available")

    _, stdout, _ = run_command(ssh, f"{WORKING_PATH}/venv/bin/python3 {WORKING_PATH}/gpu_exporter/detect_gpu.py")

    if stdout.channel.recv_exit_status() == 0:
        logger.info_color(f"GPU is available")
        return True
    
    logger.warning_color("No GPU detected")
    return False


def environment_configuration(ssh: paramiko.SSHClient, password: str, ollama_version: str, node_version: str, reinstall_ollama: bool) -> None:
    logger.debug_color(f"\[+] Setting the environment[/]")

    #primero tenemos que definir la ruta donde vamos a trabajar en el server remoto en este caso va a ser ${HOME}/llm-eval/
    run_command(ssh, f"mkdir -p {WORKING_PATH}/gpu_exporter")

    #procedemos a copiar el archivo de configuracion para establecer las librerias y cosas necesarias en el servidor
    with ssh.open_sftp() as sftp:
        sftp.put(localpath=f"{EXECUTION_PATH}/configurations.sh", remotepath=f"{WORKING_PATH}/configurations.sh")
        sftp.put(localpath=f"{EXECUTION_PATH}/gpu_exporter/requirements.txt", remotepath=f"{WORKING_PATH}/gpu_exporter/requirements.txt")
        sftp.put(localpath=f"{EXECUTION_PATH}/gpu_exporter/detect_gpu.py", remotepath=f"{WORKING_PATH}/gpu_exporter/detect_gpu.py")
        sftp.put(localpath=f"{EXECUTION_PATH}/gpu_exporter/gpu_export_metrics.py", remotepath=f"{WORKING_PATH}/gpu_exporter/gpu_export_metrics.py")
    #Ejecutamos el script configurations.sh en el servidor
    run_command(ssh, f"chmod 777 {WORKING_PATH}/gpu_exporter/*")
    run_command(ssh,f"python3 -m venv {WORKING_PATH}/venv")
    run_command(ssh,f"{WORKING_PATH}/venv/bin/pip3 install -r {WORKING_PATH}/gpu_exporter/requirements.txt")
    run_command(ssh, f"chmod 755 {WORKING_PATH}/configurations.sh")
    run_command(ssh, f'{WORKING_PATH}/configurations.sh "{password}" {ollama_version} {node_version} {reinstall_ollama}')

    logger.debug_color(f"\[+] Configured environment [/]")

def gpu_exporter_configuration(ssh: paramiko.SSHClient):
    #arrancamos el script de exportacion
    run_command(ssh, f"{WORKING_PATH}/venv/bin/python3 {WORKING_PATH}/gpu_exporter/gpu_export_metrics.py >> pepe.txt 2>&1 &")

def copy_file(src: str, dst: str):

    try:
        shutil.copyfile(src = src, dst = dst)

    except FileNotFoundError:
        logger.exception_color(f"ERROR file: {src} not found, impossible to copy")

    except Exception as e:
        logger.exception_color(f"Unexpected error while copying {src}, to {dst} : {e}")

def save_experiments():
    fixed_time = str(START_TIME).replace(" ","-").replace(":","-").split(".")[0]

    logger.debug_color(f"Saving experiments results...")

    #Defining the paths to the targets
    experiment_dir = os.path.join(EXECUTION_PATH, "experiment_results",fixed_time)

    #Tuple of tuples with (subpath, file_name)
    files_to_copy = (
        ("metrics", "ollama_metrics.csv"),
        ("metrics", "prometheus_metrics.csv"),
        ("logger", "logs.txt")
    )
    os.makedirs(experiment_dir, exist_ok = True)
    
    for subpath, file_name in files_to_copy:
        src =  os.path.join(EXECUTION_PATH, subpath, file_name)
        dst = os.path.join(experiment_dir, file_name)
        copy_file(src,dst)

    logger.debug_color(f"Experiments results saved!")
    

def clean_local_resources():
    logger.debug_color(f"Cleaning up local resources...")

    if platform.system() =="Windows":
        os.system(f'taskkill /PID 9090 /F')
    else:
        os.system(f'pkill -f prometheus')
    
    logger.debug_color(f"Local resources cleaned!")

def clean_sut_resources(ssh: paramiko.SSHClient):
    logger.debug_color(f"Cleaning up SUT resources...")

    process_to_kill = ["ollama", "node_exporter", "gpu_exporter"]
    for process in process_to_kill:
        run_command(ssh, f'pkill -f {process}')
    
    
    logger.debug_color("SUT resources cleaned!")

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
@click.option("--password", "-p",help="Password of the user to connect in target destination", default = "")
@click.option("--ip-address", "-i", required=True, callback=validarIp, help="Ip-Address of the host where the test it's going to be executed")
@click.option("--ollama-version", "-ov", callback=validate_ollama_version, help="ollama version to install in the SUT you must put the \"vx.x.x\"", default = "v0.7.0")
@click.option("--node-version", "-nv", callback=validate_node_exporter_version, help="node_exporter version to install in the SUT you must put the \"vx.x.x\"", default = "v1.9.1")
@click.option("--private-key", "-pk", help="Path to private key(including the name) in .pem format for ssh authentication", default=f"{os.getenv('HOME')}/.ssh/id_rsa")
@click.option("--reinstall-ollama", "-ro", is_flag=True, help="Force reinstallation of Ollama even if it's already installed", default=False)
def procesarLLM(ip_address: str, private_key: str, user: str, password: str, ollama_version: str, node_version: str, reinstall_ollama: bool):
    ssh = None
    ollama = None
    prometheus = None

    try:
        ssh = connection_establishment(user, password, ip_address, private_key)
        environment_configuration(ssh, password, ollama_version, node_version, reinstall_ollama)
        gpu_available = is_gpu_available(ssh)
        if gpu_available:
            gpu_exporter_configuration(ssh)
        run_command(ssh, f"OLLAMA_HOST=0.0.0.0 {OLLAMA_PATH}/bin/ollama serve > /dev/null 2>&1 &") # poner el OLLAMA_HOST
        prometheus = PrometheusHandler(ip_address, gpu_available)
        os.system("sleep 20")
        #-----Instalando LLMS-------
        ollama = OllamaHandler(ip_address)
        prometheus.start_collection()
        #Usar api de ollama para el texto
        ollama.process_models()
        prometheus.stop_collection()
        save_experiments()
       
    except (paramiko.AuthenticationException, paramiko.BadHostKeyException, paramiko.SSHException) as e:
        logger.exception_color(e)

    except Exception as e:
        logger.exception_color(e)
    
    finally:
        clean_local_resources()

        if ssh:
            clean_sut_resources(ssh)


if __name__ == '__main__':
    procesarLLM()