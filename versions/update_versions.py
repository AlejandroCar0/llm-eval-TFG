import requests
import re
import os
EXEC_PATH = os.path.dirname(os.path.realpath(__file__))

def _get_versions(repo_name: str) -> list[str]:
    versions = []
    page = 1
    per_page = 100  # máximo permitido por GitHub

    while True:
        url = f"https://api.github.com/repos/{repo_name}/releases"
        params = {'per_page': per_page, 'page': page}
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"Error al acceder al repositorio: {response.status_code}")
            break

        releases = response.json()
        if not releases:
            break  # No hay más páginas

        versions.extend([release['tag_name'] for release in releases])
        page += 1

    return versions

def _update_versions(repo_name: str, file_name: str):
    versions = _get_versions(repo_name = repo_name)
    try:
        with open(f"{EXEC_PATH}/{file_name}", "w") as f:
            f.write("\n".join(versions))
    except FileNotFoundError as e:
        print(f"File {file_name} not found in directory: {EXEC_PATH}")
    

def update_ollama_versions():
    _update_versions(repo_name= "ollama/ollama", file_name= "ollama_versions.txt")

def update_node_exporter_versions():
    _update_versions(repo_name= "prometheus/node_exporter", file_name= "node_exporter_versions.txt")

def update():
    update_ollama_versions()
    update_node_exporter_versions()

update()