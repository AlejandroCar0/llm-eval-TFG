import requests
import re
import os
EXEC_PATH = os.path.dirname(os.path.realpath(__file__))

def _get_versions(url: str) -> list[str]:
    versions = set()
    more_versions = True
    page = 1
    while more_versions:
        response = requests.get(f"{url}{page}")
        versions_from_page = response.text
        pattern = "([vV]){1}([0-9]{1,2}\.){2}[0-9]"
        versions_from_page = re.finditer(pattern, versions_from_page)
        versions_from_page = sorted(set(version.group(0) for version in versions_from_page))
        if len(versions_from_page) !=0:
            versions.update(versions_from_page)
            page += 1
        else:
            more_versions = False
    return sorted(versions,reverse= True)

def _update_versions(url: str, file_name: str):
    versions = _get_versions(url = url)
    try:
        with open(f"{EXEC_PATH}/{file_name}", "w") as f:
            f.write("\n".join(versions))
    except FileNotFoundError as e:
        print(f"File {file_name} not found in directory: {EXEC_PATH}")
    

def update_ollama_versions():
    url = "https://github.com/ollama/ollama/releases?page="
    _update_versions(url= url, file_name= "ollama_versions.txt")

def update_node_exporter_versions():
    url = "https://github.com/prometheus/node_exporter/releases?page="
    _update_versions(url= url, file_name= "node_exporter_versions.txt")

def main():
    update_ollama_versions()
    update_node_exporter_versions()

if __name__ == "__main__":
    main()