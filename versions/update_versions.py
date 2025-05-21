import requests
import re
import os
EXEC_PATH = os.path.dirname(os.path.realpath(__file__))

def update_ollama_versions():
    with open(f"{EXEC_PATH}/ollama_versions.txt","w") as f:
        versions = requests.get("https://github.com/ollama/ollama/releases/").text
        pattern = "([vV]){1}([0-9]{1}\.){2}[0-9]"
        versions = re.finditer(pattern,versions)
        versions = sorted(set(version.group(0) for version in versions))
        versions = "\n".join(versions)
        f.write(versions)
    