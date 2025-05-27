import requests
import json
class Ollama():
    def __init__(self, host_ip: str, host_port: str = "11434"):
        self.url = f"http://{host_ip}:{host_port}"
    
    def chat_prompt(self, model: str, prompts: list) -> requests.Response:
        response = requests.post(
            f"{self.url}/api/chat",
            json = {
                "model" : model,
                "messages" : prompts,
                "stream" : False
            }
        )

        return response
    
    def single_prompt(self, model: str, prompt: str) -> requests.Response:
        response = requests.post(
            f"{self.url}/api/generate",
            json = {
                "model" : model,
                "prompt" : prompt,
                "stream" : False #inside JSON is for receiving the response by chunks/token by token, if false the total response is send in one time
            }
        )

        return response
    
    def pull_model(self, model: str) -> None:
        requests.post(
            f"{self.url}/api/pull",
            json = {"name": model},
            stream = False # Stream out the json is for wait to the total response instead of recieving by chunks/token by token
        )
    
    def get_models(self) -> requests.Response:
        response  = requests.get(
            f"{self.url}/api/tags"
        )
        
        return response
    
    def list_models(self) -> None: # Function for debugging purposes
        response = self.get_models()
        data = response.json()
        models = data.get("models")
        for model in models:
            print(model.get("name"))