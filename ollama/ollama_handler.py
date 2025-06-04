from ollama.ollama import Ollama
import os
import sys
from logger.log import logger
EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))

class OllamaHandler():
    def __init__(self, ip_address: str):
        self.ollama = Ollama(ip_address)
        self.log = logger.getChild(__name__)
    def pull_models(self, models: list) -> None:

        for model in models:
            self.log.debug_color(f"Pulling model: {model}")
            self.ollama.pull_model(model)
            self.log.debug_color(f"Model\[{model}] downloaded")

    def read_models(self) -> list:
        with open(f"{EXECUTION_PATH}/modelList.txt","r") as modelList:
            models = [line.rstrip("\n") for line in modelList]

        return models

    def read_prompts(self) -> list:
        prompts = []
        with open(f"{EXECUTION_PATH}/prompts.txt","r") as promptsList:
            for line in promptsList:
                line = line.rsplit("\;")
                prompts.append(line)

        return prompts

    def process_prompt(self, prompts: list, model: str) -> None:
        if len(prompts) == 1:
            response =  self.ollama.single_prompt(model, prompts[0])
        else:
            messages = []
            for prompt in prompts:
                prompt = {"role" : "user", "content" : f"{prompt}"}
                messages.append(prompt)
                response = self.ollama.chat_prompt(model,messages)
                data = response.json()
                messages.append(data.get("message"))
                #Subdividir todo esto en funciones para dejarlo clean
            print(messages)

    def process_models(self):
        models = self.read_models()
        #pull models
        self.pull_models(models)
            
        #process prompts
        prompts = self.read_prompts()
        for prompt in prompts:
            self.log.debug_color(f"processing prompt: {prompt}")
            self.process_prompt(prompt, "phi3")
            self.log.debug_color(f"prompt processed")
        self.ollama.list_models()