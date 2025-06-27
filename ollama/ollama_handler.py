from ollama.ollama import Ollama
import os
import sys
import json
from logger.log import logger
from dateutil import parser
EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = f"{EXECUTION_PATH}/../metrics"
EXPORT_METRICS = f"timestamp;total_duration;load_duration;prompt_eval_count;prompt_eval_duration;eval_count;eval_duration;model"
class OllamaHandler():
    def __init__(self, ip_address: str):
        self.ollama = Ollama(ip_address)
        self.log = logger.getChild(__name__)

        with open(f"{METRICS_PATH}/ollama_metrics.csv", "w") as f:
            f.write(f"{EXPORT_METRICS}\n")

        with open (f"{METRICS_PATH}/response.txt","w") as f:
            f.write(f"Model;Prompt;response\n")
        
        self.models = self.get_models()
        self.prompts = self.read_prompts()
        

    def pull_models(self, models: list) -> list:
        real_models = []
        for model in models:
            self.log.debug_color(f"Pulling model: {model}")

            try:
                self.ollama.pull_model(model)
                real_models.append(model)
                self.log.debug_color(f"Model\[{model}] downloaded")
            
            except Exception as e:
                self.log.warning_color(f'{e}')
        
        return real_models

    def read_models(self) -> list:
        
        models = []
        models_file = os.path.join(EXECUTION_PATH, "model_list.txt")

        try:
            self.log.debug_color(f'Reading ollama models...')

            with open(f"{EXECUTION_PATH}/model_list.txt","r") as modelList:
                models = [line.rstrip("\n") for line in modelList]

            self.log.debug_color(f'Ollama models read')

        except FileNotFoundError:
            self.log.warning_color(f"No file {models_file} found!!!")

        return models
    
    def get_models(self) -> list:
        models = self.read_models()
        models = self.pull_models(models)

        return models

    def read_prompts(self) -> list:
        prompts = []
        prompts_file = os.path.join(EXECUTION_PATH, 'prompts.txt')

        try:
            self.log.debug_color(f'Reading prompts...')
            with open(f"{EXECUTION_PATH}/prompts.txt","r") as promptsList:
                for line in promptsList:
                    line = line.rsplit("\;")
                    prompts.append(line)

            self.log.debug_color(f'Prompts read!')

        except FileNotFoundError:
            self.log.warning_color(f'No file {prompts_file} found!!!')

        return prompts

    def load_model(self, model: str) -> None:
        self.log.debug_color(f"Loading model....")
        self.ollama.load_model(model)
        self.log.debug_color(f"Model loaded!!")

    def parse_user_prompt(self, prompt: str)-> str:

        return f"User: {prompt}\n"
    
    def parse_assistant_prompt(self, data: dict):
        assistant_response = data.get('response')
        
        return f"Assistant: {assistant_response}\n"
    
    def parse_timestamp(self, data: dict) -> float:
        timestamp = data.get('created_at')
        dt = parser.isoparse(timestamp)
        timestamp = dt.timestamp()

        return str(timestamp)
            
    def extract_metrics(self, data: dict) -> dict:
        result = {}
        metrics = EXPORT_METRICS.split(";")
        metrics = metrics[1:] #taking out timestamp

        result["timestamp"] = self.parse_timestamp(data)

        for metric in metrics:
            result[metric] = str(data.get(metric))
        
        return result
        
    def write_metrics(self, metrics: dict) -> None:
        values = []
        print(json.dumps(metrics, indent= 4))#debugging purposes
        with open(f"{METRICS_PATH}/ollama_metrics.csv", "a") as f:
            for key,value in metrics.items():
                values.append(value)
            
            f.write(";".join(values) + "\n")
                
    def save_response(self, prompt: str, response: str, model: str) -> None:
        with open(f"{METRICS_PATH}/response.txt", "a") as f:
            f.write(f"Model: {model};Prompt:{prompt};Response: {response}\n")


    def process_prompt(self, prompts: list, model: str) -> None:
        messages = []
        parsed_prompt = ""
        #They can be multiprompts
        for prompt in prompts:
            parsed_prompt += self.parse_user_prompt(prompt)

            self.log.debug_color(f"processing prompt: {prompt}")
            response = self.ollama.single_prompt(model, prompt)
            self.log.debug_color(f"Prompt processed")
            if response.status_code == 200:
                data = response.json()
                parsed_prompt += self.parse_assistant_prompt(data)

                self.write_metrics(self.extract_metrics(data))
                self.save_response(prompt, data.get('response'), model)
                print(json.dumps(response.json(), indent = 4)) #debugging purposes
                
            else:
                self.log.warning_color(f"!!ERROR, model {model}, failed processing prompt: {prompt}\nReason: {response.reason}\nBody:{response.text}")
            
        
    

    def process_models(self):
        for model in self.models:
            self.load_model(model)

            for prompt in self.prompts:
                self.process_prompt(prompt, model)
