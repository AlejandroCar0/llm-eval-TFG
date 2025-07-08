from ollama.ollama import Ollama
import re
import os
import json
from logger.log import logger
from dateutil import parser
EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_PATH = f"{EXECUTION_PATH}/../metrics"
EXPORT_METRICS = "timestamp;total_duration;load_duration;prompt_eval_count;prompt_eval_duration;eval_count;eval_duration;model"
class OllamaHandler():
    def __init__(self, ip_address: str):
        self.ollama = Ollama(ip_address)
        self.log = logger.getChild(__name__)

        with open(f"{METRICS_PATH}/ollama_metrics.csv", "w") as f:
            f.write(f"{EXPORT_METRICS}\n")

        with open (f"{METRICS_PATH}/response.txt","w") as f:
            f.write("Model;Prompt;response\n")
        
        with open (f"{METRICS_PATH}/models_score.csv", "w") as f:
            f.write("Model;Score\n")
        
        self.models = self.get_models()
        self.prompts = self.read_prompts()
        self.answers = self.read_answers()
        self.answer_index = 0
        self.model_point = 0
        

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
            self.log.debug_color('Reading ollama models...')

            with open(f"{EXECUTION_PATH}/model_list.txt","r") as modelList:
                models = [line.rstrip("\n") for line in modelList]

            self.log.debug_color('Ollama models read')

        except FileNotFoundError:
            self.log.warning_color(f"No file {models_file} found!!!")

        return models
    
    def get_models(self) -> list:
        models = self.read_models()
        models = self.pull_models(models)

        return models
    
    def read_answers(self) -> list:
        answers = []
        answers_file = os.path.join(EXECUTION_PATH, 'answers.txt')

        try:

            with open(f'{EXECUTION_PATH}/answers.txt', 'r') as answersList:
                answers = [line.rstrip("\n") for line in answersList]

        except FileNotFoundError:
            self.log.warning_color(f"No file {answers_file} found!!!")
        
        return answers

    def read_prompts(self) -> list:
        prompts = []
        prompts_file = os.path.join(EXECUTION_PATH, 'prompts.txt')

        try:
            self.log.debug_color('Reading prompts...')
            with open(f"{EXECUTION_PATH}/prompts.txt","r") as promptsList:
                for line in promptsList:
                    line = line.rsplit("\;")
                    prompts.append(line)

            self.log.debug_color('Prompts read!')

        except FileNotFoundError:
            self.log.warning_color(f'No file {prompts_file} found!!!')

        return prompts

    def load_model(self, model: str) -> None:
        self.log.debug_color("Loading model....")
        self.ollama.load_model(model)
        self.log.debug_color("Model loaded!!")

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

    def parse_response(self, response: str) -> str:
        parse_response = ""
        parse_response = re.search(r'\{.*?\}', response, re.DOTALL)
        if parse_response:
            parse_response = parse_response.group()
            parse_response = parse_response.lower()
            try:
               parse_response = json.loads(parse_response)

               if "response" not in parse_response:
                   raise Exception()
            
            except (json.JSONDecodeError,Exception):
                parse_response = json.loads('{"response": "Bad format for the response"}')

        else:
            parse_response = json.loads('{"response": "Bad format for the response"}')
        #Debug
        self.log.warning_color(f"{json.dumps(parse_response, indent=4)}")

        return parse_response

    def evaluate_response(self, response: str, model: str, possible_answers: str) -> int:
        response = self.parse_response(response)
        response = str(response['response'])

        possible_answers = possible_answers.lower()
        #Split if real_response can have different options for example if the model answer with d and option d is Spain you can have d | spain
        possible_answers = possible_answers.split("|")
        for answer in possible_answers:
            if answer in response:
                #Debug
                self.log.warning_color(f"Si, la respuesta: {answer} se encuentra en {response}")
                return 1
            
            else:
                #Debug
                self.log.warning_color(f"No, la respuesta: {answer} no se encuentra en {response}")
        
        return 0

    def process_prompt(self, prompts: list, model: str) -> None:
        messages = []
        parsed_prompt = ""
        #They can be multiprompts
        for prompt in prompts:
            parsed_prompt += self.parse_user_prompt(prompt)

            self.log.debug_color(f"processing prompt: {prompt}")
            response = self.ollama.single_prompt(model, prompt)
            self.log.debug_color("Prompt processed")
            if response.status_code == 200:
                data = response.json()
                parsed_prompt += self.parse_assistant_prompt(data)

                if "{E}" in prompt:
                    self.model_point += self.evaluate_response(data.get('response'), model, self.answers[self.answer_index])
                    self.answer_index += 1

                self.write_metrics(self.extract_metrics(data))
                self.save_response(prompt, data.get('response'), model)
                print(json.dumps(response.json(), indent = 4)) #debugging purposes
                
            else:
                self.log.warning_color(f"!!ERROR, model {model}, failed processing prompt: {prompt}\nReason: {response.reason}\nBody:{response.text}")
            
        
    

    def process_models(self):
        for model in self.models:
            self.load_model(model)
            self.answer_index = 0
            self.model_point = 0

            for prompt in self.prompts:
                self.process_prompt(prompt, model)
            
            model_final_mark = (self.model_point*10)/len(self.answers) if self.model_point != 0 else 0

            with open(f"{METRICS_PATH}/models_score.csv", "a") as f:
                f.write(f'{model};{model_final_mark}\n')
