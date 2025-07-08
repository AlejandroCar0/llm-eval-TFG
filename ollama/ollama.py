import requests

class Ollama():
    """
    Clase base para la comunicación con la API REST de Ollama.
    
    Proporciona una interfaz simplificada para interactuar con el servidor Ollama,
    encapsulando las llamadas HTTP a los diferentes endpoints de la API para
    gestión de modelos, generación de texto y operaciones de chat.
    
    Attributes:
        url (str): URL base del servidor Ollama construida con IP y puerto
    """
    def __init__(self, host_ip: str, host_port: str = "11434"):
        """
        Inicializa la conexión con el servidor Ollama.
        
        Args:
            host_ip (str): Dirección IP del servidor Ollama
            host_port (str, optional): Puerto del servidor Ollama. Por defecto "11434"
        """
        self.url = f"http://{host_ip}:{host_port}"
    
    def chat_prompt(self, model: str, prompts: list) -> requests.Response:
        """
        Realiza una conversación multi-mensaje con un modelo LLM.
        
        Utiliza el endpoint /api/chat para mantener contexto conversacional
        entre múltiples intercambios de mensajes.
        
        Args:
            model (str): Nombre del modelo a utilizar
            prompts (list): Lista de mensajes de la conversación
            
        Returns:
            requests.Response: Respuesta HTTP del servidor Ollama
        """
        response = requests.post(
            f"{self.url}/api/chat",
            json = {
                "model" : model,
                "messages" : prompts,
                "stream" : False
            }
        )

        return response

    def load_model(self, model: str):
        """
        Carga un modelo en memoria para optimizar respuestas posteriores.
        
        Args:
            model (str): Nombre del modelo a cargar
            
        Returns:
            requests.Response: Respuesta HTTP del servidor
        """
        return requests.post(f"{self.url}/api/generate")
    
    def single_prompt(self, model: str, prompt: str) -> requests.Response:
        """
        Ejecuta un prompt simple contra un modelo LLM.
        
        Utiliza el endpoint /api/generate para generar texto basado en un
        prompt único sin contexto conversacional previo.
        
        Args:
            model (str): Nombre del modelo a utilizar
            prompt (str): Texto del prompt a procesar
            
        Returns:
            requests.Response: Respuesta HTTP con la generación del modelo
        """
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
        """
        Descarga un modelo desde el repositorio de Ollama.
        
        Utiliza el endpoint /api/pull para descargar modelos desde
        la librería oficial de Ollama. Incluye validación de respuesta.
        
        Args:
            model (str): Nombre del modelo a descargar
            
        Raises:
            Exception: Si la descarga falla o el modelo no existe
        """
        response = requests.post(
            f"{self.url}/api/pull",
            json = {"name": model},
            stream = False # Stream out the json is for wait to the total response instead of recieving by chunks/token by token
        )
        if response.status_code != 200:
            raise Exception(f"Impossible to download model: {model}, you can check if the model exists in : https://ollama.com/library")
    
    def get_models(self) -> requests.Response:
        """
        Obtiene la lista de modelos disponibles en el servidor.
        
        Returns:
            requests.Response: Respuesta con la lista de modelos instalados
        """
        response  = requests.get(
            f"{self.url}/api/tags"
        )
        
        return response
    
    def list_models(self) -> None: # Function for debugging purposes
        """
        Imprime la lista de modelos disponibles en el servidor.
        
        Función de utilidad para debugging que muestra por consola
        todos los modelos instalados en el servidor Ollama.
        """
        response = self.get_models()
        data = response.json()
        models = data.get("models")
        for model in models:
            print(model.get("name"))