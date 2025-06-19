import openai
import os
from typing import Dict, List, Optional
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class API:
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize the API client with the specified provider.
        
        Args:
            provider (str): The provider to use ('openai', 'together', 'ollama', 'llama')
            api_key (str, optional): API key for the provider. If not provided, will look for environment variable.
        """
        self.provider = provider.lower()
        assert self.provider in ["openai", "llama", "ollama", "together"]
        
        # Set up provider-specific API keys
        if self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            self.base_url = None
            
        elif self.provider == "together":
            self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
            if not self.api_key:
                raise ValueError("No Together API key provided. Set TOGETHER_API_KEY environment variable or pass api_key parameter.")
            self.base_url = "https://api.together.xyz"
            
        elif self.provider == "llama":
            self.api_key = api_key or os.getenv("LLAMA_API_KEY")
            if not self.api_key:
                raise ValueError("No Llama API key provided. Set LLAMA_API_KEY environment variable or pass api_key parameter.")
            self.base_url = "https://api.llama.com/compat"
            
        elif self.provider == "ollama":
            # Ollama doesn't require an API key as it runs locally
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.api_key = "ollama"

        self.client = openai.OpenAI(base_url=self.base_url + "/v1" if self.base_url else None, api_key=self.api_key)
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_completion(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        logprobs: bool = True,
        **kwargs
    ) -> Dict:
        """
        Get a text completion from the specified model.
        
        Args:
            prompt (str): The input prompt
            model (str): The model to use
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter
            logprobs (bool): Whether to return log probabilities
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dict: Response containing completion and metadata
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                logprobs=logprobs,
                **kwargs
            )
            return {
                "text": response.choices[0].message.content,
                "logprobs": response.choices[0].logprobs if logprobs else None,
                "model": model,
                "provider": self.provider
            }
                
        except Exception as e:
            logger.error(f"Error getting completion from {self.provider}: {str(e)}")
            raise
        
    def list_models(self) -> List[str]:
        """
        List available models for the current provider.
        
        Returns:
            List[str]: List of available model names
        """
        try:
            if self.provider == "openai" or self.provider == "llama":
                models = self.client.models.list()
                return [model.id for model in models.data]
            
            elif self.provider == "together":
                headers = {
                    "accept": "application/json",
                    "authorization": f"Bearer {self.api_key}"
                }

                response = requests.get(self.base_url + "/models", headers=headers)
                response.raise_for_status()
                response = response.json()
                return [model["id"] for model in response]
                
            elif self.provider == "ollama":
                response = requests.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                return [model["name"] for model in response.json()["models"]]

            else:
                logger.warning(f"Unsupported provider: {self.provider}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing models for {self.provider}: {str(e)}")
            raise
