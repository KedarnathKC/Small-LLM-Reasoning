import os
import logging
import requests
from typing import Dict, List, Optional, Union, Tuple, Any
from functools import cached_property
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.api_models import JsonChatStr
from lm_eval.api.registry import register_model
import openai
import copy
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@register_model("multi-provider-api")
class MultiProviderAPI(TemplateAPI):
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        base_url: Optional[str] = None,
        tokenizer_backend: str = "tiktoken",
        num_concurrent: int = 1,
        max_retries: int = 3,
        batch_size: int = 1,
        append_prompt: str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        **kwargs
    ):
        """
        Initialize the API client with the specified provider.
        
        Args:
            provider (str): The provider to use ('openai', 'together', 'ollama', 'llama')
            api_key (str, optional): API key for the provider. If not provided, will look for environment variable
            model (str): The model to use
            base_url (str, optional): Override the default base URL for the provider
            tokenizer_backend (str): The tokenizer backend to use
            num_concurrent (int): Number of concurrent API requests
            max_retries (int): Maximum number of retries for failed requests
            batch_size (int): Batch size for requests
            **kwargs: Additional arguments passed to TemplateAPI
        """
        self.provider = provider.lower()
        assert self.provider in ["openai", "llama", "ollama", "together"]
        
        # Set up provider-specific API keys and base URLs
        self.base_url = base_url
        
        # Store the API key for later use in headers
        self._api_key = api_key
        self.append_prompt = append_prompt

        self.client = openai.OpenAI(base_url=self.base_url if self.base_url else None , api_key=self.api_key)
        
        # Call the parent class constructor with appropriate parameters
        super().__init__(
            model=model,
            base_url=self.base_url,
            tokenizer_backend=tokenizer_backend,
            num_concurrent=num_concurrent,
            max_retries=max_retries,
            batch_size=batch_size,
            **kwargs
        )

    
    @cached_property
    def api_key(self):
        """Return the API key for the current provider"""
        return self._api_key
    
    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate: bool = False,
        gen_kwargs: Optional[dict] = None,
        **kwargs
    ) -> dict:
        """
        Create the JSON payload for the API request
        
        Args:
            messages: The input messages or tokens
            generate: Whether this is a generation request
            gen_kwargs: Additional generation-specific parameters
            **kwargs: Additional parameters
            
        Returns:
            dict: The payload for the API request
        """
        print("Creating my custom payload!")
        # Initialize generation parameters
        gen_params = {
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 256 if generate else 0,  # Default max_tokens
        }
        
        # Update with any provided generation parameters
        if gen_kwargs:
            gen_params.update(gen_kwargs)
            
        # Format the messages based on the input type
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            # Already in chat format
            formatted_messages = messages
        elif isinstance(messages, str):
            # Single string prompt
            formatted_messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list) and messages and isinstance(messages[0], str):
            # List of strings
            formatted_messages = [{"role": "user", "content": m} for m in messages]
        else:
            # Assume tokenized input, convert back to text if needed
            # This would need a proper implementation based on your tokenizer
            if self.tokenizer:
                text_messages = self._convert_tokens_to_text(messages)
                formatted_messages = [{"role": "user", "content": m} for m in text_messages]
            else:
                raise ValueError("Cannot process tokenized input without a tokenizer")
        
        # Create the final payload
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            **gen_params
        }
        
        # Add logprobs if needed
        if generate and gen_kwargs and gen_kwargs.get("logprobs", False):
            payload["logprobs"] = True
        print("Payload: ", payload)
        return payload
    
    def _convert_tokens_to_text(self, tokens_list: List[List[int]]) -> List[str]:
        """Convert token IDs back to text"""
        if not self.tokenizer:
            raise ValueError("No tokenizer available for token conversion")
            
        return [self.tokenizer.decode(tokens) for tokens in tokens_list]
    
    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs
    ) -> List[Tuple[float, bool]]:
        """
        Parse log probabilities from API responses
        
        Args:
            outputs: The API response(s)
            tokens: The token IDs corresponding to the continuation
            ctxlens: The context lengths (for slicing the outputs)
            **kwargs: Additional arguments
            
        Returns:
            List[Tuple[float, bool]]: Pairs of (log_prob, is_greedy)
        """
        results = []
        
        # Ensure outputs is a list
        if not isinstance(outputs, list):
            outputs = [outputs]
            
        for output in outputs:
            # Extract the log probabilities from the response
            # This implementation will vary based on the provider's response format
            try:
                choice = output.get("choices", [{}])[0]
                logprobs_data = choice.get("logprobs", {})
                
                if not logprobs_data:
                    # No logprobs available
                    results.append((0.0, True))
                    continue
                
                # Process logprobs based on the provider's format
                # This is a simplified example - actual implementation depends on API response format
                token_logprobs = logprobs_data.get("token_logprobs", [])
                top_logprobs = logprobs_data.get("top_logprobs", [])
                
                if token_logprobs:
                    # Sum the log probabilities of the continuation tokens
                    log_prob = sum(token_logprobs)
                    
                    # Determine if each token was the most likely (greedy)
                    is_greedy = True
                    if top_logprobs:
                        for i, top_dict in enumerate(top_logprobs):
                            if top_dict and max(top_dict.values()) > token_logprobs[i]:
                                is_greedy = False
                                break
                    
                    results.append((log_prob, is_greedy))
                else:
                    # Fallback if token_logprobs not available
                    results.append((0.0, True))
                    
            except (KeyError, IndexError) as e:
                logger.warning(f"Error parsing logprobs: {e}")
                results.append((0.0, True))
                
        return results
    
    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        """
        Parse generated text from API responses
        
        Args:
            outputs: The API response(s)
            **kwargs: Additional arguments
            
        Returns:
            List[str]: The generated text
        """
        generations = []
        
        # Ensure outputs is a list
        if not isinstance(outputs, list):
            outputs = [outputs]
            
        for output in outputs:
            try:
                # All providers (including Together) use the same response structure
                content = output.choices[0].message.content
                generations.append(content)
            except (KeyError, IndexError, AttributeError) as e:
                logger.warning(f"Error parsing generation: {e}")
                generations.append("")
                
        return generations
    
    def model_call(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[dict]:
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)
        print("Final Message: ", messages[0] + self.append_prompt)
        bbb
        response = self.client.chat.completions.create(
                model=self.model,
                messages=[messages[0] + self.append_prompt],
                max_tokens=gen_kwargs.get("max_tokens", 256),
                temperature=gen_kwargs.get("temperature", 0.7),
                top_p=gen_kwargs.get("top_p", 1.0),
                logprobs=gen_kwargs.get("logprobs", False),
                **kwargs
        )
        print(response)
        return response