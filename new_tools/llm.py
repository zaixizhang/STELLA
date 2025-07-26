import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from openai import OpenAI
import json
import os
from typing import Optional, Dict, Any

class LLMChat:
    """Simple LLM chat provider using OpenRouter API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM client with OpenRouter API
        
        Args:
            api_key: OpenRouter API key. If not provided, will use OPENROUTER_API_KEY env var
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
        
        # Common OpenRouter model configurations
        self.model_configs = {
            # OpenAI models via OpenRouter
            "gpt-4o": {"model": "openai/gpt-4o", "temperature": 0.0, "supports_json": True},
            "gpt-4o-mini": {"model": "openai/gpt-4o-mini", "temperature": 0.0, "supports_json": True},
            "gpt-4-turbo": {"model": "openai/gpt-4-turbo", "temperature": 0.0, "supports_json": True},
            "o1": {"model": "openai/o1", "temperature": None, "supports_json": False},
            "o1-mini": {"model": "openai/o1-mini", "temperature": None, "supports_json": False},
            
            # Anthropic models via OpenRouter
            "claude-3.5-sonnet": {"model": "anthropic/claude-3.5-sonnet", "temperature": 0.0, "supports_json": True},
            "claude-3-opus": {"model": "anthropic/claude-3-opus", "temperature": 0.0, "supports_json": True},
            "claude-3-haiku": {"model": "anthropic/claude-3-haiku", "temperature": 0.0, "supports_json": True},
            
            # Google models via OpenRouter
            "gemini-2.5-pro": {"model": "google/gemini-2.5-pro", "temperature": 0.0, "supports_json": True},
            "gemini-2.0-flash": {"model": "google/gemini-2.0-flash-exp", "temperature": 0.0, "supports_json": True},
            "gemini-1.5-pro": {"model": "google/gemini-pro-1.5", "temperature": 0.0, "supports_json": True},
            
            # Other popular models
            "deepseek-r1": {"model": "deepseek/deepseek-r1", "temperature": 0.0, "supports_json": False},
            "deepseek-chat": {"model": "deepseek/deepseek-chat", "temperature": 0.0, "supports_json": True},
            "llama-3.3-70b": {"model": "meta-llama/llama-3.3-70b-instruct", "temperature": 0.0, "supports_json": True},
            "qwen-2.5-72b": {"model": "qwen/qwen-2.5-72b-instruct", "temperature": 0.0, "supports_json": True},
            
            # Default fallback
            "default": {"model": "google/gemini-2.5-pro", "temperature": 0.0, "supports_json": True}
        }
    
    def chat(self, request: str, model_name: str = "gemini-2.5-pro", 
             temperature: Optional[float] = None, 
             json_mode: bool = False,
             max_tokens: Optional[int] = None,
             system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Make a chat request to OpenRouter
        
        Args:
            request: User message content
            model_name: Model identifier (use keys from model_configs)
            temperature: Override default temperature
            json_mode: Force JSON response format
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            
        Returns:
            Dict containing response or error information
        """
        try:
            # Get model configuration
            config = self.model_configs.get(model_name, self.model_configs["default"])
            model_id = config["model"]
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": request})
            
            # Prepare request parameters
            params = {
                "model": model_id,
                "messages": messages
            }
            
            # Set temperature
            if temperature is not None:
                params["temperature"] = temperature
            elif config["temperature"] is not None:
                params["temperature"] = config["temperature"]
                    
            # Set max tokens
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            
            # Enable JSON mode if requested and supported
            if json_mode and config["supports_json"]:
                params["response_format"] = {"type": "json_object"}
            
            # Make the API call
            response = self.client.chat.completions.create(**params)
            
            # Process the response
            message = response.choices[0].message
            
            if hasattr(message, 'refusal') and message.refusal:
                return {"error": "Model refused to respond", "details": message.refusal}
            
            content = message.content
            if not content:
                return {"error": "Empty response from model"}
            
            # If JSON mode was requested, try to parse the response
            if json_mode:
                try:
                    parsed_content = self._parse_json_response(content)
                    return {"response": parsed_content, "raw_content": content}
                except json.JSONDecodeError as e:
                    return {"error": f"Failed to parse JSON response: {str(e)}", "raw_content": content}
            else:
                return {"response": content}
                
        except Exception as e:
            return {"error": f"API call failed: {str(e)}"}
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        Parse JSON response, handling markdown code blocks
        """
        # Clean markdown code blocks
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        elif content.startswith("```"):
            content = content[3:]   # Remove ```
        
        if content.endswith("```"):
            content = content[:-3]  # Remove trailing ```
        
        content = content.strip()
        return json.loads(content)
    
    def get_available_models(self) -> list:
        """Get list of available model names"""
        return list(self.model_configs.keys())
    
    def simple_chat(self, request: str, model_name: str = "gemini-2.5-pro") -> str:
        """
        Simple chat method that returns just the response content
        
        Args:
            request: User message
            model_name: Model to use
            
        Returns:
            Response content as string, or error message
        """
        result = self.chat(request, model_name)
        if "error" in result:
            return f"Error: {result['error']}"
        return result["response"]
    
    def json_chat(self, request: str, model_name: str = "gemini-2.5-pro") -> Dict[str, Any]:
        """
        Chat method that forces JSON response format
        
        Args:
            request: User message (should ask for JSON format)
            model_name: Model to use
            
        Returns:
            Parsed JSON response or error dict
        """
        result = self.chat(request, model_name, json_mode=True)
        if "error" in result:
            return result
        return result["response"]


# Global instance for easy access
_llm_instance = None

def get_llm_client(api_key: Optional[str] = None) -> LLMChat:
    """Get or create global LLM client instance"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMChat(api_key=api_key)
    return _llm_instance

def simple_llm_call(request: str, model_name: str = "gemini-2.5-pro") -> str:
    """
    Simple function for making LLM calls
    
    Args:
        request: User message
        model_name: Model to use
        
    Returns:
        Response content as string
    """
    client = get_llm_client()
    return client.simple_chat(request, model_name)

def json_llm_call(request: str, model_name: str = "gemini-2.5-pro") -> Dict[str, Any]:
    """
    Function for making LLM calls with JSON response
    
    Args:
        request: User message (should ask for JSON format)
        model_name: Model to use
        
    Returns:
        Parsed JSON response
    """
    client = get_llm_client()
    return client.json_chat(request, model_name)