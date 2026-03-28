"""OpenAI-compatible LLM client with structured (Pydantic) outputs."""

import os
from pathlib import Path
from enum import Enum
from typing import Optional, TypeVar, Type, Any, Dict
from dataclasses import dataclass
from openai import OpenAI
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass


class LLMProvider(str, Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    LOCAL = "local"
    MOCK = "mock"


PROVIDER_BASE_URLS = {
    LLMProvider.OPENAI: "https://api.openai.com/v1",
    LLMProvider.OPENROUTER: "https://openrouter.ai/api/v1",
    LLMProvider.LOCAL: "http://localhost:8000/v1",
    LLMProvider.MOCK: None,
}

API_KEY_ENV_VARS = {
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
    LLMProvider.LOCAL: "LOCAL_LLM_API_KEY",
}


@dataclass
class LLMConfig:
    provider: LLMProvider = LLMProvider.OPENROUTER
    model: str = "openai/gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    
    def __post_init__(self):
        if self.base_url is None and self.provider in PROVIDER_BASE_URLS:
            self.base_url = PROVIDER_BASE_URLS[self.provider]

        if self.api_key is None and self.provider != LLMProvider.MOCK:
            env_var = API_KEY_ENV_VARS.get(self.provider)
            if env_var:
                self.api_key = os.getenv(env_var)
            
            if self.api_key is None:
                raise ValueError(
                    f"API key required for provider {self.provider}. "
                    f"Set {env_var} environment variable or pass api_key explicitly."
                )
    
    @classmethod
    def from_env(
        cls,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> "LLMConfig":
        """Build config from env (LLM_PROVIDER, LLM_MODEL, API keys, etc.)."""
        if provider is None:
            provider_str = os.getenv("LLM_PROVIDER", "openrouter")
            provider = LLMProvider(provider_str)
        
        if model is None:
            model = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
        
        return cls(
            provider=provider,
            model=model,
            base_url=os.getenv("LLM_BASE_URL") or kwargs.get("base_url"),
            temperature=float(os.getenv("LLM_TEMPERATURE", kwargs.get("temperature", 0.3))),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", kwargs.get("max_tokens", 1024))),
            **{k: v for k, v in kwargs.items() if k not in ["base_url", "temperature", "max_tokens"]}
        )


T = TypeVar('T', bound=BaseModel)


class LLMClient:
    """OpenAI-compatible chat client; supports structured parse and mock mode."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Optional[OpenAI] = None
    
    @classmethod
    def from_env(cls, **kwargs) -> "LLMClient":
        config = LLMConfig.from_env(**kwargs)
        return cls(config)
    
    @classmethod
    def mock(cls) -> "LLMClient":
        config = LLMConfig(provider=LLMProvider.MOCK)
        return cls(config)
    
    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        return self._client
    
    def generate(
        self,
        response_format: Type[T],
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> T:
        if self.config.provider == LLMProvider.MOCK:
            return self._generate_mock(response_format)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.config.model,
                messages=messages,
                response_format=response_format,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                **kwargs
            )
            
            return completion.choices[0].message.parsed
            
        except Exception as e:
            raise LLMError(
                f"Error generating response with {self.config.provider}/{self.config.model}: {e}"
            ) from e
    
    def generate_text(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        if self.config.provider == LLMProvider.MOCK:
            return "Mock response for testing"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                **kwargs
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            raise LLMError(
                f"Error generating text with {self.config.provider}/{self.config.model}: {e}"
            ) from e
    
    def _generate_mock(self, response_format: Type[T]) -> T:
        mock_data = self._create_mock_data(response_format)
        return response_format.model_validate(mock_data)
    
    def _create_mock_data(self, model: Type[BaseModel]) -> Dict[str, Any]:
        mock_data = {}
        for field_name, field_info in model.model_fields.items():
            annotation = field_info.annotation
            mock_data[field_name] = self._get_mock_value(annotation, field_name)
        return mock_data
    
    def _get_mock_value(self, annotation: Any, field_name: str) -> Any:
        origin = getattr(annotation, "__origin__", None)

        if origin is type(None) or annotation is type(None):
            return None

        if origin is type(None):
            return None

        if hasattr(annotation, "__args__"):
            args = annotation.__args__
            if type(None) in args:
                non_none = [a for a in args if a is not type(None)]
                if non_none:
                    return self._get_mock_value(non_none[0], field_name)

        if origin is list:
            return []

        if origin is dict:
            return {}

        if annotation is str or annotation == str:
            return f"mock_{field_name}"
        if annotation is int or annotation == int:
            return 0
        if annotation is float or annotation == float:
            return 0.0
        if annotation is bool or annotation == bool:
            return False

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return self._create_mock_data(annotation)
        
        return None


class LLMError(Exception):
    pass