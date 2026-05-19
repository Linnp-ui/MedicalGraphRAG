from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from loguru import logger


@runtime_checkable
class LLMProvider(Protocol):
    """LLM提供者协议"""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本响应"""
        ...
    
    def embed(self, text: str) -> List[float]:
        """生成文本嵌入"""
        ...
    
    async def async_generate(self, prompt: str, **kwargs) -> str:
        """异步生成文本响应"""
        ...


class DashScopeProvider:
    """阿里云DashScope LLM提供者"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://dashscope.aliyuncs.com/api/text-generation")
        self.model = config.get("model", "qwen-plus")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 2000)
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            import dashscope
            dashscope.api_key = self.api_key
            if self.base_url:
                dashscope.base_url = self.base_url
            self._client = dashscope.Generation
        return self._client
    
    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        try:
            response = client.call(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if response.status_code == 200:
                return response.output.text
            else:
                logger.error(f"DashScope API error: {response.message}")
                return ""
        except Exception as e:
            logger.error(f"DashScope call failed: {e}")
            return ""
    
    def embed(self, text: str) -> List[float]:
        try:
            import dashscope
            dashscope.api_key = self.api_key
            response = dashscope.TextEmbedding.call(
                model=dashscope.TextEmbedding.Models.text_embedding_v1,
                input=text
            )
            if response.status_code == 200:
                return response.output.embeddings[0].embedding
            else:
                logger.error(f"DashScope embedding error: {response.message}")
                return []
        except Exception as e:
            logger.error(f"DashScope embedding failed: {e}")
            return []
    
    async def async_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)


class OpenAIProvider:
    """OpenAI LLM提供者"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 2000)
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client
    
    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            return ""
    
    def embed(self, text: str) -> List[float]:
        client = self._get_client()
        try:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return []
    
    async def async_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)


class AzureOpenAIProvider:
    """Azure OpenAI LLM提供者"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key")
        self.azure_endpoint = config.get("azure_endpoint")
        self.azure_deployment = config.get("azure_deployment")
        self.api_version = config.get("api_version", "2024-02-15-preview")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 2000)
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
            )
        return self._client
    
    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        try:
            response = client.chat.completions.create(
                model=self.azure_deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Azure OpenAI call failed: {e}")
            return ""
    
    def embed(self, text: str) -> List[float]:
        client = self._get_client()
        try:
            response = client.embeddings.create(
                input=text,
                model=self.azure_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Azure OpenAI embedding failed: {e}")
            return []
    
    async def async_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)


class MockProvider:
    """Mock LLM提供者（用于测试）"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def generate(self, prompt: str, **kwargs) -> str:
        return f"Mock response for: {prompt[:50]}..."
    
    def embed(self, text: str) -> List[float]:
        return [0.1] * 384
    
    async def async_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)


class LLMFactory:
    """LLM提供者工厂"""
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: type):
        """注册LLM提供者"""
        cls._providers[name] = provider_class
        logger.info(f"Registered LLM provider: {name}")
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> LLMProvider:
        """创建LLM提供者实例"""
        provider_type = config.get("type", "dashscope")
        
        if provider_type not in cls._providers:
            raise ValueError(f"Unknown LLM provider type: {provider_type}")
        
        provider_class = cls._providers[provider_type]
        logger.info(f"Creating LLM provider: {provider_type}")
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """获取可用的提供者列表"""
        return list(cls._providers.keys())


LLMFactory.register("dashscope", DashScopeProvider)
LLMFactory.register("openai", OpenAIProvider)
LLMFactory.register("azure", AzureOpenAIProvider)
LLMFactory.register("mock", MockProvider)


def get_llm_provider(config: Optional[Dict[str, Any]] = None) -> LLMProvider:
    """获取LLM提供者单例"""
    global _llm_provider
    if _llm_provider is None:
        if config is None:
            from ..config import get_settings
            settings = get_settings()
            config = {
                "type": settings.llm_provider,
                "api_key": settings.dashscope_api_key,
                "base_url": settings.dashscope_base_url,
                "model": settings.dashscope_model,
                "temperature": settings.dashscope_temperature,
                "max_tokens": settings.dashscope_max_tokens,
            }
        _llm_provider = LLMFactory.create(config)
    return _llm_provider


_llm_provider: Optional[LLMProvider] = None