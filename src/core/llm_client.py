import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import logging

import openai
import anthropic
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"

@dataclass
class LLMResponse:
    """Response from an LLM API call"""
    content: str
    provider: str
    model: str
    tokens_used: int
    response_time_ms: int
    finish_reason: str
    raw_response: Optional[Dict[str, Any]] = None

@dataclass
class LLMMessage:
    """Message for LLM conversation"""
    role: str  # system, user, assistant
    content: str
    
class LLMClient:
    """Universal client for various LLM providers"""
    
    def __init__(self, 
                 provider: str = "openai",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 3):
        
        self.provider = ModelProvider(provider.lower())
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(f"llm_client_{provider}")
        
        # Initialize provider-specific clients
        self._init_client(api_key, base_url)
    
    def _init_client(self, api_key: Optional[str], base_url: Optional[str]):
        """Initialize the provider-specific client"""
        try:
            if self.provider == ModelProvider.OPENAI:
                self.client = openai.AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.timeout
                )
            elif self.provider == ModelProvider.ANTHROPIC:
                self.client = anthropic.AsyncAnthropic(
                    api_key=api_key,
                    timeout=self.timeout
                )
            elif self.provider == ModelProvider.GOOGLE:
                if api_key:
                    genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self.model)
            elif self.provider == ModelProvider.AZURE:
                self.client = openai.AsyncAzureOpenAI(
                    api_key=api_key,
                    api_version="2024-02-01",
                    azure_endpoint=base_url,
                    timeout=self.timeout
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.provider.value} client: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_response(self,
                               messages: List[LLMMessage],
                               temperature: float = 0.1,
                               max_tokens: int = 2000,
                               **kwargs) -> LLMResponse:
        """Generate a response from the LLM"""
        start_time = time.time()
        
        try:
            if self.provider == ModelProvider.OPENAI:
                response = await self._call_openai(messages, temperature, max_tokens, **kwargs)
            elif self.provider == ModelProvider.ANTHROPIC:
                response = await self._call_anthropic(messages, temperature, max_tokens, **kwargs)
            elif self.provider == ModelProvider.GOOGLE:
                response = await self._call_google(messages, temperature, max_tokens, **kwargs)
            elif self.provider == ModelProvider.AZURE:
                response = await self._call_azure(messages, temperature, max_tokens, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            response_time = int((time.time() - start_time) * 1000)
            response.response_time_ms = response_time
            
            self.logger.info(f"LLM response generated: {self.provider.value}/{self.model} "
                           f"({response.tokens_used} tokens, {response_time}ms)")
            
            return response
            
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            raise
    
    async def _call_openai(self, 
                          messages: List[LLMMessage], 
                          temperature: float, 
                          max_tokens: int, 
                          **kwargs) -> LLMResponse:
        """Call OpenAI API"""
        formatted_messages = [
            {"role": msg.role, "content": msg.content} 
            for msg in messages
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            provider=self.provider.value,
            model=self.model,
            tokens_used=response.usage.total_tokens,
            response_time_ms=0,  # Will be set by caller
            finish_reason=response.choices[0].finish_reason,
            raw_response=response.model_dump()
        )
    
    async def _call_anthropic(self, 
                             messages: List[LLMMessage], 
                             temperature: float, 
                             max_tokens: int, 
                             **kwargs) -> LLMResponse:
        """Call Anthropic API"""
        # Convert messages to Anthropic format
        system_message = None
        conversation = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=conversation,
            **kwargs
        )
        
        return LLMResponse(
            content=response.content[0].text,
            provider=self.provider.value,
            model=self.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            response_time_ms=0,
            finish_reason=response.stop_reason,
            raw_response=response.model_dump()
        )
    
    async def _call_google(self, 
                          messages: List[LLMMessage], 
                          temperature: float, 
                          max_tokens: int, 
                          **kwargs) -> LLMResponse:
        """Call Google Gemini API"""
        # Convert messages to Google format
        conversation = []
        for msg in messages:
            if msg.role == "user":
                conversation.append({"parts": [{"text": msg.content}]})
            elif msg.role == "assistant":
                conversation.append({"role": "model", "parts": [{"text": msg.content}]})
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            **kwargs
        }
        
        # This is a simplified version - real implementation would need proper async handling
        response = self.client.generate_content(
            conversation,
            generation_config=generation_config
        )
        
        return LLMResponse(
            content=response.text,
            provider=self.provider.value,
            model=self.model,
            tokens_used=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
            response_time_ms=0,
            finish_reason="stop",
            raw_response={"response": response.text}
        )
    
    async def _call_azure(self, 
                         messages: List[LLMMessage], 
                         temperature: float, 
                         max_tokens: int, 
                         **kwargs) -> LLMResponse:
        """Call Azure OpenAI API"""
        formatted_messages = [
            {"role": msg.role, "content": msg.content} 
            for msg in messages
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            provider=self.provider.value,
            model=self.model,
            tokens_used=response.usage.total_tokens,
            response_time_ms=0,
            finish_reason=response.choices[0].finish_reason,
            raw_response=response.model_dump()
        )
    
    async def generate_structured_response(self,
                                         messages: List[LLMMessage],
                                         response_schema: Dict[str, Any],
                                         temperature: float = 0.1,
                                         max_tokens: int = 2000) -> Dict[str, Any]:
        """Generate a structured response that conforms to a schema"""
        # Add schema instruction to the last message
        if messages:
            schema_instruction = (
                f"\n\nPlease respond with a valid JSON object that conforms to this schema:\n"
                f"{json.dumps(response_schema, indent=2)}\n"
                f"Your response must be valid JSON and nothing else."
            )
            messages[-1].content += schema_instruction
        
        response = await self.generate_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        try:
            # Extract JSON from response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            parsed_response = json.loads(content)
            return parsed_response
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse structured response: {e}")
            self.logger.error(f"Raw response: {response.content}")
            raise ValueError(f"LLM did not return valid JSON: {e}")
    
    def create_messages(self, 
                       system_prompt: str = None,
                       user_prompt: str = None,
                       conversation_history: List[Dict[str, str]] = None) -> List[LLMMessage]:
        """Helper to create message list"""
        messages = []
        
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        
        if conversation_history:
            for msg in conversation_history:
                messages.append(LLMMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", "")
                ))
        
        if user_prompt:
            messages.append(LLMMessage(role="user", content=user_prompt))
        
        return messages
    
    async def test_connection(self) -> bool:
        """Test the connection to the LLM provider"""
        try:
            test_messages = [
                LLMMessage(role="user", content="Hello, please respond with 'OK'")
            ]
            
            response = await self.generate_response(
                messages=test_messages,
                max_tokens=10,
                temperature=0.0
            )
            
            self.logger.info(f"Connection test successful: {self.provider.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider configuration"""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        } 