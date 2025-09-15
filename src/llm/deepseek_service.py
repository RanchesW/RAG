from vllm import LLM, SamplingParams
import torch
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DeepSeekService:
    def __init__(self, 
                 model_name: str = "deepseek-ai/DeepSeek-V3-Base",
                 tensor_parallel_size: int = 2,
                 gpu_memory_utilization: float = 0.85):
        
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        
        logger.info(f"Loading DeepSeek model on {tensor_parallel_size} GPUs...")
        
        try:
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=4096,
                trust_remote_code=True,
                enforce_eager=False
            )
            
            # Russian-optimized sampling parameters
            self.default_sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=1000,
                stop=["\n\n", "Human:", "Человек:", "Пользователь:"]
            )
            
            logger.info("✓ DeepSeek model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load DeepSeek model: {e}")
            raise
    
    def generate(self, 
                 prompt: str, 
                 sampling_params: Optional[SamplingParams] = None) -> str:
        """Generate response for single prompt"""
        if sampling_params is None:
            sampling_params = self.default_sampling_params
        
        try:
            outputs = self.llm.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def create_rag_prompt(self, 
                         query: str, 
                         context: str, 
                         language: str = "russian") -> str:
        """Create RAG prompt template for Russian/Kazakh queries"""
        
        if language.lower() == "russian":
            template = f"""Ты - полезный ассистент для работы с корпоративными документами в Казахстане. 
Отвечай точно и полезно на основе предоставленного контекста.

Контекст из документов:
{context}

Вопрос пользователя: {query}

Дай подробный и точный ответ на русском языке на основе контекста:"""
        else:
            template = f"""You are a helpful assistant for corporate documents in Kazakhstan.
Answer accurately based on the provided context.

Context from documents:
{context}

User question: {query}

Answer:"""
        
        return template
