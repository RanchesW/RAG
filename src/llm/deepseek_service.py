import requests
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Конфигурация генерации"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_ctx: int = 4096
    max_tokens: Optional[int] = None
    repeat_penalty: float = 1.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для API"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Декоратор для retry logic с экспоненциальным backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, requests.Timeout) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
                except Exception as e:
                    # Не retry для других типов ошибок
                    logger.error(f"Non-retryable error: {e}")
                    raise
            
            raise last_exception
        return wrapper
    return decorator

class DeepSeekService:
    def __init__(self, 
                 model_name: str = "deepseek-r1:14b",
                 ollama_url: str = "http://localhost:11434",
                 config: Optional[GenerationConfig] = None,
                 timeout: int = 120,
                 max_retries: int = 3,
                 **kwargs):
        
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip('/')
        self.config = config or GenerationConfig()
        self.timeout = timeout
        self.max_retries = max_retries
        
        logger.info(f"Initializing DeepSeek service with model: {model_name}")
        
        try:
            self._validate_connection()
            logger.info("✓ Ollama DeepSeek service ready")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            raise
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def _validate_connection(self):
        """Проверка подключения к Ollama с retry"""
        response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if self.model_name in model_names:
                logger.info(f"✓ Model {self.model_name} found in Ollama")
            else:
                logger.warning(f"Model {self.model_name} not found. Available: {model_names}")
                logger.info("Downloading model...")
                self._pull_model()
        else:
            raise requests.RequestException(f"Ollama not responding: {response.status_code}")
    
    def _pull_model(self) -> bool:
        """Загрузка модели с улучшенным логированием"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": self.model_name},
                stream=True,
                timeout=300
            )
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        status = data.get('status', '')
                        
                        if 'error' in data:
                            logger.error(f"Pull error: {data['error']}")
                            return False
                        
                        # Логируем прогресс
                        if 'completed' in data and 'total' in data:
                            percent = (data['completed'] / data['total']) * 100
                            print(f"📥 Downloading: {percent:.1f}% - {status}")
                        elif status:
                            print(f"📥 Pull status: {status}")
                            
                    except json.JSONDecodeError:
                        continue
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def generate(self, 
                 prompt: str, 
                 config: Optional[GenerationConfig] = None) -> str:
        """Генерация с улучшенной обработкой и retry logic"""
        
        # Валидация входных данных
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        generation_config = config or self.config
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": generation_config.to_dict()
        }
        
        print(f"🤖 Sending request to Ollama...")
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            raw_response = result.get("response", "")
            
            # Логируем метрики
            generation_time = time.time() - start_time
            prompt_tokens = result.get("prompt_eval_count", 0)
            completion_tokens = result.get("eval_count", 0)
            
            if prompt_tokens > 0 and completion_tokens > 0:
                tokens_per_sec = completion_tokens / generation_time if generation_time > 0 else 0
                print(f"📊 Generation: {completion_tokens} tokens in {generation_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
            
            print(f"📥 Raw Ollama response: {raw_response[:500]}...")
            
            # Улучшенная обработка ответа
            processed_response = self._process_response(raw_response)
            print(f"✨ Processed response: {processed_response[:200]}...")
            
            return processed_response
            
        else:
            error_msg = f"Ollama API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise requests.RequestException(error_msg)
    
    def _process_response(self, raw_response: str) -> str:
        """Улучшенная обработка и форматирование ответа"""
        if not raw_response:
            return "❓ **Ответ не найден**"
        
        import re
        
        # Удаляем все виды thinking тегов более надежно
        thinking_patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'<reflection>.*?</reflection>',
            r'<\|thinking\|>.*?<\|/thinking\|>',
            r'<внутренние размышления>.*?</внутренние размышления>'
        ]
        
        clean_response = raw_response
        for pattern in thinking_patterns:
            clean_response = re.sub(pattern, '', clean_response, flags=re.DOTALL)
        
        # Извлекаем основной ответ с несколькими паттернами
        answer_patterns = [
            (r'(?:Ответ|Answer):\s*(.*)', True),
            (r'(?:На основе документов|Based on documents).*?:\s*(.*)', True),
            (r'(?:Исходя из|Drawing from).*?:\s*(.*)', True),
            (r'(?:Вывод|Conclusion):\s*(.*)', True)
        ]
        
        extracted_answer = ""
        for pattern, use_group in answer_patterns:
            match = re.search(pattern, clean_response, re.DOTALL | re.IGNORECASE)
            if match and use_group:
                extracted_answer = match.group(1).strip()
                if extracted_answer and len(extracted_answer) > 10:
                    break
        
        # Если паттерны не сработали, используем очищенный текст
        if not extracted_answer:
            extracted_answer = clean_response.strip()
        
        # Финальная очистка и форматирование
        final_answer = self._format_answer(extracted_answer)
        
        return final_answer if final_answer else "❌ **Не удалось сформировать ответ**"
    
    def _format_answer(self, answer: str) -> str:
        """Форматирование ответа для красивого отображения"""
        if not answer:
            return ""
        
        import re
        
        # Удаляем лишние пробелы и переносы
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        answer = re.sub(r' {2,}', ' ', answer)
        answer = answer.strip()
        
        # Разбиваем на абзацы
        paragraphs = [p.strip() for p in answer.split('\n') if p.strip()]
        
        formatted_parts = []
        
        for paragraph in paragraphs:
            # Проверяем разные типы контента
            if self._is_header(paragraph):
                formatted_parts.append(f"\n**{paragraph.rstrip(':')}:**\n")
            elif self._is_list_item(paragraph):
                clean_item = paragraph.strip('- •*').strip()
                formatted_parts.append(f"• {clean_item}")
            elif self._is_step(paragraph):
                formatted_parts.append(f"\n**{paragraph}**")
            else:
                # Обычный текст
                formatted_parts.append(paragraph)
        
        # Объединяем с правильными отступами
        result = '\n\n'.join(formatted_parts)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    def _is_header(self, text: str) -> bool:
        """Определяет заголовки"""
        text_lower = text.lower()
        header_keywords = [
            'инструкция', 'шаги', 'порядок', 'алгоритм', 'процедура',
            'требования', 'настройка', 'подключение', 'установка', 'конфигурация'
        ]
        return (any(keyword in text_lower for keyword in header_keywords) 
                and len(text) < 100 
                and not text.lower().startswith(('для', 'чтобы', 'если')))
    
    def _is_list_item(self, text: str) -> bool:
        """Определяет элементы списка"""
        import re
        return bool(re.match(r'^[-•*]\s+', text.strip()) or 
                   re.match(r'^\d+\.\s+', text.strip()))
    
    def _is_step(self, text: str) -> bool:
        """Определяет шаги инструкций"""
        import re
        text_lower = text.lower().strip()
        step_patterns = [
            r'^\d+\.',  # 1. 2. 3.
            r'^шаг \d+',  # Шаг 1
            r'^этап \d+',  # Этап 1
            r'^сначала\b',  # Сначала
            r'^затем\b',  # Затем  
            r'^далее\b',  # Далее
            r'^после этого\b',  # После этого
        ]
        return any(re.match(pattern, text_lower) for pattern in step_patterns)
    
    def create_rag_prompt(self, 
                         query: str, 
                         context: str, 
                         language: str = "russian",
                         style: str = "detailed",
                         max_context_length: int = 2000) -> str:
        """Улучшенное создание RAG промптов с разными стилями"""
        
        # Валидация входных данных
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Обрезаем контекст если слишком длинный
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        if style == "detailed":
            return self._create_detailed_prompt(query, context, language)
        else:
            return self._create_concise_prompt(query, context, language)
    
    def _create_detailed_prompt(self, query: str, context: str, language: str) -> str:
        """Детальный промпт с инструкциями"""
        
        if language.lower() == "russian":
            return f"""<thinking>
Пользователь задал вопрос: {query}

Мне нужно проанализировать предоставленный контекст и дать структурированный, полезный ответ на русском языке.

Контекст из документов:
{context}

Я должен найти релевантную информацию и представить её в удобном формате.
</thinking>

Ты - корпоративный ассистент по документации в Казахстане. Отвечай точно на основе предоставленного контекста.

КОНТЕКСТ ИЗ ДОКУМЕНТОВ:
{context}

ВОПРОС: {query}

ТРЕБОВАНИЯ К ОТВЕТУ:
- Отвечай ТОЛЬКО на основе контекста
- Структурируй ответ с заголовками и списками
- Для инструкций используй пошаговый формат
- Выделяй важные моменты
- Будь конкретным и полезным

Ответ:"""
        
        else:
            return f"""<thinking>
User asked: {query}

I need to analyze the provided context and give a structured, helpful answer.

Context from documents:
{context}

I should find relevant information and present it in a useful format.
</thinking>

You are a corporate documentation assistant in Kazakhstan. Answer accurately based on the provided context.

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {query}

ANSWER REQUIREMENTS:
- Answer ONLY based on context
- Structure with headers and lists
- Use step-by-step format for instructions
- Highlight important points
- Be specific and helpful

Answer:"""
    
    def _create_concise_prompt(self, query: str, context: str, language: str) -> str:
        """Краткий промпт"""
        
        if language.lower() == "russian":
            return f"""<thinking>
Вопрос: {query}
Контекст: {context}
Нужен краткий точный ответ на русском.
</thinking>

Контекст: {context}

Вопрос: {query}

Дай точный краткий ответ на основе контекста:

Ответ:"""
        
        else:
            return f"""<thinking>
Question: {query}
Context: {context}
Need a concise accurate answer.
</thinking>

Context: {context}

Question: {query}

Provide a concise accurate answer based on context:

Answer:"""
