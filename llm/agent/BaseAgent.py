import logging
import time
import random
import threading
from datetime import datetime
import openai
import re
from config import Config
from langchain_core.output_parsers import StrOutputParser

# Global concurrency control: limit max concurrent LLM API requests at any time
_LLM_SEMAPHORE = threading.Semaphore(3)

class BaseAgent:

    # ---- Global token accumulation (thread-safe) ----
    _total_tokens: dict = {}          # {model_name: {'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int}}
    _token_lock = threading.Lock()

    @classmethod
    def get_total_token_usage(cls) -> dict:
        """Return accumulated token usage, format: {model: {prompt_tokens, completion_tokens, total_tokens}, 'total': {...}}"""
        with cls._token_lock:
            result = {m: dict(v) for m, v in cls._total_tokens.items()}
        # Calculate totals
        total = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        for v in result.values():
            total['prompt_tokens'] += v['prompt_tokens']
            total['completion_tokens'] += v['completion_tokens']
            total['total_tokens'] += v['total_tokens']
        result['total'] = total
        return result

    @classmethod
    def reset_token_count(cls):
        """Reset global token counter"""
        with cls._token_lock:
            cls._total_tokens.clear()

    def __init__(self, url=None):
        self.output_parser = StrOutputParser()
        
        # If no url provided, read from config
        if url is None:
            url = Config.OPENAI_API_BASE + "/chat/completions" if Config.OPENAI_API_BASE else ""
        
        # Keep url parameter for subclass compatibility
        self.url = url
        
        # Create OpenAI client
        self.client = openai.OpenAI(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_API_BASE,
            timeout=120,
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_llm_response(self, system_prompt, user_prompt):
        """
        Call LLM via OpenAI SDK, get token stats directly from response usage field.

        Returns: (content, usage) tuple
            - content: str, LLM response text
            - usage: dict {'prompt_tokens', 'completion_tokens', 'total_tokens'} or None
        Returns (None, None) on failure.
        """
        max_retries = 5
        base_delay = 2  # Base delay 2 seconds

        for attempt in range(max_retries):
            _LLM_SEMAPHORE.acquire()
            try:
                print('Sending request to LLM...openai.chat.completions.create')
                response = self.client.chat.completions.create(
                    model=Config.MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,
                )
                
                # Get token stats from OpenAI response usage attribute
                usage = None
                if response.usage:
                    usage = {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens,
                    }
                
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content is not None:
                        return content, usage
                
                self.logger.error(f"Unexpected response format: {response}")
                return None, None
                
            except openai.APIStatusError as http_err:
                status_code = getattr(http_err, 'status_code', None)
                # 429 (rate limit) and 5xx (server errors) trigger retry, other 4xx do not retry
                if status_code is not None and (status_code == 429 or 500 <= status_code < 600):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        self.logger.warning(
                            f"LLM request failed (HTTP {status_code}), retry {attempt + 1}/{max_retries}, "
                            f"waiting {delay:.2f}s, reason: {http_err}"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error(
                            f"LLM request ultimately failed after {max_retries} retries, HTTP error: {http_err}"
                        )
                        return None, None
                else:
                    self.logger.error(f"HTTP error (no retry): {http_err}")
                    return None, None
            except (openai.APITimeoutError, openai.APIConnectionError) as req_err:
                # Timeout and network errors also trigger retry
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    self.logger.warning(
                        f"LLM request exception, retry {attempt + 1}/{max_retries}, "
                        f"waiting {delay:.2f}s, reason: {req_err}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    self.logger.error(
                        f"LLM request ultimately failed after {max_retries} retries, request exception: {req_err}"
                    )
                    return None, None
            except Exception as e:
                self.logger.error(f"Unknown error occurred (no retry): {e}")
                return None, None
            finally:
                _LLM_SEMAPHORE.release()

        self.logger.error(f"All LLM request retries failed, total {max_retries} attempts")
        return None, None

    def analyze_with_prompt(self, system_prompt: str, formatted_prompt: str):
        try:
            self.logger.info("Starting analysis")
            start_time = datetime.now()
            
            # Show processing status
            print(f"[{start_time.strftime('%H:%M:%S')}] Calling LLM...")
            
            # Call LLM and get API usage info
            result, usage = self.get_llm_response(system_prompt, formatted_prompt)
            
            if result is None:
                raise Exception("LLM request failed, response is empty")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Get token stats from OpenAI response usage field
            if usage:
                input_tokens = usage['prompt_tokens']
                output_tokens = usage['completion_tokens']
                total_tokens = usage['total_tokens']
                # Accumulate to global counter (thread-safe, bucketed by model)
                model_name = Config.MODEL or "unknown"
                with BaseAgent._token_lock:
                    if model_name not in BaseAgent._total_tokens:
                        BaseAgent._total_tokens[model_name] = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                    BaseAgent._total_tokens[model_name]['prompt_tokens'] += input_tokens
                    BaseAgent._total_tokens[model_name]['completion_tokens'] += output_tokens
                    BaseAgent._total_tokens[model_name]['total_tokens'] += total_tokens
            else:
                # Fallback: if API did not return usage, mark as unknown
                input_tokens = output_tokens = total_tokens = -1
                self.logger.warning("API response does not contain usage info, token stats unavailable")
            
            print(f"[TOKEN] Input Prompt Token count: {input_tokens}")
            self.logger.info(f"Input token count: {input_tokens}")
            
            self.logger.info(f"LLM call completed - duration: {duration:.2f}s, response length: {len(str(result))} chars")
            self.logger.info(f"Token usage - input: {input_tokens}, output: {output_tokens}, total: {total_tokens}")
            
            print(f"[{end_time.strftime('%H:%M:%S')}] LLM call completed, took {duration:.2f}s")
            print(f"[TOKEN] Output Token count: {output_tokens}")
            print(f"[TOKEN] Total Token count: {total_tokens}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            print(f"[ERROR] LLM call failed: {str(e)}")
            raise
    
    def clean_json_string(self, json_str):
        """
        Clean common formatting errors in JSON strings
        
        Args:
            json_str (str): JSON string to clean
            
        Returns:
            str: Cleaned JSON string
        """
        try:
            # Remove possible Markdown code block markers
            json_str = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r'^```\s*$', '', json_str, flags=re.MULTILINE)
            
            # Remove trailing commas (commas before } or ])
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)
            
            # Remove extra newlines and whitespace
            json_str = json_str.strip()
            
            return json_str
        except Exception as e:
            self.logger.warning(f"Error cleaning JSON string: {e}")
            return json_str
