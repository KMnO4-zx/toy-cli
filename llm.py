import requests
import json
import os, time
import pprint

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


class BaseLLM:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        self.model = model

    def get_response(self, messages: list = None, tools: list = None, **kwargs) -> dict:

        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            **{k: v for k, v in kwargs.items()}
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    if response:
                        try:
                            error_data = response.json()
                            print(f"Error response: {error_data}")
                        except:
                            print(f"Response content: {response.text}")
                    print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    if response:
                        try:
                            error_data = response.json()
                            pprint.pprint(error_data)
                        except:
                            print(f"Response content: {response.text}")
                    print(f"Request failed after {max_retries} attempts: {e}")
                    raise

        # 如果所有重试都失败了，返回一个空的响应或抛出异常
        raise requests.exceptions.RequestException("All retry attempts failed")
    
class SiliconflowLLM(BaseLLM):

    def __init__(self, api_key: str = None, model: str = "moonshotai/Kimi-K2-Instruct-0905"):
        self.api_key = api_key if api_key else os.getenv("Siliconflow_API_KEY")
        self.base_url = "https://api.siliconflow.cn/v1"
        self.model = model

class LocalLLM(BaseLLM):
    def __init__(self, api_key: str = None, model: str = "qwen/qwen3-4b-2507"):
        super().__init__(api_key, model)
        self.api_key = api_key if api_key else "xxxxxxx"
        self.base_url = "http://192.168.196.125:1234/v1"  
        self.model = model