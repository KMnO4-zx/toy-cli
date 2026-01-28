import requests
import json
import os, time
import pprint

from dotenv import load_dotenv, find_dotenv
# 自动查找并加载 .env 文件；使用 'utf-8-sig' 编码是为了自动兼容并剥离 
# Windows PowerShell 默认可能产生的 BOM (Byte Order Mark) 字符，确保 Key 读取正确。
_ = load_dotenv(find_dotenv(), encoding='utf-8-sig')


class BaseLLM:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        self.model = model

    def get_response(self, messages: list = None, tools: list = None, **kwargs) -> dict:
        """
        获取完整的LLM响应（非流式）
        """
        return self._make_request(messages, tools, stream=False, **kwargs)
    
    def get_streaming_response(self, messages: list = None, tools: list = None, **kwargs):
        """
        获取流式LLM响应
        返回一个生成器，每次yield一个token或完整的工具调用
        """
        return self._make_request(messages, tools, stream=True, **kwargs)
    
    def _make_request(self, messages: list = None, tools: list = None, stream: bool = False, **kwargs):
        """
        统一的请求方法，支持流式和非流式
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
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
                if stream:
                    response = requests.post(url, json=payload, headers=headers, stream=True)
                    response.raise_for_status()
                    return self._handle_streaming_response(response)
                else:
                    response = requests.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    if 'response' in locals() and response:
                        try:
                            error_data = response.json()
                            print(f"Error response: {error_data}")
                        except:
                            print(f"Response content: {response.text}")
                    print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    if 'response' in locals() and response:
                        try:
                            error_data = response.json()
                            pprint.pprint(error_data)
                        except:
                            print(f"Response content: {response.text}")
                    print(f"Request failed after {max_retries} attempts: {e}")
                    raise

        # 如果所有重试都失败了，返回一个空的响应或抛出异常
        raise requests.exceptions.RequestException("All retry attempts failed")
    
    def _handle_streaming_response(self, response):
        """
        处理流式响应，解析SSE格式
        """
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # 去掉 'data: ' 前缀
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        continue
    
class SiliconflowLLM(BaseLLM):

    def __init__(self, api_key: str = None, model: str = "moonshotai/Kimi-K2-Instruct-0905"):
        self.api_key = api_key if api_key else os.getenv("Siliconflow_API_KEY")
        self.base_url = "https://api.siliconflow.cn/v1"
        self.model = model
        self.platform = "siliconflow"


class DeepSeekLLM(BaseLLM):
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key if api_key else os.getenv("DeepSeek_API_KEY")
        self.base_url = "https://api.deepseek.com"
        self.model = model
        self.platform = "DeepSeek"


class LocalLLM(BaseLLM):
    def __init__(self, api_key: str = None, model: str = "agentcpm-explore@f16"):
        super().__init__(api_key, model)
        self.api_key = api_key if api_key else "xxxxxxx"
        # self.base_url = "http://192.168.1.5:1234/v1"
        self.base_url = "http://127.0.0.1:1234/v1"
        self.model = model
        self.platform = "LMStudio"