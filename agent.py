import json
from utils import function_to_json
from prompt import SYSTEM_PROMPT
from tools import safe_path, run_bash, run_read, run_write, run_edit, get_real_time
from llm import BaseLLM, SiliconflowLLM, LocalLLM
import pprint

# ANSI 颜色代码
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
GRAY = "\033[90m"
RESET = "\033[0m"


class Agent:
    def __init__(self, llm: BaseLLM = None):
        self.llm = llm if llm else SiliconflowLLM()
        self.tool_jsons, self.tool_map = self._load_tools()
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self._info_print()

    def _load_tools(self):
        tools = [
            run_bash, run_read, run_write, run_edit, get_real_time
        ]
        tool_jsons = [function_to_json(tool) for tool in tools]
        tool_map = {tool.__name__: tool for tool in tools}
        return tool_jsons, tool_map

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name not in self.tool_map:
            return f"Error: Tool {tool_name} not found"

        tool_func = self.tool_map[tool_name]
        try:
            result = tool_func(**arguments)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
        
    def _assistant_print(self, content: str):
        print(f"{BLUE}Assistant:{RESET} {content}")

    def _tool_print(self, content: str):
        print(f"{GRAY}Tool: {content}{RESET}")

    def _info_print(self):
        content = f'Using model: {self.llm.model} || Platform: {getattr(self.llm, "platform", "Unknown")}'
        print(f"{GREEN}Info: {content} {RESET}")
        
    def response_loop(self, user_input: str, history: list = None) -> list:
        """
        仅处理单轮响应，直到不存在工具调用
        """
        history = history if history is not None else []

        messages = history + [{"role": "user", "content": user_input}]

        while True:
            response = self.llm.get_response(
                messages=messages,
                tools=self.tool_jsons,
                temperature=1.0,
                max_tokens=4000,
            )

            response_message = response['choices'][0]['message']

            # 模型文本回复
            if "content" in response_message and response_message["content"]:
                messages.append({"role": "assistant", "content": response_message["content"]})
                self._assistant_print(response_message["content"])

            # 如果存在工具 处理工具调用
            if "tool_calls" in response_message and response_message["tool_calls"]:
                for tool_call in response_message["tool_calls"]:
                    tool_call_id = tool_call["id"]
                    function = tool_call["function"]
                    tool_name = function["name"]
                    arguments = json.loads(function["arguments"])

                    self._tool_print(f"Calling tool: {tool_name} with args {arguments}")
                    tool_result = self._execute_tool(tool_name, arguments)
                    self._tool_print(f"Tool result: {tool_result}")

                    # 工具结果加入历史
                    messages.append({
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tool_call_id
                    })
                # 下一次循环不需要新的 user_input，用历史继续
                user_input = None
            else:
                return messages
        
    def loop(self):

        history = [
                {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        while True:
            user_input = input(f"{YELLOW}User:{RESET} ")

            if user_input.lower() in {"exit", "quit"}:
                print("Exiting.")
                break

            history = self.response_loop(user_input, history)

            
    

if __name__ == "__main__":
    llm = LocalLLM()
    agent = Agent()

    agent.loop()