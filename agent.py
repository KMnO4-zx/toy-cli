import json
from utils import function_to_json
from prompt import SYSTEM_PROMPT
from tools import safe_path, run_bash, run_read, run_write, run_edit, get_real_time, run_todo
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
        self.rounds_without_todo = 0  # 跟踪多少轮没有使用 todo

    def _load_tools(self):
        tools = [
            run_bash, run_read, run_write, run_edit, get_real_time, run_todo
        ]
        tool_jsons = [function_to_json(tool) for tool in tools]

        # run_todo 需要手动的详细 schema，因为 function_to_json 无法描述嵌套对象结构
        todo_schema = {
            "type": "function",
            "function": {
                "name": "run_todo",
                "description": "Update the task list. Use to plan and track progress.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "description": "Complete list of tasks (replaces existing)",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {
                                        "type": "string",
                                        "description": "Task description"
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "in_progress", "completed"],
                                        "description": "Task status"
                                    },
                                    "activeForm": {
                                        "type": "string",
                                        "description": "Present tense action, e.g. 'Reading files'"
                                    },
                                },
                                "required": ["content", "status", "activeForm"],
                            },
                        }
                    },
                    "required": ["items"],
                },
            },
        }

        tool_jsons.append(todo_schema)

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
        
    def response_loop(self, user_input: str, history: list = None) -> tuple:
        """
        仅处理单轮响应，直到不存在工具调用

        返回:
            tuple: (messages, used_todo) - 更新后的消息历史和是否使用了 todo
        """
        history = history if history is not None else []
        used_todo_this_turn = False

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

                    self._tool_print(f"Calling tool: {tool_name}")
                    tool_result = self._execute_tool(tool_name, arguments)
                    self._tool_print(f"Tool result: {tool_result}")

                    # 跟踪 todo 使用
                    if tool_name == "run_todo":
                        used_todo_this_turn = True

                    # 工具结果加入历史
                    messages.append({
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tool_call_id
                    })
                # 下一次循环不需要新的 user_input，用历史继续
                user_input = None
            else:
                # 更新计数器
                if used_todo_this_turn:
                    self.rounds_without_todo = 0
                else:
                    self.rounds_without_todo += 1

                return messages, used_todo_this_turn
        
    def loop(self):

        history = [
                {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        first_message = True

        while True:
            user_input = input(f"{YELLOW}User:{RESET} ")

            if user_input.lower() in {"exit", "quit"}:
                print("Exiting.")
                break

            # 构建用户消息内容，可能包含提醒
            content = []

            if first_message:
                # 在开始时给出温和的提醒
                content.append({"type": "text", "text": "<reminder>对于复杂任务，请使用 TodoWrite 工具来规划和跟踪进度。</reminder>"})
                first_message = False
            elif self.rounds_without_todo > 10:
                # 如果模型长时间没有使用 todo，发出提醒
                content.append({"type": "text", "text": "<reminder>已经 10+ 轮没有更新 todo 了，请更新任务列表。</reminder>"})

            content.append({"type": "text", "text": user_input})

            # 如果有额外内容（提醒），使用 content 数组；否则使用纯文本
            if len(content) > 1:
                user_message = {"role": "user", "content": content}

            history.append(user_message)

            history, _ = self.response_loop(user_input, history)

            
    

if __name__ == "__main__":
    llm = LocalLLM()
    agent = Agent(llm=llm)

    agent.loop()