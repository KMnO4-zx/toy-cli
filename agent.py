import json
from utils import function_to_json
from prompt import SYSTEM_PROMPT
from tools import safe_path, run_bash, run_read, run_write, run_edit, get_real_time, run_todo, run_powershell
from llm import BaseLLM, DeepSeekLLM, SiliconflowLLM, LocalLLM

# ANSI 颜色代码
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
GRAY = "\033[90m"
RESET = "\033[0m"
RED = "\033[91m"


class Agent:
    def __init__(self, llm: BaseLLM = None, use_todo: bool = True, streaming: bool = True):
        self.llm = llm if llm else SiliconflowLLM()
        self.use_todo = use_todo
        self.streaming = streaming
        self.tool_jsons, self.tool_map = self._load_tools()
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self._info_print()
        self.rounds_without_todo = 0  # 跟踪多少轮没有使用 todo

    def _is_reasoner(self) -> bool:
        return "reasoner" in (self.llm.model or "").lower()

    def _strip_reasoning_from_history(self, history: list) -> list:
        """
        DeepSeek reasoner: reasoning_content 只用于“工具调用链”内部；
        一旦进入新的用户问题，要从历史里移除 reasoning_content，否则可能 400。
        """
        if not history:
            return history
        cleaned = []
        for m in history:
            if isinstance(m, dict) and "reasoning_content" in m:
                mm = dict(m)
                mm.pop("reasoning_content", None)
                cleaned.append(mm)
            else:
                cleaned.append(m)
        return cleaned

    def _load_tools(self):
        tools = [
            run_powershell, run_bash, run_read, run_write, run_edit, get_real_time
        ]
        if self.use_todo:
            tools.append(run_todo)

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
        if self.use_todo:
            # 替换 function_to_json 生成的简单 schema 为详细的 schema
            for i, tool_json in enumerate(tool_jsons):
                if tool_json["function"]["name"] == "run_todo":
                    tool_jsons[i] = todo_schema
                    break

        tool_map = {tool.__name__: tool for tool in tools}
        return tool_jsons, tool_map

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name not in self.tool_map:
            return f"Error: Tool {tool_name} not found"

        tool_func = self.tool_map[tool_name]
        try:
            result = tool_func(**arguments)
            # 保证工具返回永远是字符串
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def _process_tool_calls(self, tool_calls_data, used_todo_flag):
        """
        处理工具调用，返回工具消息列表和是否使用了todo
        """
        messages_additions = []
        used_todo = used_todo_flag
        
        for tool_call in tool_calls_data:
            tool_call_id = tool_call.get('id', '')
            function = tool_call.get('function', {})
            tool_name = function.get('name', '')

            try:
                arguments = json.loads(function.get('arguments', '{}') or '{}')
            except json.JSONDecodeError:
                print(f"{RED}模型调用参数解析失败: {function.get('arguments','')}{RESET}")
                arguments = {}
            
            self._tool_print(f"Calling tool: {tool_name}")
            tool_result = self._execute_tool(tool_name, arguments)
            self._tool_print(f"Tool result: {tool_result}")

            # 跟踪 todo 使用
            if tool_name == "run_todo":
                used_todo = True

            # 工具结果加入历史
            messages_additions.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_call_id
            })
        
        return messages_additions, used_todo
    
    def _update_rounds_counter(self, used_todo):
        """
        更新没有使用todo的轮数计数器
        """
        if used_todo:
            self.rounds_without_todo = 0
        else:
            self.rounds_without_todo += 1
        
    def _assistant_print(self, content: str):
        if self.streaming:
            print(content, end="", flush=True)
        else:
            print(f"{BLUE}Assistant:{RESET} {content}")

    def _tool_print(self, content: str):
        print(f"{GRAY}Tool: {content}{RESET}")

    def _info_print(self):
        content = f'Using model: {self.llm.model} || Platform: {getattr(self.llm, "platform", "Unknown")}'
        print(f"{GREEN}Info: {content} {RESET}")
        
    def response_loop(self, user_input: str, history: list = None) -> tuple:
        """
        处理一轮对话，直到不存在工具调用
        返回: (messages, used_todo)
        """
        history = history if history is not None else []

        # ✅ 关键：新的 user 问题开始前，清掉 history 里的 reasoning_content（reasoner 要求）
        if self._is_reasoner():
            history = self._strip_reasoning_from_history(history)

        messages = history + [{"role": "user", "content": user_input}]

        if self.streaming:
            return self._response_loop_streaming(messages)
        else:
            return self._response_loop_non_streaming(messages)
    
    def _response_loop_non_streaming(self, messages: list) -> tuple:
        """
        非流式响应处理
        """
        used_todo_this_turn = False

        while True:
            response = self.llm.get_response(
                messages=messages,
                tools=self.tool_jsons,
                temperature=1.0,
                max_tokens=4000,
            )

            response_message = response['choices'][0]['message']

            tool_calls = response_message.get("tool_calls") or []
            content = response_message.get("content") or ""
            reasoning = response_message.get("reasoning_content") or ""

            # ✅ 关键：无论 content 是否为空，只要有 tool_calls，也要 append assistant(tool_calls)
            assistant_msg = {"role": "assistant", "content": content}
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning  # ✅ 单独字段回传
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if content:
                self._assistant_print(content)
            
            if reasoning:
                self._assistant_print(reasoning)

            if tool_calls:
                tool_messages, used_todo_this_turn = self._process_tool_calls(
                    tool_calls,
                    used_todo_this_turn
                )
                messages.extend(tool_messages)
            else:
                # 更新计数器
                self._update_rounds_counter(used_todo_this_turn)
                return messages, used_todo_this_turn
    
    def _response_loop_streaming(self, messages: list) -> tuple:
        """
        流式响应处理
        """
        used_todo_this_turn = False

        while True:
            # 收集完整的响应内容
            full_content = ""
            full_reasoning = ""  # ✅ 单独累积 reasoning_content
            tool_calls = []
            
            # 开始流式响应
            print(f"{BLUE}Assistant:{RESET} ", end="", flush=True)
            
            for chunk in self.llm.get_streaming_response(
                messages=messages,
                tools=self.tool_jsons,
                temperature=1.0,
                max_tokens=4000,
            ):
                if 'choices' not in chunk or not chunk['choices']:
                    continue
                    
                delta = chunk['choices'][0].get('delta', {})

                # ✅ 推理内容：只累积，不混进 content
                if 'reasoning_content' in delta and delta['reasoning_content']:
                    full_reasoning += delta['reasoning_content']
                    # 如需调试输出可打开下一行（但不建议默认打印）
                    self._assistant_print(f"{delta['reasoning_content']}")

                # 文本内容
                if 'content' in delta and delta['content']:
                    content_piece = delta['content']
                    full_content += content_piece
                    self._assistant_print(content_piece)
                
                # 工具调用
                if 'tool_calls' in delta and delta['tool_calls']:
                    for tool_call_delta in delta['tool_calls']:
                        index = tool_call_delta.get('index', 0)
                        
                        # 如果是新的工具调用
                        if index >= len(tool_calls):
                            tool_calls.append({
                                'id': tool_call_delta.get('id', ''),
                                'type': 'function',
                                'function': {
                                    'name': '',
                                    'arguments': ''
                                }
                            })
                        
                        # 更新工具调用信息
                        if 'function' in tool_call_delta:
                            func_delta = tool_call_delta['function']

                            # ✅ name 不用 +=，避免拼坏
                            if 'name' in func_delta and func_delta['name']:
                                if not tool_calls[index]['function']['name']:
                                    tool_calls[index]['function']['name'] = func_delta['name']

                            # arguments 可能分片，继续 +=
                            if 'arguments' in func_delta and func_delta['arguments']:
                                tool_calls[index]['function']['arguments'] += func_delta['arguments']
            
            # 流式输出结束，换行
            print()
            
            # ✅ 无论 full_content 是否为空，都 append assistant；tool_calls 存在则附加
            assistant_msg = {"role": "assistant", "content": full_content or ""}
            if full_reasoning:
                assistant_msg["reasoning_content"] = full_reasoning  # ✅ 单独字段回传
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if tool_calls:
                # 使用辅助方法处理工具调用
                tool_messages, used_todo_this_turn = self._process_tool_calls(
                    tool_calls,
                    used_todo_this_turn
                )
                messages.extend(tool_messages)
            else:
                # 更新计数器
                self._update_rounds_counter(used_todo_this_turn)
                return messages, used_todo_this_turn
        
    def loop(self):
        history = [{"role": "system", "content": self.SYSTEM_PROMPT}]
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

            # 如果有额外提示消息，加入历史信息
            if content:
                history.append({"role": "user", "content": content})

            history, _ = self.response_loop(user_input, history)

            
    

if __name__ == "__main__":
    # DeepSeek reasoner：
    # llm = DeepSeekLLM(model="deepseek-reasoner")
    # llm = DeepSeekLLM(model="deepseek-chat")
    # 其他：
    # llm = SiliconflowLLM(model="deepseek-ai/DeepSeek-V3.2")
    llm = LocalLLM(model="agentcpm-explore@q4_k_m")

    agent = Agent(llm=llm, use_todo=True)

    agent.loop()