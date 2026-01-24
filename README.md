# Toy CLI

<div align='center'>
    <img src="./images/kimi.png" alt="alt text" width="70%">
    <p>Kimi-K2 ToDo Tools</p>
</div>

本地 lmstudio 部署的 Qwen3-4B Q4 量化的小模型也可以实现工具调用，但在使用todo工具时，效果不佳，建议使用更大size的模型。比如如果你手里有 AMD 395 Max 或是 Nvidia DGX Spark 可以尝试更大的模型。

<div align='center'>
    <img src="./images/qwen3-4b.png" alt="alt text" width="75%">
    <img src="./images/bad-case.png" alt="alt text" width="75%">
</div>

AgentCPM 这个 4B 的模型让我非常的惊艳，在本地部署的情况下，也能实现使用ToDo Manage管理工具调用和多轮交互。即使是在Q4量化的情况下，也能有不错的效果，比起之前尝试的 Qwen3-32B-A3B 模型，效果好了太多。

<div align='center'>
    <img src="./images/agentcpm.png" alt="alt text" width="100%">
    <p>AgentCPM ToDo Tools</p>
</div>


这是一个简化的 Claude Code 风格的代码 Agent，展示了工具调用和 Agent 循环的核心概念.

## 仓库结构

```
toy-cli/
├── agent.py      # Agent 核心实现，包含 agent loop
├── tools.py      # 可用工具集合（bash、文件操作、todo 管理）
├── llm.py        # LLM 接口抽象（支持本地和云端模型）
├── utils.py      # 工具函数（将 Python 函数转换为 JSON Schema）
├── prompt.py     # 系统提示词
└── .env          # API 密钥配置
```

## 核心概念

### 工具调用的重要性

工具调用（Tool Calling）是 Agent 从"聊天机器人"进化为"行动者"的关键：

```
普通聊天模型:         Agent with Tools:
输入 -> 思考 -> 输出    输入 -> 思考 -> 调用工具 -> 获取结果 -> 继续思考 -> ... -> 输出
                         ↑
                    能够与真实世界交互
```

通过工具，Agent 可以：
- 执行 shell 命令
- 读写文件
- 获取系统信息
- 管理任务列表

### Agent Loop - 核心

Agent Loop 是 Claude Code 等代码 Agent 的核心架构：

```python
# 伪代码展示核心逻辑
while True:
    # 1. 获取模型响应
    response = llm.chat(messages, tools)

    # 2. 处理文本回复
    if response.text:
        print(response.text)
        messages.append({"role": "assistant", "content": response.text})

    # 3. 处理工具调用
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call.name, tool_call.args)
            messages.append({"role": "tool", "content": result})

    # 4. 如果没有工具调用，退出循环
    if not response.tool_calls:
        break
```

这个循环让 Agent 能够：
1. **自主决策**：根据当前状态决定下一步行动
2. **多步推理**：通过工具调用链完成复杂任务
3. **上下文累积**：将所有交互保持在历史记录中

### Todo 管理

TodoWrite 工具，让 Agent 能够：

```python
# Agent 可以创建和管理任务列表
[
    {"content": "分析代码结构", "status": "pending", "activeForm": "正在分析代码结构"},
    {"content": "实现新功能", "status": "in_progress", "activeForm": "正在实现新功能"},
    {"content": "编写测试", "status": "completed", "activeForm": "编写了测试"}
]
```

显示效果：
```
[ ] 分析代码结构
[>] 实现新功能 <- 正在实现新功能
[x] 编写了测试

(1/3 completed)
```

**约束即赋能**：Todo 的约束（最多 20 项、只能一个 in_progress）反而让 Agent 能够：
- 聚焦单一任务
- 追踪整体进度
- 避免迷失方向

## 使用方法

```bash
# 安装依赖
pip install requests python-dotenv chardet

# 配置 .env 文件
echo "Siliconflow_API_KEY=your_key_here" > .env
echo "DeepSeek_API_KEY=your_key_here" > .env

# 运行
python agent.py
```

> 注意：如果你没有 Siliconflow API Key，可以使用 LocalLLM 连接本地模型。也可以在 [Siliconflow](https://cloud.siliconflow.cn/i/ybUFvmqK) 平台申请试用。

## 致谢

本代码大部分参考 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) 仓库。

**ShareAI-Lab** 是一个富有想象力和创新精神的公司。如果你对 Agent 感兴趣，欢迎加入他们！

- [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)
- [MoonshotAI/kimi-cli](https://github.com/MoonshotAI/kimi-cli)