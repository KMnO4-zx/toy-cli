import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

WORKDIR = Path.cwd()


def get_real_time() -> str:
    """
    Get the current system time in ISO 8601 format.
    """
    return datetime.now().isoformat()

def safe_path(p: str) -> Path:
    """
    Ensure path stays within workspace (security measure).

    Prevents the model from accessing files outside the project directory.
    Resolves relative paths and checks they don't escape via '../'.
    """
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """
    Execute shell command with safety checks.

    Security: Blocks obviously dangerous commands.
    Timeout: 60 seconds to prevent hanging.
    Output: Truncated to 50KB to prevent context overflow.
    """
    # Basic safety - block dangerous patterns
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=60
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out (60s)"
    except Exception as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    """
    Read file contents with optional line limit.

    For large files, use limit to read just the first N lines.
    Output truncated to 50KB to prevent context overflow.
    """
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()

        if limit and limit < len(lines):
            lines = lines[:limit]
            lines.append(f"... ({len(text.splitlines()) - limit} more lines)")

        return "\n".join(lines)[:50000]

    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """
    Write content to file, creating parent directories if needed.

    This is for complete file creation/overwrite.
    For partial edits, use edit_file instead.
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"

    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    Replace exact text in a file (surgical edit).

    Uses exact string matching - the old_text must appear verbatim.
    Only replaces the first occurrence to prevent accidental mass changes.
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()

        if old_text not in content:
            return f"Error: Text not found in {path}"

        # Replace only first occurrence for safety
        new_content = content.replace(old_text, new_text, 1)
        fp.write_text(new_content)
        return f"Edited {path}"

    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# TodoManager - 管理结构化任务列表
# =============================================================================

class TodoManager:
    """
    管理结构化任务列表，强制执行约束。

    设计原则:
    ---------
    1. 最多 20 项：防止模型创建无限列表
    2. 只能有一个 in_progress：强制专注 - 一次只能做一件事
    3. 必填字段：每项需要 content, status, activeForm

    activeForm 字段说明:
    -------------------
    - 正在发生的事情的现在时态形式
    - 当 status 为 "in_progress" 时显示
    - 示例: content="添加测试", activeForm="正在添加单元测试..."

    这提供了 agent 正在做什么的实时可见性。
    """

    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        """
        验证并更新 todo 列表。

        模型每次发送完整的新列表。我们验证它，存储它，
        并返回模型将看到的渲染视图。

        验证规则:
        - 每项必须有: content, status, activeForm
        - Status 必须是: pending | in_progress | completed
        - 只能有一项处于 in_progress 状态
        - 最多允许 20 项

        返回:
            todo 列表的渲染文本视图
        """
        validated = []
        in_progress_count = 0

        for i, item in enumerate(items):
            # 提取并验证字段
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).lower()
            active_form = str(item.get("activeForm", "")).strip()

            # 验证检查
            if not content:
                raise ValueError(f"Item {i}: content required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {i}: invalid status '{status}'")
            if not active_form:
                raise ValueError(f"Item {i}: activeForm required")

            if status == "in_progress":
                in_progress_count += 1

            validated.append({
                "content": content,
                "status": status,
                "activeForm": active_form
            })

        # 强制执行约束
        if len(validated) > 20:
            raise ValueError("Max 20 todos allowed")
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.items = validated
        return "\n" + self.render()

    def render(self) -> str:
        """
        将 todo 列表渲染为人类可读的文本。

        格式:
            [x] 已完成的任务
            [>] 进行中的任务 <- 正在做某事...
            [ ] 待处理的任务

            (2/3 已完成)

        这个渲染的文本是模型作为工具结果看到的。
        然后它可以根据其当前状态更新列表。
        """
        if not self.items:
            return "No todos."

        lines = []
        for item in self.items:
            if item["status"] == "completed":
                lines.append(f"[x] {item['content']}")
            elif item["status"] == "in_progress":
                lines.append(f"[>] {item['content']} <- {item['activeForm']}")
            else:
                lines.append(f"[ ] {item['content']}")

        completed = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({completed}/{len(self.items)} completed)")

        return "\n".join(lines)


# 全局 todo 管理器实例
TODO = TodoManager()


def run_todo(items: list) -> str:
    """
    更新 todo 列表。

    模型发送完整的新列表（不是差异）。
    我们验证它并返回渲染视图。
    """
    try:
        return TODO.update(items)
    except Exception as e:
        return f"Error: {e}"