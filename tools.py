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
    Execute shell command with safety checks and basic encoding support.

    Security: Blocks obviously dangerous commands.
    Timeout: 60 seconds to prevent hanging.
    Output: Truncated to 50KB to prevent context overflow.
    
    Note: For PowerShell commands, use run_powershell() instead for better
    encoding support and PowerShell-specific features.
    
    Encoding handling:
    - On Windows: Tries UTF-8, falls back to system locale encoding
    - On Unix/Linux: Uses UTF-8
    """
    import platform
    
    # Basic safety - block dangerous patterns
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        # Use universal_newlines for better cross-platform compatibility
        # This lets Python handle the encoding automatically
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            universal_newlines=True,
            timeout=60
        )
        
        # Safely combine stdout and stderr
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(result.stderr)
        
        output = "".join(output_parts).strip()
        return output[:50000] if output else "(no output)"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out (60s)"
    except Exception as e:
        return f"Error: {e}"

def run_powershell(command: str) -> str:
    """
    Execute PowerShell command with proper encoding handling for Windows.
    
    This function is specifically designed for PowerShell commands and provides
    better encoding support and PowerShell-specific features than run_bash().
    
    Features:
    - Proper UTF-8 encoding for PowerShell output
    - Handles both PowerShell Core (pwsh) and Windows PowerShell
    - Supports Chinese/English output on Windows systems
    - Includes error handling for common PowerShell issues
    - Uses -NoProfile and -ExecutionPolicy Bypass for consistent behavior
    
    Usage examples:
    - run_powershell("Get-Process | Select-Object Name, CPU")
    - run_powershell("Get-ChildItem -Path . -Filter *.py")
    - run_powershell("$env:USERNAME")  # Get current username
    - run_powershell("Write-Output '中文测试'")  # Chinese output
    
    Note: This function automatically uses PowerShell's -EncodedCommand parameter
    to avoid quoting issues with complex commands.
    """
    import platform
    import subprocess
    
    if platform.system() != 'Windows':
        return "Error: PowerShell is only available on Windows systems"
    
    # PowerShell-specific safety checks
    dangerous_patterns = [
        "Remove-Item -Recurse -Force",
        "Format-Volume",
        "Stop-Computer",
        "Restart-Computer",
        "Invoke-Expression",
        "iex ",
        "Start-Process -Verb RunAs",  # Elevation
        "Remove-Variable",  # Could remove important variables
        "Clear-Host",  # Could clear console output
    ]
    
    command_lower = command.lower()
    for pattern in dangerous_patterns:
        if pattern.lower() in command_lower:
            return f"Error: Potentially dangerous PowerShell command blocked: {pattern}"
    
    try:
        # Construct the PowerShell command
        # Use -NoProfile for faster startup and cleaner environment
        # Use -ExecutionPolicy Bypass to avoid policy restrictions
        # For complex commands with quotes, use -EncodedCommand to avoid quoting issues
        import base64
        
        # Encode the command as UTF-16LE and then base64
        encoded_bytes = command.encode('utf-16le')
        encoded_command = base64.b64encode(encoded_bytes).decode('ascii')
        
        full_command = f'powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded_command}'
        
        # Use universal_newlines for automatic encoding handling
        result = subprocess.run(
            full_command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            universal_newlines=True,
            timeout=60
        )
        
        # Combine and format output
        output_parts = []
        
        if result.stdout:
            stdout = result.stdout.strip()
            if stdout:
                output_parts.append(stdout)
        
        if result.stderr:
            stderr = result.stderr.strip()
            if stderr:
                # Mark stderr clearly
                output_parts.append(f"[PowerShell Error] {stderr}")
        
        if output_parts:
            output = "\n".join(output_parts)
            return output[:50000] if len(output) > 50000 else output
        else:
            return "(PowerShell command executed successfully with no output)"
            
    except subprocess.TimeoutExpired:
        return "Error: PowerShell command timed out (60s)"
    except Exception as e:
        return f"Error executing PowerShell command: {str(e)}"



def run_read(path: str, limit: int = None, encoding: str = "utf-8") -> str:
    """
    Read file contents with optional line limit and encoding support.

    For large files, use limit to read just the first N lines.
    Output truncated to 50KB to prevent context overflow.
    
    Encoding support:
    - Defaults to UTF-8
    - Automatically detects common encodings (UTF-8, GB2312, GBK, UTF-16)
    - Can be explicitly specified via the encoding parameter
    
    Common encodings for Windows:
    - 'utf-8': Unicode UTF-8 (recommended for new files)
    - 'gb2312': Simplified Chinese (common on Chinese Windows)
    - 'gbk': Extended Chinese encoding
    - 'utf-16': Unicode UTF-16 (Windows Unicode)
    """
    import chardet
    
    try:
        fp = safe_path(path)
        
        # First try with the specified encoding
        try:
            text = fp.read_text(encoding=encoding)
        except UnicodeDecodeError:
            # If specified encoding fails, try to detect the encoding
            raw_data = fp.read_bytes()
            detected = chardet.detect(raw_data)
            detected_encoding = detected.get('encoding', 'utf-8')
            
            # Common encoding mappings for Windows
            if detected_encoding.lower() in ['gb2312', 'gbk', 'gb18030']:
                # Use GBK which is a superset of GB2312
                text = raw_data.decode('gbk', errors='replace')
            elif detected_encoding.lower() == 'ascii':
                text = raw_data.decode('utf-8', errors='replace')
            else:
                # Try with detected encoding, fall back to UTF-8 with replacement
                try:
                    text = raw_data.decode(detected_encoding, errors='replace')
                except:
                    text = raw_data.decode('utf-8', errors='replace')
        
        lines = text.splitlines()

        if limit and limit < len(lines):
            lines = lines[:limit]
            lines.append(f"... ({len(text.splitlines()) - limit} more lines)")

        return "\n".join(lines)[:50000]

    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str, encoding: str = "utf-8") -> str:
    """
    Write content to file with encoding support, creating parent directories if needed.

    This is for complete file creation/overwrite.
    For partial edits, use edit_file instead.
    
    Encoding support:
    - Defaults to UTF-8 (recommended for cross-platform compatibility)
    - Supports common Windows encodings: utf-8, gb2312, gbk, utf-16
    - Use 'utf-8-sig' for UTF-8 with BOM (compatible with some Windows applications)
    
    Best practices:
    - Use UTF-8 for new files to ensure cross-platform compatibility
    - Use GBK/GB2312 only when specifically needed for legacy systems
    - Include encoding parameter in the function call for clarity
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate encoding parameter
        valid_encodings = ['utf-8', 'utf-8-sig', 'gb2312', 'gbk', 'gb18030', 'utf-16', 'utf-16-le', 'utf-16-be']
        if encoding.lower() not in [e.lower() for e in valid_encodings]:
            # Default to UTF-8 if invalid encoding specified
            encoding = 'utf-8'
        
        fp.write_text(content, encoding=encoding)
        return f"Wrote {len(content.encode(encoding))} bytes to {path} (encoding: {encoding})"

    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str, encoding: str = None) -> str:
    """
    Replace exact text in a file (surgical edit) with encoding support.

    Uses exact string matching - the old_text must appear verbatim.
    Only replaces the first occurrence to prevent accidental mass changes.
    
    Encoding handling:
    - If encoding is specified, uses that encoding for both read and write
    - If encoding is None, tries to detect the file's current encoding
    - Falls back to UTF-8 if detection fails
    """
    import chardet
    
    try:
        fp = safe_path(path)
        
        # Read with appropriate encoding
        if encoding:
            # Use specified encoding
            content = fp.read_text(encoding=encoding)
        else:
            # Try to detect encoding
            raw_data = fp.read_bytes()
            detected = chardet.detect(raw_data)
            detected_encoding = detected.get('encoding', 'utf-8')
            
            # Common encoding mappings
            if detected_encoding.lower() in ['gb2312', 'gbk', 'gb18030']:
                content = raw_data.decode('gbk', errors='replace')
                encoding = 'gbk'
            elif detected_encoding.lower() == 'ascii':
                content = raw_data.decode('utf-8', errors='replace')
                encoding = 'utf-8'
            else:
                try:
                    content = raw_data.decode(detected_encoding, errors='replace')
                    encoding = detected_encoding
                except:
                    content = raw_data.decode('utf-8', errors='replace')
                    encoding = 'utf-8'

        if old_text not in content:
            return f"Error: Text not found in {path}"

        # Replace only first occurrence for safety
        new_content = content.replace(old_text, new_text, 1)
        fp.write_text(new_content, encoding=encoding)
        return f"Edited {path} (encoding: {encoding})"

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