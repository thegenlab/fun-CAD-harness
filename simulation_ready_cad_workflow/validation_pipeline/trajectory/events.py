"""Project public-event projection. Original backend logs remain untouched."""
from datetime import datetime, timezone
import json
import re

PAGE_BYTES = 512 * 1024
PAGE_ITEMS = 100
TEXT_LIMIT = 12000
_SECRET = re.compile(r"(?i)(?<![\w-])((?:[\w-]*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)|authorization)\s*[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)")
_BEARER = re.compile(r"(?i)\bBearer\s+[^\s\"']+")


def display_text(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    value = _SECRET.sub(r"\1[redacted]", _BEARER.sub("Bearer [redacted]", value))
    return value[:TEXT_LIMIT] + ("\n…（显示已截断，原始记录保留）" if len(value) > TEXT_LIMIT else "")


def _content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(row.get("text", "") for row in value if isinstance(row, dict) and row.get("type") == "text")
    return ""


def _blocks(content, kind):
    return [block for block in content if isinstance(block, dict) and block.get("type") == kind] if isinstance(content, list) else []


def _item_event(event):
    """Read App notifications and CLI events without changing the raw object."""
    kind = event.get("type")
    item = event.get("item") or {}
    if kind is None and event.get("method") in {"item/started", "item/updated", "item/completed", "turn/completed"}:
        kind = event["method"].replace("/", ".")
        item = (event.get("params") or {}).get("item") or {}
    if isinstance(item, dict):
        item = dict(item)
        item["type"] = {"agentMessage": "agent_message", "commandExecution": "command_execution",
                        "mcpToolCall": "mcp_tool_call", "fileChange": "file_change"}.get(item.get("type"), item.get("type"))
        for source, target in (("aggregatedOutput", "aggregated_output"), ("exitCode", "exit_code")):
            if source in item:
                item.setdefault(target, item[source])
        if item.get("status") == "inProgress":
            item["status"] = "in_progress"
    return kind, item


def event_rows(event, offset):
    if not isinstance(event, dict):
        return []
    kind, item = _item_event(event)
    stamp = event.get("timestamp") or ""
    def row(category, title, text="", **extra):
        if event.get('source')=='harness_program':extra['source']='harness_program'
        return {"id":str(offset), "kind":category, "title":title, "text":display_text(text), "at":stamp, **extra}
    if kind in {"item.started", "item.updated", "item.completed"} and isinstance(item, dict):
        category = item.get("type")
        if category == "agent_message":
            return [row("message", "Agent", item.get("text", ""), key=item.get("id"))]
        if category == "command_execution":
            return [row("tool", "运行命令", item.get("command", ""), key=item.get("id"),
                        output=display_text(item.get("aggregated_output", "")), exit_code=item.get("exit_code"),
                        status=item.get("status") or ("completed" if kind == "item.completed" else "running"))]
        if category == "mcp_tool_call":
            return [row("tool", str(item.get("tool") or "工具调用"), item.get("arguments", {}), key=item.get("id"),
                        output=display_text(item.get("result") or item.get("error") or ""),
                        status=item.get("status") or ("completed" if kind == "item.completed" else "running"))]
        if category == "file_change":
            return [row("tool", "修改文件", item.get("changes", []), key=item.get("id"), status=item.get("status", "completed"))]
        # Reasoning/internal-analysis events are deliberately not projected.
        return []
    if kind == "assistant":
        message = event.get("message") or {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        text = _content(content)
        rows = [row("message", "Agent", text)] if text else []
        # Claude Code streams tool calls as tool_use blocks and their results as user tool_result blocks.
        return rows + [row("tool", str(block.get("name") or "工具调用"), block.get("input", {}), key=block.get("id"), status="running")
                       for block in _blocks(content, "tool_use")]
    if kind == "user":
        message = event.get("message") or {}
        rows = []
        for block in _blocks(message.get("content") if isinstance(message, dict) else None, "tool_result"):
            result = row("tool", "", key=block.get("tool_use_id"), output=display_text(_content(block.get("content"))),
                         status="failed" if block.get("is_error") else "completed")
            # Update the tool_use row by key without replacing its title and input.
            del result["title"], result["text"]
            rows.append(result)
        return rows
    if kind == "tool_call":
        tool = event.get("tool_call") or {}
        if not isinstance(tool, dict):
            return []
        name, payload = next(iter(tool.items()), ("工具调用", {}))
        if name == "function" and isinstance(payload, dict):
            name, payload = payload.get("name", name), {"args":payload.get("arguments", "")}
        payload = payload if isinstance(payload, dict) else {}
        return [row("tool", str(name), payload.get("args", {}), key=event.get("call_id"),
                    output=display_text(payload.get("result") or ""), status=event.get("subtype", "running"))]
    if kind in {"error", "turn.failed"}:
        return [row("error", "执行错误", event.get("message") or event.get("error") or "该轮执行失败")]
    if kind == "turn.completed":
        return [row("status", "本轮执行结束", event.get("usage") or "")]
    return []


def event_ms(event):
    value = event.get("timestamp_ms", event.get("emittedAtMs"))
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    stamp = event.get("timestamp")
    if isinstance(stamp, str) and stamp:
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
    return None


def completed_tool_output(event):
    """Original result for indexing associations, before display truncation."""
    value = None
    kind, item = _item_event(event)
    if kind == 'item.completed' and isinstance(item, dict):
        if item.get('type') == 'command_execution':
            value = item.get('aggregated_output')
        elif item.get('type') == 'mcp_tool_call':
            value = item.get('result') or item.get('error')
    elif event.get('type') == 'tool_call' and event.get('subtype') == 'completed':
        tool = event.get('tool_call') or {}
        values = [payload['result'] for payload in tool.values() if isinstance(payload,dict) and 'result' in payload]
        value = values
    elif event.get('type') == 'user' and isinstance(event.get('message'), dict):
        value = '\n'.join(_content(block.get('content')) for block in _blocks(event['message'].get('content'), 'tool_result'))
    if value is None:return ''
    return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)


def merge_rows(records):
    items, keys = [], {}
    for _, row in records:
        key = row.pop("key", None)
        if key and key in keys:
            target = items[keys[key]]
            row["id"] = target["id"]
            target.update(row)
        else:
            if key:
                keys[key] = len(items)
            items.append(row)
    return items
