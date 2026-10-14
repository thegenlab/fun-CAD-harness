"""Author-only visual feedback restrictions; no geometry or verdict changes.

Tool guards reject the maintained visual interfaces. This is not a semantic
classifier for arbitrary programs or a replacement for the folder sandbox.
"""
from __future__ import annotations

import ast
import json
import re
import shlex
from pathlib import Path

FULL = "full"
WORKER_NO_VISUALS = "worker_no_visuals_v1"
POLICIES = {FULL, WORKER_NO_VISUALS}
VISUAL_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".tif", ".tiff",
    ".ppm", ".pgm", ".pbm", ".exr", ".ico", ".mp4", ".mov", ".avi", ".webm",
    ".glb", ".gltf", ".html", ".htm", ".pdf",
})
# STL/OBJ/PLY/STEP and numeric traces are physics/CAD data and stay available.
VISUAL_OPERATIONS = frozenset({
    "render_views", "_render_views", "export_replay", "view_image", "emitImage",
    "savefig", "show_object", "show_all", "export_svg", "export_png",
    "to_svg", "render_image", "save_image", "screenshot", "capture_screen_image",
    "export_glb", "export_gltf", "ExportSVG", "V3d_Viewer", "V3d_View", "ToPixMap",
})
VISUAL_MODULES = frozenset({
    "matplotlib", "mpl_toolkits", "PIL", "cv2", "pyvista", "pyrender", "bpy",
    "ocp_vscode", "ocp_tessellate", "pyglet", "pygame", "selenium", "playwright",
})
VISUAL_PROGRAMS = frozenset({
    "blender", "ffmpeg", "ffplay", "display", "magick", "convert", "xdg-open",
    "freecad", "freecadcmd", "openscad", "meshlab", "meshlabserver",
    "chromium", "google-chrome", "firefox",
})
MARKER = Path(__file__).with_name("author-visual-policy.json")


class VisualAccessDisabled(RuntimeError):
    pass


def validate(policy: str | None) -> str:
    value = FULL if policy is None else policy
    if value not in POLICIES:
        raise ValueError(f"Unknown author visual policy: {value!r}")
    return value


def runtime_policy() -> str:
    """This marker is installed only in the author's read-only public SDK."""
    if not MARKER.exists():
        return FULL
    value = json.loads(MARKER.read_text(encoding="utf-8"))
    if value != {"schema_version": 1, "author_visual_policy": WORKER_NO_VISUALS}:
        raise ValueError("Invalid immutable author visual policy")
    return WORKER_NO_VISUALS


def effective_policy(policy=None) -> str:
    explicit = validate(policy)
    return WORKER_NO_VISUALS if runtime_policy() == WORKER_NO_VISUALS else explicit


def is_visual_artifact(path) -> bool:
    if not isinstance(path, (str, Path)):
        return False
    return Path(str(path).split("?", 1)[0].split("#", 1)[0]).suffix.lower() in VISUAL_SUFFIXES


def record_denial(log, *, command, reason):
    """Optional cooperative draft log; never used as trusted enforcement proof."""
    if log:
        path = Path(log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"command": command, "reason": reason}) + "\n")


def deny(command, log=None):
    reason = "worker visual feedback is disabled for this author turn"
    record_denial(log, command=command, reason=reason)
    raise VisualAccessDisabled(f"{reason}: {command}")


def require_nonvisual(command, policy=None, log=None):
    if effective_policy(policy) == WORKER_NO_VISUALS:
        deny(command, log)


def redact_history(value):
    """Remove visual payloads, keeping text feedback and numeric physics data."""
    if isinstance(value, dict):
        hidden = {"image", "images", "image_paths", "evidence_images", "preview_path",
                  "replay_path", "motion_model_path", "thumbnails", "image_url", "image_data"}
        result = {}
        for key, item in value.items():
            if key in hidden:
                continue
            if isinstance(item, str) and is_visual_artifact(item) and key not in {"summary", "feedback", "issues", "message", "text", "description"}:
                continue
            result[key] = redact_history(item)
        return result
    if isinstance(value, list):
        return [redact_history(item) for item in value if not isinstance(item, str) or not is_visual_artifact(item)]
    if isinstance(value, str):
        suffix = "|".join(re.escape(x[1:]) for x in sorted(VISUAL_SUFFIXES))
        value = re.sub(r"data:(?:image|video)/[^\s\"'<>]+", "[visual payload withheld]", value, flags=re.I)
        value = re.sub(r"<img\b[^>]*>", "[visual payload withheld]", value, flags=re.I)
        value = re.sub(r"<(svg|canvas|video)\b[^>]*>.*?</\1>",
                       "[visual payload withheld]", value, flags=re.I | re.S)
        return re.sub(r"[^\s\"'<>`]+\.(?:" + suffix + r")(?=[?#\s\"'<>`),;]|$)(?:[?#][^\s\"'<>`]*)?",
                      "[visual artifact withheld]", value, flags=re.I)
    return value


def _source_reason(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    docstrings = {id(node.value) for node in ast.walk(tree)
                  if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)}
    parents = {id(child): parent for parent in ast.walk(tree)
               for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            parent = parents.get(id(node))
            text_output = (isinstance(parent, ast.Call) and node in parent.args and
                           ((isinstance(parent.func, ast.Name) and parent.func.id == "print") or
                            (isinstance(parent.func, ast.Attribute) and parent.func.attr in {"write_text", "write"})))
            prose = text_output and any(char.isspace() for char in node.value)
            if (is_visual_artifact(node.value) and not prose) or node.value.startswith(("data:image/", "data:video/")):
                return "visual_artifact_access"
        if isinstance(node, ast.Import):
            modules = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        if any(name.split(".")[0] in VISUAL_MODULES or name.startswith(("OCP.V3d", "OCP.AIS", "OCP.OpenGl")) for name in modules):
            return "visual_library"
        if isinstance(node, ast.Call):
            call = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if call in VISUAL_OPERATIONS:
                return "visual_operation"
            if call in {"run", "Popen", "call", "check_call", "check_output", "system", "popen"} and node.args:
                try:
                    command = ast.literal_eval(node.args[0])
                except (ValueError, TypeError):
                    continue
                if isinstance(command, str) or (isinstance(command, list) and all(isinstance(x, str) for x in command)):
                    reason = check_tool("Bash", {"command": command})
                    if reason:
                        return reason
    return None


def _shell_source(command):
    """Separate heredoc data from the shell program that consumes it."""
    try:
        outer = _program_tokens(shlex.split(command))
    except ValueError:
        outer = []
    if outer and Path(outer[0]).name in {"bash", "sh", "zsh", "fish"} and any(
            flag == "--command" or re.fullmatch(r"-[a-zA-Z]*c[a-zA-Z]*", flag) for flag in outer[1:2]):
        return command, []
    lines = command.splitlines(keepends=True)
    shell, bodies = [], []
    index = 0
    while index < len(lines):
        line = lines[index]
        shell.append(line)
        index += 1
        matches = list(re.finditer(r"<<-?[ \t]*(?:'([^']+)'|\"([^\"]+)\"|([A-Za-z_][A-Za-z_0-9]*))", line))
        for match in matches:
            delimiter = next(value for value in match.groups() if value is not None)
            body = []
            while index < len(lines) and lines[index].strip() != delimiter:
                body.append(lines[index])
                index += 1
            bodies.append((line, "".join(body)))
            if index < len(lines):
                index += 1
    return "".join(shell), bodies


def _command_segments(command):
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = "#"
    segments, current = [], []
    for token in lexer:
        if token and all(char in ";&|()\n" for char in token):
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _program_tokens(tokens):
    """Locate executable position without interpreting ordinary path arguments."""
    tokens = list(tokens)
    while tokens:
        if tokens[0] in {"if", "then", "elif", "else", "while", "until", "do", "!", "exec"} or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=.*", tokens[0], re.S):
            tokens.pop(0)
            continue
        program = Path(tokens[0]).name
        if program == "command":
            index = 1
            while index < len(tokens):
                flag = tokens[index]
                if flag == "--":
                    index += 1
                    break
                if not re.fullmatch(r"-[pvV]+", flag):
                    break
                if "v" in flag or "V" in flag:
                    return tokens
                index += 1
            tokens = tokens[index:]
            continue
        if program == "env":
            tokens.pop(0)
            while tokens and (tokens[0].startswith("-") or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=.*", tokens[0], re.S)):
                option = tokens.pop(0)
                if option in {"-u", "--unset", "-C", "--chdir"} and tokens:
                    tokens.pop(0)
            continue
        if program == "timeout":
            tokens.pop(0)
            while tokens and tokens[0].startswith("-"):
                option = tokens.pop(0)
                if option == "--":
                    break
                if option in {"-k", "--kill-after", "-s", "--signal"} and tokens:
                    tokens.pop(0)
            if tokens:
                tokens.pop(0)  # timeout duration
            continue
        if program == "time":
            tokens.pop(0)
            while tokens and tokens[0].startswith("-"):
                option = tokens.pop(0)
                if option == "--":
                    break
                if option in {"-f", "--format", "-o", "--output"} and tokens:
                    tokens.pop(0)
            continue
        break
    return tokens


def _artifact_scan_command(command, *, _depth=0):
    """A negative rg filename glob names files to exclude, not files to read."""
    if _depth > 16:
        return command
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return command
    result = []
    segment = []
    def append_segment():
        if segment and Path(segment[0]).name in {"bash", "sh", "zsh", "fish"}:
            for index in range(1, len(segment) - 1):
                flag = segment[index]
                if flag == "--command" or re.fullmatch(r"-[a-zA-Z]*c[a-zA-Z]*", flag):
                    segment[index + 1] = _artifact_scan_command(segment[index + 1], _depth=_depth + 1)
                    break
                if not flag.startswith("-"):
                    break
        if not segment or Path(segment[0]).name != "rg" or "--files" not in segment:
            result.extend(segment)
            return
        skip = False
        for index, token in enumerate(segment):
            if skip:
                skip = False
                continue
            if token in {"-g", "--glob", "--iglob"} and index + 1 < len(segment) and segment[index + 1].startswith("!"):
                skip = True
                continue
            if token.startswith(("-g!", "--glob=!", "--iglob=!")):
                continue
            result.append(token)
    for token in tokens:
        if token and all(char in ";&|()<>" for char in token):
            append_segment()
            segment = []
            result.append(token)
        else:
            segment.append(token)
    append_segment()
    return " ".join(result)


def check_tool(tool_name, tool_input, cwd=None, *, _depth=0):
    """Check maintained visual interfaces at execution positions, not prose."""
    if _depth > 16:
        return "script_inspection_limit"
    name = str(tool_name).rsplit(".", 1)[-1].lower()
    if any(part in name for part in ("view_image", "imagegen", "image_gen", "screenshot", "render", "browser", "computer", "emitimage")):
        return "visual_tool"
    if not isinstance(tool_input, dict):
        return "invalid_tool_input"
    if name in {"write_stdin", "send_keys", "feed_chars"}:
        return None if tool_input.get("chars", tool_input.get("text", "")) == "" else "uninspected_interactive_input"
    for key in ("path", "file_path", "filename", "url", "output"):
        if is_visual_artifact(tool_input.get(key)):
            return "visual_artifact_access"
    command = tool_input.get("command", tool_input.get("cmd", tool_input.get("input", "")))
    argv = command if isinstance(command, list) and all(isinstance(item, str) for item in command) else None
    if argv is not None:
        command = shlex.join(argv)
    if not isinstance(command, str):
        return "invalid_command"
    if "data:image/" in command or "data:video/" in command:
        return "visual_payload"
    if name == "apply_patch":
        sections = re.split(r"^\*\*\* (?:Add|Update) File: ", command, flags=re.M)[1:]
        for section in sections:
            path, _, patch = section.partition("\n")
            if is_visual_artifact(path):
                return "visual_artifact_access"
            source = "\n".join(line[1:] for line in patch.splitlines() if line.startswith(("+", " ")))
            reason = _source_reason(source) if path.endswith(".py") else check_tool("Bash", {"command": source}, cwd, _depth=_depth + 1) if path.endswith(".sh") else None
            if reason:
                return reason
        return None
    shell, heredocs = _shell_source(command)
    for header, source in heredocs:
        try:
            parts = _command_segments(header)
        except ValueError:
            return "uninspectable_shell_command"
        consumer = _program_tokens(parts[-1]) if parts else []
        shell_body = bool(consumer and Path(consumer[0]).name in {"bash", "sh", "zsh", "fish"})
        if not shell_body and consumer and Path(consumer[0]).name == "cat":
            shell_body = any(token.endswith(".sh") for token in consumer[1:])
        if shell_body:
            directory_changes = [shlex.join(part) for part in parts[:-1]
                                 if _program_tokens(part)[:1] == ["cd"]]
            nested = dict(tool_input, command="\n".join([*directory_changes, source]))
            reason = check_tool("Bash", nested, cwd, _depth=_depth + 1)
        else:
            reason = _source_reason(source)
        if reason:
            return reason
    try:
        segments = [argv] if argv is not None else _command_segments(shell)
    except ValueError:
        return "uninspectable_shell_command"
    root = Path(cwd).resolve() if cwd else None
    base = root
    if root is not None:
        selected = tool_input.get("workdir", tool_input.get("cwd"))
        if isinstance(selected, str):
            base = (root / selected).resolve()
        if not base.is_relative_to(root):
            return "script_workspace_mismatch"
    pending = []
    suffix = "|".join(re.escape(x[1:]) for x in sorted(VISUAL_SUFFIXES))
    for segment in segments:
        tokens = _program_tokens(segment)
        if not tokens:
            continue
        program = "." if tokens[0] == "." else Path(tokens[0]).name.lower()
        for index, token in enumerate(tokens[:-1]):
            if token in {">", ">>", "<"} and is_visual_artifact(tokens[index + 1]):
                return "visual_artifact_access"
        if program in {"echo", "printf"} and "|" in shell:
            for token in tokens[1:]:
                reason = _source_reason(token)
                if reason:
                    return reason
        if program == "find":
            for index, token in enumerate(tokens):
                if token in {"-exec", "-execdir"}:
                    nested = tokens[index + 1:]
                    if nested and nested[-1] in {";", "+"}:
                        nested = nested[:-1]
                    reason = check_tool("Bash", {"command": nested}, base, _depth=_depth + 1)
                    if reason:
                        return reason
        if program in VISUAL_PROGRAMS | VISUAL_OPERATIONS | {"preview", "render"}:
            return "visual_operation"
        if program == "cd" and base is not None and len(tokens) > 1:
            base = (base / tokens[1]).resolve()
            if not base.is_relative_to(root):
                return "script_workspace_mismatch"
        wrapped = False
        if program in {"bash", "sh", "zsh", "fish"}:
            for index, flag in enumerate(tokens[1:], 1):
                if flag == "--command" or re.fullmatch(r"-[a-zA-Z]*c[a-zA-Z]*", flag):
                    if index + 1 >= len(tokens):
                        return "uninspected_interactive_input"
                    nested = {"command": tokens[index + 1]}
                    if base is not None:
                        nested["workdir"] = str(base)
                    reason = check_tool("Bash", nested, root, _depth=_depth + 1)
                    if reason:
                        return reason
                    wrapped = True
                    break
                if not flag.startswith("-"):
                    break
        if wrapped:
            continue
        interpreter = program in {"bash", "sh", "zsh", "fish", "node", "ipython", "cad-python", "chrono-python"} or re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?(?:\.exe)?", program)
        metadata_only = program in {"ls", "stat", "echo", "printf", "pwd", "command"} or (
            program == "find" and not any(token in {"-exec", "-execdir", "-ok", "-okdir"} for token in tokens))
        if not metadata_only:
            scan_tokens = list(tokens)
            if interpreter and program != "node":
                for index, option in enumerate(scan_tokens[:-1]):
                    if option in {"-c", "-e", "--eval"}:
                        scan_tokens[index + 1] = ""
                        break
            scan = _artifact_scan_command(shlex.join(scan_tokens))
            if re.search(r"\.(?:" + suffix + r")(?=[\s\"'<>`),;]|$)", scan, re.I):
                return "visual_artifact_access"
        script_tokens = []
        if interpreter:
            tail = tokens[1:]
            if not tail:
                return "uninspected_interactive_input"
            python_interpreter = program in {"ipython", "cad-python", "chrono-python"} or re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?(?:\.exe)?", program)
            index = 0
            while index < len(tail):
                option = tail[index]
                if python_interpreter and option.startswith("-") and not option.startswith("--"):
                    for position, flag in enumerate(option[1:], 1):
                        if flag in {"W", "X", "c", "m"}:
                            if "i" in option[1:position]:
                                return "uninspected_interactive_input"
                            value = option[position + 1:]
                            tail[index:index + 1] = ["-" + flag, value] if value else ["-" + flag]
                            option = tail[index]
                            break
                        if flag not in "bBdEhIiOPqsSuvVx":
                            break
                if python_interpreter and option in {"-W", "-X", "--check-hash-based-pycs"}:
                    if index + 1 >= len(tail):
                        return "uninspectable_interpreter_arguments"
                    index += 2
                    continue
                if python_interpreter and option.startswith("--check-hash-based-pycs="):
                    index += 1
                    continue
                if option == "--":
                    script_tokens = tail[index + 1:]
                    if not script_tokens or (script_tokens[0] == "-" and not heredocs and "|" not in command):
                        return "uninspected_interactive_input"
                    break
                if option in {"-i", "--interactive", "--login"} or re.fullmatch(r"-[a-zA-Z]*i[a-zA-Z]*", option):
                    return "uninspected_interactive_input"
                if option in {"-c", "-e", "--eval"}:
                    if index + 1 >= len(tail):
                        return "uninspected_interactive_input"
                    reason = _source_reason(tail[index + 1])
                    if reason:
                        return reason
                    break
                if option == "-m":
                    if index + 1 >= len(tail):
                        return "uninspected_interactive_input"
                    module = tail[index + 1]
                    if "harness" in module and any(x in {"preview", "render"} for x in tail[index + 2:]):
                        return "visual_operation"
                    if base is not None:
                        for relative in (module.replace(".", "/") + ".py", module.replace(".", "/") + "/__main__.py"):
                            candidate = (base / relative).resolve()
                            if candidate.is_relative_to(root) and candidate.is_file():
                                pending.append((candidate, base))
                    break
                if option == "-":
                    if not heredocs and "|" not in command:
                        return "uninspected_interactive_input"
                    break
                if not option.startswith("-"):
                    script_tokens = tail[index:]
                    if "harness" in Path(option).name and any(x in {"preview", "render"} for x in script_tokens[1:]):
                        return "visual_operation"
                    break
                index += 1
            else:
                if not any(flag in tail for flag in ("--version", "--help", "-h", "-V")):
                    return "uninspected_interactive_input"
        elif program.endswith((".py", ".sh")):
            script_tokens = tokens[:1]
        elif program in {"source", "."}:
            script_tokens = tokens[1:2]
        if base is not None:
            for token in script_tokens:
                if token.endswith((".py", ".sh")):
                    candidate = (base / token).resolve()
                    if candidate.is_relative_to(root) and candidate.is_file():
                        pending.append((candidate, base))
    visited = set()
    while pending:
        path, invocation_cwd = pending.pop()
        key = (path, invocation_cwd)
        if key in visited:
            continue
        visited.add(key)
        if len(visited) > 256 or path.stat().st_size > 1024 * 1024:
            return "script_inspection_limit"
        source = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".sh":
            reason = check_tool("Bash", {"command": source, "workdir": str(invocation_cwd)}, root, _depth=_depth + 1)
        else:
            reason = _source_reason(source)
        if reason:
            return reason
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            modules = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for module in modules:
                for relative in (module.replace(".", "/") + ".py", module.replace(".", "/") + "/__init__.py"):
                    dep = (path.parent / relative).resolve()
                    if dep.is_relative_to(root) and dep.is_file():
                        pending.append((dep, invocation_cwd))
    return None
