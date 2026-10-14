"""Find the Chinese UI strings the dashboard can show and check the English catalog.

The browser translates rendered text with ``static/i18n.js`` and the catalog in
``static/i18n/en.js``. This module lists the Chinese display strings in the
frontend sources and in the Python modules whose strings reach the page, so a
test can require an English entry for each of them.

JS and HTML sources: string and template literals are read with a small
scanner; ``${...}`` becomes ``{0}``, ``{1}`` … and the text is split at HTML
tags (the attribute values the browser translates are kept). Python sources:
``ast`` string constants, f-strings and ``+``/``%``/``.format`` templates;
docstrings and comparison operands (data matching, not display) are skipped.

Keys are normalised with :func:`canonical`, mirroring ``i18n.js``: whitespace
is collapsed and separator punctuation at either end is dropped (the browser
keeps it and only looks up the core text).

Run ``python -m simulation_ready_cad_workflow.validation_pipeline.dashboard.i18n_catalog``
to print the strings that still lack an English entry.
"""

from __future__ import annotations

import ast
import html
import json
import re
import sys
from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path


DASHBOARD_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = DASHBOARD_ROOT / "static"
CATALOG_PATH = STATIC_ROOT / "i18n" / "en.js"

# Frontend sources rendered into the page. The viewer files are the sources of
# the viewer3d.js bundle; their labels reach the DOM the same way.
JS_SOURCES = (
    "static/app.js",
    "static/launch.js",
    "static/runs.js",
    "static/workbench.js",
    "static/parts3d.js",
    "static/viewer3d.source.js",
    "static/viewer3d_mechanics.js",
)
HTML_SOURCES = ("static/index.html", "static/launch.html", "static/runs.html")
# Python modules whose strings are sent to the page (labels, summaries, errors),
# relative to the dashboard package.
PY_SOURCES = (
    "model.py",
    "project_summary.py",
    "server.py",
    "launcher.py",
    "agent_trajectory.py",
    "history_markers.py",
    "history_preview.py",
    "image_service.py",
    "image_worker.py",
    "physics_evidence.py",
    "preview_service.py",
    "preview_worker.py",
    "project_usage.py",
    "publication.py",
    "run_config.py",
    # Agent trajectory titles and notes shown in the trajectory drawer.
    "../trajectory/events.py",
    "../trajectory/store.py",
    "../trajectory/capture.py",
)

# Han ideographs: the browser only translates text that contains one.
CJK_RE = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
# Ideographs plus CJK/full-width punctuation: none may remain in English text.
CJK_ANY_RE = re.compile("[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")
PLACEHOLDER_RE = re.compile(r"\{(\d+)\}")
# Must match EDGE in static/i18n.js.
EDGE_CHARS = " \t\r\n\u3000\u00b7\u2022|\uff5c\uff0c,\uff1b;\uff1a:\u3001\u3002.\uff01!\uff1f?\u2026\u2192\u2190\u2197\u2194\u25b6\u275a\u00d7\u2713\u2717\u2304\u25cf\u25cb\u25c6\u25c7\u25a0\u25a1\u2605\u2606\u26a0"
TRANSLATED_ATTRIBUTES = ("title", "aria-label", "placeholder", "alt", "value")

_SLOT = "\x00"
_TAG_RE = re.compile(r"<[^<>]*>")
_ATTR_RE = re.compile(r"""\b(title|aria-label|placeholder|alt|value)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_WS_RE = re.compile(r"\s+")
_WRAPPERS = {("\uff08", "\uff09"), ("(", ")")}
_BRACKETS_RE = re.compile("[()\uff08\uff09]")


@dataclass(frozen=True)
class Entry:
    key: str
    source: str
    line: int


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def canonical(text: str) -> str:
    """Return the lookup key for ``text`` (same rule as ``splitEdges`` in i18n.js).

    Whitespace is collapsed and separator punctuation is dropped from both ends;
    brackets around the whole text, as in "（实验性）", are dropped too.
    """
    core = _WS_RE.sub(" ", text).strip(EDGE_CHARS)
    if len(core) > 2 and (core[0], core[-1]) in _WRAPPERS and not _BRACKETS_RE.search(core[1:-1]):
        core = core[1:-1].strip(EDGE_CHARS)
    return core


def placeholders(text: str) -> list[str]:
    return sorted(set(PLACEHOLDER_RE.findall(text)), key=int)


# --------------------------------------------------------------------------- JS


@dataclass
class _Literal:
    text: str  # decoded text; _SLOT marks each ${...}
    start: int
    end: int
    line: int
    skip: bool = False


_JS_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}
_REGEX_PRECEDERS = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_KEYWORDS = {"return", "typeof", "case", "do", "else", "in", "of", "new", "delete", "void", "throw", "instanceof", "yield", "await"}
_COMPARISON_TAILS = ("===", "!==", "==", "!=", "includes(", "startsWith(", "endsWith(", "indexOf(", "split(", "match(", "RegExp(", "has(", "test(")


class _JsScanner:
    def __init__(self, source: str) -> None:
        self.s = source
        self.n = len(source)
        self.literals: list[_Literal] = []

    def line(self, pos: int) -> int:
        return self.s.count("\n", 0, pos) + 1

    def scan(self) -> list[_Literal]:
        self._code(0, nested=False)
        return self.literals

    def _prev_significant(self, pos: int) -> tuple[str, str]:
        """Return (last char, last word) of code before ``pos``."""
        j = pos - 1
        while j >= 0 and self.s[j].isspace():
            j -= 1
        if j < 0:
            return "", ""
        k = j
        while k >= 0 and (self.s[k].isalnum() or self.s[k] in "_$"):
            k -= 1
        return self.s[j], self.s[k + 1 : j + 1]

    def _is_comparison(self, start: int, end: int) -> bool:
        before = self.s[max(0, start - 16) : start].rstrip()
        if before.endswith(_COMPARISON_TAILS):
            return True
        after = self.s[end : end + 4].lstrip()
        return after.startswith(("===", "!==", "== ", "!= "))

    def _code(self, i: int, nested: bool) -> int:
        """Scan code from ``i``; with ``nested`` stop after the matching ``}``."""
        depth = 0
        s, n = self.s, self.n
        while i < n:
            c = s[i]
            if c in "\"'":
                i = self._string(i)
            elif c == "`":
                i = self._template(i)
            elif c == "/" and i + 1 < n and s[i + 1] == "/":
                nl = s.find("\n", i)
                i = n if nl < 0 else nl
            elif c == "/" and i + 1 < n and s[i + 1] == "*":
                close = s.find("*/", i + 2)
                i = n if close < 0 else close + 2
            elif c == "/":
                last, word = self._prev_significant(i)
                if not last or last in _REGEX_PRECEDERS or word in _REGEX_KEYWORDS:
                    i = self._regex(i)
                else:
                    i += 1
            elif c == "{":
                depth += 1
                i += 1
            elif c == "}":
                if nested and depth == 0:
                    return i + 1
                depth -= 1
                i += 1
            else:
                i += 1
        return i

    def _escape(self, i: int) -> tuple[str, int]:
        """Decode the escape at ``s[i] == '\\'``; return (text, next index)."""
        s = self.s
        c = s[i + 1] if i + 1 < self.n else ""
        if c == "u":
            if s[i + 2 : i + 3] == "{":
                close = s.index("}", i + 3)
                return chr(int(s[i + 3 : close], 16)), close + 1
            return chr(int(s[i + 2 : i + 6], 16)), i + 6
        if c == "x":
            return chr(int(s[i + 2 : i + 4], 16)), i + 4
        if c == "\n":
            return "", i + 2
        return _JS_ESCAPES.get(c, c), i + 2

    def _string(self, i: int) -> int:
        quote, start = self.s[i], i
        out: list[str] = []
        i += 1
        while i < self.n and self.s[i] != quote:
            if self.s[i] == "\\":
                text, i = self._escape(i)
                out.append(text)
            else:
                out.append(self.s[i])
                i += 1
        end = i + 1
        self.literals.append(
            _Literal("".join(out), start, end, self.line(start), self._is_comparison(start, end))
        )
        return end

    def _template(self, i: int) -> int:
        start = i
        out: list[str] = []
        i += 1
        while i < self.n and self.s[i] != "`":
            if self.s[i] == "\\":
                text, i = self._escape(i)
                out.append(text)
            elif self.s.startswith("${", i):
                out.append(_SLOT)
                i = self._code(i + 2, nested=True)
            else:
                out.append(self.s[i])
                i += 1
        end = i + 1
        self.literals.append(
            _Literal("".join(out), start, end, self.line(start), self._is_comparison(start, end))
        )
        return end

    def _regex(self, i: int) -> int:
        i += 1
        in_class = False
        while i < self.n:
            c = self.s[i]
            if c == "\\":
                i += 2
                continue
            if c == "\n":
                return i
            if c == "[":
                in_class = True
            elif c == "]":
                in_class = False
            elif c == "/" and not in_class:
                i += 1
                while i < self.n and self.s[i].isalpha():
                    i += 1
                return i
            i += 1
        return i


def _merge_concatenations(source: str, literals: list[_Literal]) -> list[_Literal]:
    """Join ``"a" + "b"`` (literal + literal) chains into one literal."""
    ordered = sorted(literals, key=lambda item: item.start)
    merged: list[_Literal] = []
    for item in ordered:
        if merged:
            last = merged[-1]
            gap = source[last.end : item.start]
            if last.end <= item.start and gap.strip() == "+":
                merged[-1] = _Literal(last.text + item.text, last.start, item.end, last.line, last.skip and item.skip)
                continue
        merged.append(item)
    return merged


def _number_slots(text: str) -> str:
    counter = iter(range(10_000))
    return re.sub(_SLOT, lambda _match: "{%d}" % next(counter), text)


def _split_markup(text: str) -> list[str]:
    """Split text at HTML tags; keep the attribute values the browser translates.

    Attribute assignments are also read outside complete tags because markup is
    often assembled from fragments such as ``'disabled title="…"'``.
    """
    pieces = [m.group(2) if m.group(2) is not None else m.group(3) for m in _ATTR_RE.finditer(text)]
    pieces.extend(_TAG_RE.split(_ATTR_RE.sub(" ", text)))
    return pieces


def _keys_from_text(text: str, markup: bool = True) -> list[str]:
    keys = []
    for piece in _split_markup(text) if markup else [text]:
        if "&" in piece:
            piece = html.unescape(piece)
        key = canonical(_number_slots(piece))
        if has_cjk(key):
            keys.append(key)
    return keys


def extract_js(source: str, name: str = "<js>") -> list[Entry]:
    literals = _merge_concatenations(source, _JsScanner(source).scan())
    entries = []
    for literal in literals:
        if literal.skip or not has_cjk(literal.text):
            continue
        for key in _keys_from_text(literal.text):
            entries.append(Entry(key, name, literal.line))
    return entries


def extract_html(source: str, name: str = "<html>") -> list[Entry]:
    entries = []
    body = re.sub(r"<script\b.*?</script>|<!--.*?-->", "", source, flags=re.S)
    for key in _keys_from_text(body):
        line = source.count("\n", 0, max(0, source.find(key[:4]))) + 1
        entries.append(Entry(key, name, line))
    return entries


# ----------------------------------------------------------------------- Python


_PRINTF_RE = re.compile(r"%(?:\([^)]*\))?[-+ #0]*(?:\d+|\*)?(?:\.(?:\d+|\*))?[sdifgeExXrac]")
_FORMAT_FIELD_RE = re.compile(r"\{[^{}]*\}")


class _PyExtractor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.entries: list[Entry] = []

    def emit(self, text: str, line: int) -> None:
        if not has_cjk(text):
            return
        for key in _keys_from_text(text):
            self.entries.append(Entry(key, self.name, line))

    def visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]  # docstring
            for child in ast.iter_child_nodes(node):
                if child in node.body and child not in body:
                    continue
                self.visit(child)
            return
        if isinstance(node, ast.Compare):
            # Operands of comparisons match data (e.g. agent titles), not display.
            for child in [node.left, *node.comparators]:
                if not (isinstance(child, ast.Constant) and isinstance(child.value, str)):
                    self.visit(child)
            return
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                self.emit(node.value, node.lineno)
            return
        if isinstance(node, ast.JoinedStr):
            self.emit(self._joined(node), node.lineno)
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            operands = self._flatten_add(node)
            if any(self._literal_text(op) is not None for op in operands):
                parts = []
                for op in operands:
                    text = self._literal_text(op)
                    if text is None:
                        parts.append(_SLOT)
                        self.visit(op)
                    else:
                        parts.append(text)
                        if isinstance(op, ast.JoinedStr):
                            self._visit_joined_values(op)
                self.emit("".join(parts), node.lineno)
                return
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            self.emit(_PRINTF_RE.sub(_SLOT, node.left.value.replace("%%", "%")), node.lineno)
            self.visit(node.right)
            return
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and isinstance(node.func.value, ast.Constant)
            and isinstance(node.func.value.value, str)
        ):
            text = node.func.value.value.replace("{{", "\x01").replace("}}", "\x02")
            text = _FORMAT_FIELD_RE.sub(_SLOT, text).replace("\x01", "{").replace("\x02", "}")
            self.emit(text, node.lineno)
            for child in [*node.args, *node.keywords]:
                self.visit(child)
            return
        for child in ast.iter_child_nodes(node):
            self.visit(child)

    def _flatten_add(self, node: ast.AST) -> list[ast.AST]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self._flatten_add(node.left) + self._flatten_add(node.right)
        return [node]

    def _literal_text(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return self._joined(node, visit=False)
        return None

    def _joined(self, node: ast.JoinedStr, visit: bool = True) -> str:
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            else:
                parts.append(_SLOT)
                if visit and isinstance(value, ast.FormattedValue):
                    self.visit(value.value)
        return "".join(parts)

    def _visit_joined_values(self, node: ast.JoinedStr) -> None:
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                self.visit(value.value)


def extract_python(source: str, name: str = "<py>") -> list[Entry]:
    extractor = _PyExtractor(name)
    extractor.visit(ast.parse(source))
    return extractor.entries


# ---------------------------------------------------------------------- catalog


def extract_all(root: Path = DASHBOARD_ROOT) -> list[Entry]:
    entries: list[Entry] = []
    for relative in JS_SOURCES:
        path = root / relative
        if path.is_file():
            entries += extract_js(path.read_text(encoding="utf-8"), relative)
    for relative in HTML_SOURCES:
        path = root / relative
        if path.is_file():
            entries += extract_html(path.read_text(encoding="utf-8"), relative)
    for relative in PY_SOURCES:
        path = root / relative
        if path.is_file():
            entries += extract_python(path.read_text(encoding="utf-8"), relative)
    return entries


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, str]:
    """Read the JSON object between the BEGIN/END markers of ``en.js``."""
    text = path.read_text(encoding="utf-8")
    start = text.index("/* BEGIN CATALOG */") + len("/* BEGIN CATALOG */")
    end = text.index("/* END CATALOG */", start)
    return json.loads(text[start:end])


class Template(NamedTuple):
    pattern: re.Pattern[str]
    key: str
    literal: str  # longest fixed part, a cheap pre-check
    strict: bool  # under 3 fixed Han characters: captures must translate too


def compile_templates(catalog: dict[str, str]) -> list[Template]:
    """Compile the keys with placeholders, most fixed text first (as i18n.js)."""
    templates = []
    for key in catalog:
        if not PLACEHOLDER_RE.search(key):
            continue
        parts = PLACEHOLDER_RE.split(key)
        literals = parts[::2]
        pattern = "".join(re.escape(part) if index % 2 == 0 else "(.*?)" for index, part in enumerate(parts))
        strict = len(CJK_RE.findall("".join(literals))) < 3
        templates.append(Template(re.compile(pattern + r"\Z", re.S), key, max(literals, key=len), strict))
    templates.sort(key=lambda item: -len(PLACEHOLDER_RE.sub("", item.key)))
    return templates


def covered(key: str, catalog: dict[str, str], templates: list[Template] | None = None) -> bool:
    """True when the browser finds ``key`` exactly or through a template."""
    if key in catalog:
        return True
    templates = compile_templates(catalog) if templates is None else templates
    for template in templates:
        if template.literal not in key:
            continue
        match = template.pattern.match(key)
        if match and (
            not template.strict
            or all(not has_cjk(value) or covered(canonical(value), catalog, templates) for value in match.groups())
        ):
            return True
    return False


def missing_keys(entries: list[Entry] | None = None, catalog: dict[str, str] | None = None) -> dict[str, Entry]:
    catalog = load_catalog() if catalog is None else catalog
    templates = compile_templates(catalog)
    missing: dict[str, Entry] = {}
    for entry in extract_all() if entries is None else entries:
        if entry.key not in missing and not covered(entry.key, catalog, templates):
            missing[entry.key] = entry
    return missing


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    entries = extract_all()
    if "--all" in argv:
        unique: dict[str, Entry] = {}
        for entry in entries:
            unique.setdefault(entry.key, entry)
        rows = unique
    else:
        rows = missing_keys(entries) if CATALOG_PATH.is_file() else {e.key: e for e in entries}
    json.dump({key: f"{entry.source}:{entry.line}" for key, entry in rows.items()}, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
