"""The dashboard's English catalog covers its Chinese UI strings."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from simulation_ready_cad_workflow.validation_pipeline.dashboard import render_lang
from simulation_ready_cad_workflow.validation_pipeline.dashboard.i18n_catalog import (
    CJK_ANY_RE,
    EDGE_CHARS,
    STATIC_ROOT,
    canonical,
    compile_templates,
    covered,
    extract_all,
    extract_js,
    extract_python,
    has_cjk,
    load_catalog,
    placeholders,
)
from simulation_ready_cad_workflow.validation_pipeline.dashboard.project_summary import activity_view
from simulation_ready_cad_workflow.validation_pipeline.dashboard.render import render_live_shell, render_snapshot
from simulation_ready_cad_workflow.validation_pipeline.trajectory.store import HistoryStore


# Chinese literals in the scanned sources that are never shown as UI text.
# Keep each entry justified; display strings belong in static/i18n/en.js.
NOT_UI: frozenset[str] = frozenset()


class DashboardI18nCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()
        cls.templates = compile_templates(cls.catalog)

    def test_every_chinese_ui_string_has_an_english_entry(self) -> None:
        missing = {}
        for entry in extract_all():
            if entry.key in NOT_UI or entry.key in missing:
                continue
            if not covered(entry.key, self.catalog, self.templates):
                missing[entry.key] = f"{entry.source}:{entry.line}"
        self.assertEqual(
            missing,
            {},
            "add these to dashboard/static/i18n/en.js (python -m "
            "simulation_ready_cad_workflow.validation_pipeline.dashboard.i18n_catalog lists them)",
        )

    def test_entries_are_english_and_keep_their_placeholders(self) -> None:
        problems = []
        for key, value in self.catalog.items():
            if not isinstance(value, str) or not value.strip():
                problems.append((key, "empty"))
            elif CJK_ANY_RE.search(value):
                problems.append((key, "CJK text in the English value"))
            elif placeholders(key) != placeholders(value):
                problems.append((key, f"placeholders {placeholders(key)} != {placeholders(value)}"))
            if not has_cjk(key):
                problems.append((key, "key has no Chinese text"))
            elif canonical(key) != key:
                problems.append((key, f"key is not canonical: {canonical(key)!r}"))
        self.assertEqual(problems, [])

    def test_not_ui_allowlist_only_names_extracted_strings(self) -> None:
        extracted = {entry.key for entry in extract_all()}
        self.assertEqual(sorted(NOT_UI - extracted), [])

    def test_browser_and_python_use_the_same_edge_punctuation(self) -> None:
        source = (STATIC_ROOT / "i18n.js").read_text(encoding="utf-8")
        literal = re.search(r'var EDGE = ("(?:[^"\\]|\\.)*");', source)
        self.assertIsNotNone(literal)
        self.assertEqual(json.loads(literal.group(1)), EDGE_CHARS)


class DashboardI18nExtractorTests(unittest.TestCase):
    def keys(self, entries):
        return [entry.key for entry in entries]

    def test_js_templates_become_placeholders_and_split_at_tags(self) -> None:
        source = (
            'const a = `<p title="查看 ${name} 详情">共 ${n} 项，${m} 项失败</p><b>完成。</b>`;\n'
            "const b = '停止' + '运行';\n"
            'if (title === "修改文件") {}\n'
            "// 注释里的中文不算\n"
            'const c = /[中文]/.test(x) ? "\\u5df2\\u4fdd\\u5b58" : `${ok ? `${k} 个版本` : ""}`;\n'
        )
        self.assertEqual(
            sorted(self.keys(extract_js(source))),
            sorted(["查看 {0} 详情", "共 {0} 项，{1} 项失败", "完成", "停止运行", "已保存", "{0} 个版本"]),
        )

    def test_python_f_strings_concatenation_and_skips(self) -> None:
        source = (
            '"""模块说明不算。"""\n'
            "def f(x, title):\n"
            '    """函数说明不算。"""\n'
            '    if title == "修改文件":\n'
            "        return None\n"
            '    a = f"已完成 {x} 项"\n'
            "    b = '运行命令：' + x.splitlines()[0]\n"
            "    c = '%s 失败' % x\n"
            "    d = '共 {} 条'.format(x)\n"
            "    return {'label': f\"{x or '该机构'} 的记录\"}\n"
        )
        self.assertEqual(
            sorted(self.keys(extract_python(source))),
            sorted(["已完成 {0} 项", "运行命令：{0}", "{0} 失败", "共 {0} 条", "该机构", "{0} 的记录"]),
        )

    def test_canonical_keys_trim_separators_and_wrapping_brackets(self) -> None:
        self.assertEqual(canonical("  继续处理：\n "), "继续处理")
        self.assertEqual(canonical("（实验性）"), "实验性")
        self.assertEqual(canonical("准确长方体（来自 STEP）"), "准确长方体（来自 STEP）")
        self.assertEqual(canonical("▶ 播放 {0} 个画面"), "播放 {0} 个画面")

    def test_template_coverage(self) -> None:
        catalog = {"已完成 {0} 项": "{0} done"}
        templates = compile_templates(catalog)
        self.assertTrue(covered("已完成 {0} 项", catalog, templates))
        self.assertTrue(covered("已完成 12 项", catalog, templates))
        self.assertFalse(covered("未完成 12 项", catalog, templates))
        # Like i18n.js: little fixed Chinese text needs fully translatable captures.
        catalog = {"{0}任务": "{0} task", "整机": "Whole-machine"}
        templates = compile_templates(catalog)
        self.assertTrue(covered("整机任务", catalog, templates))
        self.assertTrue(covered("3任务", catalog, templates))
        self.assertFalse(covered("新任务", catalog, templates))


class DashboardI18nShellTests(unittest.TestCase):
    def test_pages_load_the_language_switch_before_the_app(self) -> None:
        for page, app_script in (("index.html", "/static/app.js"), ("launch.html", "/static/launch.js")):
            html = render_live_shell("/base", page=page)
            switch = html.index('<script src="/base/static/i18n.js"></script>')
            self.assertLess(html.index("window.QUEST_BOOT="), switch)
            self.assertLess(switch, html.index(f'src="/base{app_script}"'))

    def test_snapshot_inlines_the_switch_and_catalog(self) -> None:
        html = render_snapshot({"run": {}})
        self.assertNotIn('src="/static/i18n.js"', html)
        catalog = html.index("/* BEGIN CATALOG */")
        switch = html.index("function createTranslator")
        self.assertLess(catalog, switch)
        self.assertLess(switch, html.index('id="app"'))
        self.assertNotIn("</script", html[catalog : html.index("/* END CATALOG */")])

    def test_activity_keeps_agent_text_apart_from_its_label(self) -> None:
        # The header renders the label as UI text and the detail as translate="no".
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / ".runtime_runs" / "projects" / "demo"
            project.mkdir(parents=True)
            store = HistoryStore(project)
            events = [
                {"kind": "message", "title": "公开消息", "text": "模型\n构建失败"},
                {"kind": "status", "title": "本轮执行结束", "text": ""},
            ]
            for seq, event in enumerate(events, 1):
                with store.connect(write=True) as db:
                    db.execute(
                        "INSERT INTO events (seq,raw_path,offset,length,turn,at_ms,observed_ms,payload) VALUES (?,?,?,?,?,?,?,?)",
                        (seq, "raw.jsonl", seq, 1, "t001", 1000 * seq, None, json.dumps([event], ensure_ascii=False)),
                    )
                view = activity_view(project)
                if event["kind"] == "message":
                    self.assertEqual((view["summary"], view["label"], view["detail"]), ("消息：模型 构建失败", "消息", "模型 构建失败"))
                else:
                    self.assertEqual(view["summary"], "本轮执行结束")
                    self.assertNotIn("detail", view)

    def test_render_language_defaults_to_chinese(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(render_lang.ENV_VAR, None)
            self.assertEqual(render_lang.label("中文", "English"), "中文")
            self.assertIn("Noto Sans CJK SC", render_lang.sans_serif_fonts())
        with mock.patch.dict(os.environ, {render_lang.ENV_VAR: "en"}):
            self.assertEqual(render_lang.label("中文", "English"), "English")
            self.assertEqual(render_lang.sans_serif_fonts(), ["DejaVu Sans"])


if __name__ == "__main__":
    unittest.main()
