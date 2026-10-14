"""不同设计同名Python模块隔离；复现真实BM03→BM10串内力入口的问题。"""

from pathlib import Path
import tempfile, unittest, sys, types
from benchmark_templates.source_loading import build_authored_model


class SourceLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def case(self, name, value, fail=False):
        d = self.root / name
        d.mkdir()
        (d / "source_laws.py").write_text("VALUE=" + repr(value) + "\nCOUNT=0\n")
        text = """from types import SimpleNamespace
from source_laws import VALUE
def build():
    import source_laws
    source_laws.COUNT+=1
    m=SimpleNamespace(keepalive=(),before_step=None,after_step=None,commit_inputs=None,value=VALUE,count=source_laws.COUNT)
    def after(t,dt):
        import source_laws
        m.observed=source_laws.VALUE
    m.after_step=after
    return m
"""
        if fail:
            text = 'raise RuntimeError("factory failed")'
        (d / "native_model.py").write_text(text)
        return d / "native_model.py"

    def test_factories_and_late_callback_imports_are_source_specific(self):
        a = build_authored_model(self.case("a", 11), "build")
        b = build_authored_model(self.case("b", 22), "build")
        a.after_step(0, 0.1)
        b.after_step(0, 0.1)
        a.after_step(0, 0.1)
        self.assertEqual((11, 22, 11, 22), (a.value, b.value, a.observed, b.observed))

    def test_repeated_builds_do_not_share_module_mutable_state(self):
        p = self.case("a", 11)
        a = build_authored_model(p, "build")
        b = build_authored_model(p, "build")
        self.assertEqual((1, 1), (a.count, b.count))

    def test_source_dictionary_keepalive_is_preserved_as_object(self):
        p = self.case("dictionary", 11)
        text = p.read_text().replace("keepalive=()", "keepalive={'retained':object()}")
        p.write_text(text)
        m = build_authored_model(p, "build")
        self.assertIsInstance(m.keepalive, dict)
        self.assertIn("retained", m.keepalive)

    def test_none_returning_installer_and_callback_keep_original_dict(self):
        from types import SimpleNamespace
        from benchmark_templates.source_loading import install_authored_laws, retain_objects

        folder = self.root / "installer"
        folder.mkdir()
        (folder / "source_laws.py").write_text(
            "def install(m):\n    m.keepalive['count']=0\n    def after(t,dt):m.keepalive['count']+=1\n    m.after_step=after\n"
        )
        original = {}
        m = SimpleNamespace(
            keepalive=original, before_step=None, after_step=None, commit_inputs=None
        )
        self.assertIs(install_authored_laws(folder / "source_laws.py", "install", m), m)
        retain_objects(m, object())
        m.after_step(0, 0.1)
        self.assertIs(m.keepalive, original)
        self.assertEqual(1, original["count"])

    def test_existing_module_and_path_restored_after_factory_error(self):
        old = sys.modules.get("source_laws")
        sentinel = types.ModuleType("source_laws")
        sys.modules["source_laws"] = sentinel
        before = sys.path[:]
        try:
            with self.assertRaisesRegex(RuntimeError, "factory failed"):
                build_authored_model(self.case("bad", 0, True), "build")
            self.assertIs(sys.modules["source_laws"], sentinel)
            self.assertEqual(before, sys.path)
        finally:
            if old is None:
                sys.modules.pop("source_laws", None)
            else:
                sys.modules["source_laws"] = old


if __name__ == "__main__":
    unittest.main()
