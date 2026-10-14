"""独立工程的Python导入作用域，避免同名source_laws.py在多设计间串用。"""

from pathlib import Path
from contextlib import contextmanager
import importlib.util, sys, itertools

_counter = itertools.count()


class SourceScope:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.names = {p.stem for p in self.folder.glob("*.py") if p.stem != "__init__"}
        self.names.update(
            p.name for p in self.folder.iterdir() if p.is_dir() and (p / "__init__.py").exists()
        )
        self.modules = {}

    def belongs(self, name):
        return name.split(".", 1)[0] in self.names

    @contextmanager
    def activate(self):
        # 同进程顺序调用；并发仿真使用独立进程，不共享Python全局导入表。
        saved = {k: v for k, v in tuple(sys.modules.items()) if self.belongs(k)}
        old_path = sys.path[:]
        try:
            for key in saved:
                sys.modules.pop(key, None)
            sys.modules.update(self.modules)
            sys.path.insert(0, str(self.folder))
            yield
        finally:
            current = {k: v for k, v in tuple(sys.modules.items()) if self.belongs(k)}
            self.modules.update(current)
            for key in current:
                sys.modules.pop(key, None)
            sys.modules.update(saved)
            sys.path[:] = old_path

    def wrap(self, callback):
        def call(*args, **kwargs):
            with self.activate():
                return callback(*args, **kwargs)

        return call


def _call_source(entry_path, entrypoint, *args, **kwargs):
    path = Path(entry_path).resolve(strict=True)
    scope = SourceScope(path.parent)
    name = "_benchmark_authored_" + str(next(_counter))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with scope.activate():
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            result = getattr(module, entrypoint)(*args, **kwargs)
        finally:
            sys.modules.pop(name, None)
    return result, scope, module


def _attach_scope(model, scope, module):
    for hook in ("before_step", "after_step", "commit_inputs"):
        callback = getattr(model, hook, None)
        if callback is not None:
            setattr(model, hook, scope.wrap(callback))
    # 原回调可能继续读写keepalive，连容器类型也原样保留。
    scopes = getattr(model, "_authored_python_scopes", [])
    scopes.append((scope, module))
    model._authored_python_scopes = scopes
    return model


def build_authored_model(entry_path, entrypoint, *args, **kwargs):
    """调用原模型工厂（必须返回model），隔离工厂及后续回调的工程模块。"""
    model, scope, module = _call_source(entry_path, entrypoint, *args, **kwargs)
    if model is None:
        raise ValueError("源工厂没有返回模型；原地安装内力请使用install_authored_laws")
    return _attach_scope(model, scope, module)


def install_authored_laws(entry_path, entrypoint, model, *args, **kwargs):
    """在已载入模型上调用原内力installer；不要求原installer返回model。

    这是显式原地安装契约，不猜测工厂含义。原installer返回的附加对象也保留引用。
    """
    result, scope, module = _call_source(entry_path, entrypoint, model, *args, **kwargs)
    _attach_scope(model, scope, module)
    if result is not None and result is not model:
        retain_objects(model, result)
    return model


def retain_objects(model, *objects):
    """保留judge/公共观察器对象，不改变源工程自己的keepalive容器及其访问语义。"""
    retained = getattr(model, "_benchmark_retained_objects", [])
    retained.extend(objects)
    model._benchmark_retained_objects = retained
