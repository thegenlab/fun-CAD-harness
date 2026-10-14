"""Explicit snapshot paths; original evidence is never rewritten."""
import json
import posixpath
from functools import lru_cache
from pathlib import Path

MARKER = '.hf-import.json'


def imported(workspace):
    path = Path(workspace) / MARKER
    if not path.is_file():
        return None
    stat = path.stat()
    return _load(str(path), (stat.st_mtime_ns, stat.st_size, stat.st_ino))


@lru_cache(maxsize=64)
def _load(path, stamp):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def mapped_path(workspace, value):
    workspace = Path(workspace).resolve()
    record = imported(workspace)
    if not record:
        return None
    name = str(value).replace('\\', '/')
    mapping = record['path_mapping']
    if name in mapping:
        return workspace / mapping[name]
    path = Path(value)
    # Imports often live inside the producing checkout (pulling one's own main).
    # A receiving path must not be interpreted again as a producing-root path.
    if path.is_absolute() and path.is_relative_to(workspace):
        return path
    source = record['source_workspace'].replace('\\', '/').rstrip('/')
    if name.startswith(source + '/'):
        return workspace / name[len(source) + 1:]
    # Local relative references and already resolved receiving paths are valid.
    if not path.is_absolute() and not (len(name) > 1 and name[1] == ':'):
        return workspace / name
    raise ValueError(f'File not included in imported snapshot: {value}')


def resolve(workspace, value, *, relative_to=None):
    name = str(value).replace('\\', '/')
    absolute = name.startswith('/') or (len(name) > 1 and name[1] == ':')
    if not absolute and relative_to is not None:
        record = imported(workspace)
        if record:
            base = Path(relative_to).resolve().relative_to(Path(workspace).resolve()).as_posix()
            original_base = next((old for old, new in record['path_mapping'].items() if new == base), None)
            if original_base is None and not base.startswith('.hf_external/'):
                original_base = record['source_workspace'].replace('\\', '/').rstrip('/') + '/' + base
            if original_base:
                return mapped_path(workspace, posixpath.normpath(original_base + '/' + name))
        return (Path(relative_to) / name).resolve()
    mapped = mapped_path(workspace, value)
    return mapped if mapped is not None else (Path(workspace) / value).resolve()


def workspace_for(path):
    path = Path(path).resolve()
    for parent in (path, *path.parents):
        if (parent / MARKER).is_file():
            return parent
    return None


def binding_paths(workspace, bindings):
    return {str(resolve(workspace, path)): digest for path, digest in bindings.items()}


def implementation_matches(workspace, saved, current):
    if saved == current:
        return True
    record = imported(workspace)
    if not record:
        return saved == current
    old_root = record['source_workspace'].replace('\\', '/').rstrip('/') + '/'
    code_root = Path(__file__).resolve().parents[2].as_posix() + '/'
    def relative(values, root):
        return {str(key).replace('\\', '/').removeprefix(root): value for key, value in values.items()}
    return relative(saved, old_root) == relative(current, code_root)
