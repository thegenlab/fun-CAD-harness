"""Per-project Linux accounts that keep concurrent experiments out of each other's files.

A trusted runtime configuration selects this with an ``isolation`` section whose
``backend`` is ``unix_user``. The Harness runs as root. Each project's author and
reviewer clients, and the authored programs the Harness executes for it, run as
that project's own account directly inside the project directory. Ordinary file
permissions, not a copied or virtualised workspace, keep other projects,
experiments, credentials and private benchmark material unreadable.

Inside ``private_root`` every entry that does not lead to the projects or a
configured read path loses its permissions for other accounts; owners and root
are unaffected.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

CODE_ROOT = Path(__file__).resolve().parents[3]
ENV_PROJECT = 'CADHARNESS_ISOLATED_PROJECT'
ENV_VISUAL = 'CADHARNESS_AUTHOR_VISUAL_POLICY'
STATE = '.isolation'
PRIVATE_CODE = ('benchmark_templates/bindings', 'benchmark_templates/reviews',
                'benchmark_templates/reference_models', 'docs', '.git')
_DROPPED_ENV = ('CODEX_HOME', 'CADHARNESS_SANDBOX_CONFIG', 'CADHARNESS_SANDBOX_STOP_FILE')
_FIELDS = {'backend', 'uid_range', 'private_root', 'read_paths', 'home_files', 'executables',
           'native_visual_capability'}


def config_path(workspace):
    selected = os.environ.get('CADHARNESS_SANDBOX_CONFIG')
    path = Path(selected).expanduser() if selected else Path(workspace) / '.runtime_runs/benchmark_sandbox/runtime.json'
    if not path.is_absolute():
        raise ValueError('CADHARNESS_SANDBOX_CONFIG must name an absolute trusted configuration file')
    return path


def _section(path):
    try:
        config = json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    return config.get('isolation') if isinstance(config, dict) else None


def load(workspace):
    """Return the configured isolation, or None when another backend is selected."""
    section = _section(config_path(workspace))
    return None if section is None else Isolation(workspace, section)


def turns_for(context):
    """Resolve a bound worker's launches at turn time; None keeps the formal sandbox transport."""
    section = _section(context['config_path'])
    if section is None:
        return None
    return Isolation(context['workspace'], section).turns(
        context['project'], context['role'], context['client'], context.get('author_visual_policy', 'full'))


def harness_umask(workspace):
    """Keep every file the root Harness creates out of other accounts' reach from process start."""
    if os.name == 'posix' and os.geteuid() == 0 and load(workspace) is not None:
        os.umask(0o007)


def confined_compute(workspace, project, env):
    """Runner options for authored code: {} when already confined, None without isolation."""
    project = Path(project).resolve()
    if os.name == 'posix' and os.geteuid() != 0 and os.environ.get(ENV_PROJECT) == str(project):
        return {}
    isolation = load(workspace)
    return None if isolation is None else isolation.compute_options(project, env)


def _absolute(value, label):
    if not isinstance(value, str) or not Path(value).is_absolute() or Path(value).parent == Path(value):
        raise ValueError(f'{label} must be an absolute non-root path')
    return Path(value)


class Isolation:
    def __init__(self, workspace, section):
        if not isinstance(section, dict) or section.get('backend') != 'unix_user':
            raise ValueError('The isolation backend must be unix_user')
        if set(section) - _FIELDS:
            raise ValueError('Unknown isolation fields: ' + ', '.join(sorted(set(section) - _FIELDS)))
        low, high = section.get('uid_range') or (None, None)
        if not all(type(value) is int for value in (low, high)) or not 1000 < low <= high:
            raise ValueError('uid_range must be two integers above the system account range')
        self.workspace = Path(workspace).resolve()
        self.uid_range = (low, high)
        self.private_root = _absolute(section.get('private_root'), 'private_root')
        dependencies = self.workspace / '.runtime_runs/python_deps'
        self.python_path = [CODE_ROOT, *([dependencies] if dependencies.is_dir() else [])]
        self.read_paths = [_absolute(path, 'read path') for path in section.get('read_paths', [])] + self.python_path
        self.home_files = {client: {relative: _absolute(source, 'home file source') for relative, source in files.items()}
                           for client, files in (section.get('home_files') or {}).items()}
        if any(Path(relative).is_absolute() or '..' in Path(relative).parts
               for files in self.home_files.values() for relative in files):
            raise ValueError('Home file targets must stay inside the project account home')
        self.executables = {client: _absolute(path, 'client executable')
                            for client, path in (section.get('executables') or {}).items()}
        proof = section.get('native_visual_capability')
        self.native_visual_capability = _absolute(proof, 'native_visual_capability') if proof else None
        if self.private_root not in self.workspace.parents:
            raise ValueError('The Harness workspace must be inside private_root')
        if CODE_ROOT == self.workspace or self.workspace in CODE_ROOT.parents:
            raise ValueError('The importable code root must be outside the private Harness workspace')

    def account(self, project):
        """Create or reuse the project's account; its uid doubles as its private group id."""
        import grp
        import pwd
        name = 'cadh' + hashlib.sha256(str(project).encode('utf-8')).hexdigest()[:12]
        try:
            return name, pwd.getpwnam(name).pw_uid
        except KeyError:
            pass
        low, high = self.uid_range
        used = {row.pw_uid for row in pwd.getpwall()} | {row.gr_gid for row in grp.getgrall()}
        owned = project.stat().st_gid
        # A recreated container loses /etc/passwd but keeps the project's numeric group.
        if low <= owned <= high and owned not in used:
            uid = owned
        else:
            used |= {child.stat().st_gid for child in project.parent.iterdir() if child.is_dir()}
            uid = next((value for value in range(low, high + 1) if value not in used), None)
            if uid is None:
                raise RuntimeError('No free project account id remains in the configured uid_range')
        subprocess.run(['groupadd', '-g', str(uid), name], check=True, capture_output=True)
        subprocess.run(['useradd', '-M', '-N', '-u', str(uid), '-g', str(uid), '-d', str(project / STATE),
                        '-s', '/usr/sbin/nologin', name], check=True, capture_output=True)
        return name, uid

    def prepare(self, project):
        if os.name != 'posix' or os.geteuid() != 0:
            raise RuntimeError('unix_user isolation requires the Harness to run as root on Linux')
        os.umask(0o007)
        self.secure_host()
        name, uid = self.account(project)
        marker = project / STATE / 'prepared.json'
        try:
            prepared = json.loads(marker.read_text(encoding='utf-8')).get('gid')
        except (OSError, ValueError):
            prepared = None
        if prepared != uid:
            _share_tree(project, uid)
            marker.parent.mkdir(exist_ok=True)
            marker.write_text(json.dumps({'account': name, 'gid': uid}) + '\n', encoding='utf-8')
        for path in (project, marker.parent):
            os.chown(path, 0, uid)
            os.chmod(path, 0o2770)
        return name, uid

    def secure_host(self):
        """Inside private_root, project accounts reach only projects and the runtime; all else is closed."""
        root = self.private_root
        runtime = self.workspace / '.runtime_runs'
        projects, tools = runtime / 'projects', runtime / 'author_tools'
        for path in (projects, tools):
            path.mkdir(parents=True, exist_ok=True)
        for path in self.read_paths:
            if not path.is_dir() or stat.S_IMODE(path.stat().st_mode) & 0o005 != 0o005:
                raise ValueError(f'Isolation read path is not a world-readable directory: {path}')
        reachable = [projects, tools, *self.read_paths]

        def readable(path):
            return any(path == target or target in path.parents for target in self.read_paths)

        def on_chain(path):
            return any(path == target or path in target.parents for target in reachable)

        chain = {parent for target in reachable for parent in target.parents
                 if parent == root or root in parent.parents}
        for directory in chain:
            if not readable(directory):
                _gate(directory)
                _prune(directory, on_chain)
        _gate(projects)
        _prune(projects, lambda path: False)
        _prune(tools, lambda path: False)
        os.chmod(tools, 0o1733)
        for relative in PRIVATE_CODE:
            path = CODE_ROOT / relative
            if path.exists() and not path.is_symlink():
                _close(path)
        for files in self.home_files.values():
            for source in files.values():
                _close(source)

    def turns(self, project, role, client, author_visual_policy='full'):
        return ProjectTurns(self, Path(project).resolve(), role, client, author_visual_policy)

    def compute_options(self, project, env):
        _name, uid = self.prepare(project)
        home, tmp = _role_dirs(project, 'compute', uid)
        return {'env': _account_environment(os.environ if env is None else env, home, tmp, project, self.python_path),
                'popen_options': _popen_options(uid)}


class ProjectTurns:
    """Client launches for one project and role."""

    def __init__(self, isolation, project, role, client, author_visual_policy):
        self.isolation, self.project, self.role, self.client = isolation, project, role, client
        self.author_visual_policy = author_visual_policy

    @property
    def executable(self):
        return self.isolation.executables.get(self.client)

    def launch(self):
        name, uid = self.isolation.prepare(self.project)
        home, tmp = _role_dirs(self.project, self.role, uid)
        for relative, source in self.isolation.home_files.get(self.client, {}).items():
            target = home / relative
            for parent in reversed(Path(relative).parents[:-1]):
                _owned_dir(home / parent, uid)
            shutil.copyfile(source, target)
            os.chown(target, uid, uid)
            os.chmod(target, 0o600)
        return TurnLaunch(self, name, uid, home, tmp)


class TurnLaunch:
    def __init__(self, turns, name, uid, home, tmp):
        self.turns, self.name, self.uid, self.home, self.tmp = turns, name, uid, home, tmp
        self.popen_options = _popen_options(uid)
        self.codex_home = home / '.codex'
        self._visual_holder = None

    def environment(self, base):
        environment = _account_environment(base, self.home, self.tmp, self.turns.project, self.turns.isolation.python_path)
        environment.update(USER=self.name, LOGNAME=self.name)
        if self.turns.role == 'author' and self.turns.author_visual_policy != 'full':
            environment[ENV_VISUAL] = self.turns.author_visual_policy
        return environment

    def codex_command(self, command, output_dir, python_executable):
        """Install the native no-visual guard in this author's own Codex home."""
        from ..benchmark_sandbox import native_visual_hook
        config = self.codex_home / 'config.toml'
        if self.turns.role != 'author' or self.turns.author_visual_policy == 'full':
            config.unlink(missing_ok=True)
            return command
        if not command or command[-1] != '-' or 'exec' not in command:
            raise ValueError('No-visual control requires an original Codex exec turn')
        proof = self.turns.isolation.native_visual_capability
        if proof is None:
            raise RuntimeError('No-visual Codex turns require isolation.native_visual_capability')
        executable = shutil.which(command[0], path=self.environment(os.environ).get('PATH'))
        if executable is None:
            raise RuntimeError(f'Codex executable is not available to the project account: {command[0]}')
        command = [str(Path(executable).resolve()), *command[1:]]
        holder = Path(output_dir) / '.visual-guard'
        holder.mkdir()
        folder, hooks, flags = native_visual_hook.place_guard(holder, python_executable)
        probe_home = holder / 'probe-home'
        (probe_home / '.codex').mkdir(parents=True)
        capabilities = native_visual_hook.capability(command[0], {
            **os.environ, 'HOME': str(probe_home), 'CODEX_HOME': str(probe_home / '.codex')}, str(probe_home))
        native_visual_hook.verify_proof(json.loads(proof.read_text(encoding='utf-8')), capabilities)
        (folder / 'capability.json').write_text(json.dumps(capabilities | {
            'native_denial_verified': True, 'verified_by': 'matching_native_preflight',
            'guard_configured': True, 'coverage': 'maintained_visual_interfaces'}, indent=2), encoding='utf-8')
        for path, mode in ((holder, 0o2750), (folder, 0o2750), (folder / 'native_visual_hook.py', 0o440),
                           (folder / 'visual_policy.py', 0o440), (folder / 'audit.jsonl', 0o660)):
            os.chown(path, 0, self.uid)
            os.chmod(path, mode)
        _owned_dir(self.codex_home, self.uid)
        config.unlink(missing_ok=True)
        config.write_text(hooks + '\n', encoding='utf-8')
        os.chown(config, 0, self.uid)
        os.chmod(config, 0o440)
        self._visual_holder = holder
        position = command.index('resume') if 'resume' in command else len(command) - 1
        return [*command[:position], *flags, *command[position:]]

    def finish(self, output_dir):
        """Fail the turn unless every native command execution was observed by the guard."""
        if self._visual_holder is None:
            return None
        from ..benchmark_sandbox.native_visual_hook import audit_summary, checked_audit
        from ..mechanism.io import write_json_atomic
        summary = audit_summary(self._visual_holder)
        if summary is not None:
            write_json_atomic(Path(output_dir) / 'native-visual-guard.json', summary)
        events = []
        for line in (Path(output_dir) / 'events.jsonl').read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        proof = json.loads(self.turns.isolation.native_visual_capability.read_text(encoding='utf-8'))
        return checked_audit(summary, proof, [event for event in events if isinstance(event, dict)])


def _role_dirs(project, role, uid):
    base = Path(project) / STATE / role
    base.mkdir(parents=True, exist_ok=True)
    os.chown(base, 0, uid)
    os.chmod(base, 0o2770)
    return _owned_dir(base / 'home', uid), _owned_dir(base / 'tmp', uid)


def _owned_dir(path, uid):
    path.mkdir(exist_ok=True)
    os.chown(path, uid, uid)
    os.chmod(path, 0o700)
    return path


def _account_environment(base, home, tmp, project, python_path):
    environment = {key: value for key, value in dict(base).items() if key not in _DROPPED_ENV}
    environment.update(HOME=str(home), TMPDIR=str(tmp), PYTHONPATH=os.pathsep.join(str(path) for path in python_path),
                       **{ENV_PROJECT: str(Path(project).resolve())})
    return environment


def _popen_options(uid):
    return {'user': uid, 'group': uid, 'extra_groups': [], 'umask': 0o007}


def _share_tree(project, gid):
    for current, directories, files in os.walk(project):
        for name in [*directories, *files]:
            path = Path(current) / name
            try:
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    continue
                directory = stat.S_ISDIR(info.st_mode)
                mode = stat.S_IMODE(info.st_mode) & ~0o007
                mode |= 0o060 | (0o010 if directory or mode & 0o100 else 0) | (0o2000 if directory else 0)
                os.chown(path, -1, gid)
                os.chmod(path, mode)
            except FileNotFoundError:
                # Live ledgers are replaced atomically; their temporary names vanish mid-walk.
                continue


def _gate(path):
    info = path.lstat()
    mode = stat.S_IMODE(info.st_mode)
    if not stat.S_ISLNK(info.st_mode) and mode & 0o007 != 0o001:
        os.chmod(path, (mode & ~0o007) | 0o001)


def _close(path):
    info = Path(path).lstat()
    mode = stat.S_IMODE(info.st_mode)
    if not stat.S_ISLNK(info.st_mode) and mode & 0o007:
        os.chmod(path, mode & ~0o007)


def _prune(directory, keep):
    for child in directory.iterdir():
        if not keep(child):
            _close(child)
