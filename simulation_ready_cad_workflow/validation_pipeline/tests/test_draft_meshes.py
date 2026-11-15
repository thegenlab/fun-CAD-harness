"""Cached draft meshes are stored once per project, keep every recorded path, and only recent attempts keep OBJ files."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ..slim_harness.draft_meshes import (REMOVED_NOTE, declared_sources, is_derived_mesh, prune_older_attempt_meshes,
                                         revision_copy_ignore, share_derived_meshes)

ATTEMPTS = '.cadharness-draft/.runtime_runs/author_tools/'


def write(root, files):
    for name, data in files.items():
        path = Path(root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def listing(root):
    root = Path(root)
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob('*')) if path.is_file() and not path.is_symlink()}


class DerivedMeshTests(unittest.TestCase):
    def test_only_cached_geometry_of_an_attempt_is_a_derived_mesh(self):
        for name in ('build-a/geometry/body-001.obj', 'check-b/geometry/source_components/body-001-solid-001.obj',
                     'build-a/geometry/assembly.GLB', 'probe-c/cost-probe/geometry/body.obj'):
            self.assertTrue(is_derived_mesh(ATTEMPTS + name), name)
        for name in (ATTEMPTS + 'build-a/design/body-001.step', ATTEMPTS + 'build-a/geometry/geometry.json',
                     ATTEMPTS + 'build-a/design/part.obj', ATTEMPTS + 'geometry/body.obj',
                     ATTEMPTS + 'sim-a/tests/t1/physical.trace.data/samples.f64',
                     'geometry/body.obj', 'part.obj', '.cadharness-draft/build/geometry/body.obj'):
            self.assertFalse(is_derived_mesh(name), name)


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name).resolve() / 'runs'
        self.previous = self.run / 'r01/design'
        self.design = self.run / 'r02/design'
        self.files = {
            'model.py': b'source', 'part.step': b'declared step',
            'scene.json': json.dumps({'bodies': [{'id': 'a', 'source': 'part.step'},
                {'id': 'b', 'source': ATTEMPTS + 'build-a/geometry/declared.obj'}]}).encode(),
            '__pycache__/model.pyc': b'cache', '.presim_gate/gate.json': b'{}',
            ATTEMPTS + 'build-a/result.json': b'{"status": "BUILT"}',
            ATTEMPTS + 'build-a/design/part.step': b'step snapshot',
            ATTEMPTS + 'build-a/geometry/geometry.json': b'{"bodies": []}',
            ATTEMPTS + 'build-a/geometry/body.obj': b'o' * 4096,
            ATTEMPTS + 'build-a/geometry/source_components/body-solid.obj': b'o' * 4096,
            ATTEMPTS + 'build-a/geometry/assembly.glb': b'g' * 2048,
            ATTEMPTS + 'build-a/geometry/declared.obj': b'declared mesh source',
            ATTEMPTS + 'sim-b/tests/t1/physical.trace.data/samples.f64': b'trace',
            '.cadharness-draft/.runtime_runs/projects/own/construction-evidence/build-a.json': b'{}',
        }
        write(self.previous, self.files)
        self.meshes = [ATTEMPTS + 'build-a/geometry/body.obj', ATTEMPTS + 'build-a/geometry/assembly.glb',
                       ATTEMPTS + 'build-a/geometry/source_components/body-solid.obj']

    def revise(self, previous=None, design=None):
        previous, design = previous or self.previous, design or self.design
        design.parent.mkdir(parents=True)
        shutil.copytree(previous, design, ignore=revision_copy_ignore(previous))
        return share_derived_meshes(previous, design)

    def test_both_revisions_keep_every_path_and_meshes_are_stored_once(self):
        before = listing(self.previous)
        report = self.revise()
        expected = {name: data for name, data in self.files.items()
                    if not name.startswith(('__pycache__', '.presim_gate'))}
        self.assertEqual(listing(self.design), expected)
        self.assertEqual(listing(self.previous), before, 'Draft records name the path in the revision that made the attempt')
        self.assertEqual(report, {'files': 3, 'bytes': 4096 + 4096 + 2048})
        for name in self.meshes:
            earlier, newer = os.stat(self.previous / name), os.stat(self.design / name)
            self.assertEqual((earlier.st_ino, earlier.st_nlink), (newer.st_ino, 2), 'One copy of the bytes, two paths')
        for name in ('model.py', ATTEMPTS + 'build-a/design/part.step', ATTEMPTS + 'build-a/result.json'):
            self.assertNotEqual(os.stat(self.previous / name).st_ino, os.stat(self.design / name).st_ino, name)

    def test_a_chain_of_revisions_still_stores_each_mesh_once(self):
        self.revise()
        third = self.run / 'r03/design'
        self.assertEqual(self.revise(self.design, third)['files'], 3)
        for name in self.meshes:
            self.assertEqual(os.stat(third / name).st_nlink, 3)
            self.assertEqual((self.previous / name).read_bytes(), (third / name).read_bytes())

    def test_declared_mesh_source_is_copied_not_shared(self):
        declared = ATTEMPTS + 'build-a/geometry/declared.obj'
        self.assertIn((self.previous / declared).resolve(), declared_sources(self.previous))
        self.revise()
        self.assertEqual((self.previous / declared).read_bytes(), b'declared mesh source')
        self.assertEqual((self.design / declared).read_bytes(), b'declared mesh source')
        self.assertNotEqual(os.stat(self.previous / declared).st_ino, os.stat(self.design / declared).st_ino)

    def test_an_existing_file_in_the_new_revision_is_never_replaced(self):
        self.design.parent.mkdir(parents=True)
        shutil.copytree(self.previous, self.design, ignore=revision_copy_ignore(self.previous))
        (self.design / self.meshes[0]).write_bytes(b'already authored')
        report = share_derived_meshes(self.previous, self.design)
        self.assertEqual(report['files'], 2)
        self.assertEqual((self.design / self.meshes[0]).read_bytes(), b'already authored')
        self.assertEqual((self.previous / self.meshes[0]).read_bytes(), b'o' * 4096)

    def test_a_mesh_that_cannot_be_linked_is_copied(self):
        self.design.parent.mkdir(parents=True)
        shutil.copytree(self.previous, self.design, ignore=revision_copy_ignore(self.previous))
        with patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.draft_meshes.os.link',
                   side_effect=OSError('links are not supported here')):
            self.assertEqual(share_derived_meshes(self.previous, self.design), {'files': 0, 'bytes': 0})
        for name in self.meshes:
            self.assertEqual((self.design / name).read_bytes(), (self.previous / name).read_bytes())
            self.assertNotEqual(os.stat(self.design / name).st_ino, os.stat(self.previous / name).st_ino)

    def test_selecting_what_to_share_changes_nothing(self):
        before = listing(self.previous)
        revision_copy_ignore(self.previous)
        self.assertEqual(listing(self.previous), before)

    @unittest.skipIf(os.name == 'nt', 'POSIX links')
    def test_linked_attempt_folders_are_not_shared(self):
        outside = self.run.parent / 'outside/geometry'
        write(outside, {'body.obj': b'OTHER_PROJECT'})
        os.symlink(outside.parent, self.previous / ATTEMPTS / 'linked-attempt')
        os.symlink(outside / 'body.obj', self.previous / ATTEMPTS / 'build-a/geometry/linked.obj')
        self.assertEqual(self.revise()['files'], 3)
        self.assertEqual((outside / 'body.obj').read_bytes(), b'OTHER_PROJECT')
        self.assertEqual(os.stat(outside / 'body.obj').st_nlink, 1)

    def test_design_without_a_draft_copies_as_before(self):
        shutil.rmtree(self.previous / '.cadharness-draft')
        self.assertEqual(self.revise(), {'files': 0, 'bytes': 0})
        self.assertEqual(listing(self.design), {'model.py': b'source', 'part.step': b'declared step',
                                                'scene.json': self.files['scene.json']})


class RecentAttemptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name).resolve() / 'work'
        self.attempts = self.work / ATTEMPTS
        for index, name in enumerate(('build-e', 'check-a', 'build-c', 'check-d', 'build-b')):
            # Folder names and modification times say nothing about age after restaging.
            write(self.attempts / name, {
                'capture.json': json.dumps({'started_ms': 1000 + index}).encode(),
                'result.json': b'{"status": "BUILT"}', 'design/body.step': b'step snapshot',
                'geometry/geometry.json': b'{}', 'geometry/body.obj': b'o' * 4096,
                'geometry/source_components/body-solid.obj': b'o' * 4096, 'geometry/assembly.glb': b'g' * 2048})
        self.current = self.attempts / 'check-new'
        self.current.mkdir()
        write(self.work, {'model.py': b'source', 'scene.json': b'{"bodies": []}'})

    def meshes(self):
        return sorted(path.relative_to(self.attempts).parts[0] for path in self.attempts.rglob('*.obj'))

    def test_current_and_two_most_recent_attempts_keep_their_obj_meshes(self):
        prune_older_attempt_meshes(self.attempts, self.current, self.work)
        self.assertEqual(self.meshes(), ['build-b', 'build-b', 'check-d', 'check-d'])
        for name in ('build-e', 'check-a', 'build-c'):
            kept = listing(self.attempts / name)
            note = kept.pop(REMOVED_NOTE).decode()
            self.assertIn('2 cached OBJ meshes (8192 bytes)', note)
            self.assertEqual(sorted(kept), ['capture.json', 'design/body.step', 'geometry/assembly.glb',
                                            'geometry/geometry.json', 'result.json'])
        self.assertFalse((self.attempts / 'check-d' / REMOVED_NOTE).exists())

    def test_running_command_and_saved_simulation_request_keep_their_meshes(self):
        (self.attempts / 'build-e/result.json').unlink()  # Still running: no result yet.
        write(self.attempts / 'check-a', {'tests/t1/request.json': b'{"scene": {}}'})
        prune_older_attempt_meshes(self.attempts, self.current, self.work)
        self.assertEqual(self.meshes(), ['build-b', 'build-b', 'build-e', 'build-e', 'check-a', 'check-a',
                                         'check-d', 'check-d'])
        self.assertFalse((self.attempts / 'build-e' / REMOVED_NOTE).exists())
        self.assertFalse((self.attempts / 'check-a' / REMOVED_NOTE).exists())
        self.assertTrue((self.attempts / 'build-c' / REMOVED_NOTE).is_file())
        # Once the running command has finished, a later command reduces it.
        write(self.attempts / 'build-e', {'result.json': b'{"status": "BUILT"}'})
        prune_older_attempt_meshes(self.attempts, self.current, self.work)
        self.assertNotIn('build-e', self.meshes())

    def test_repeating_the_command_changes_nothing_more(self):
        prune_older_attempt_meshes(self.attempts, self.current, self.work)
        before = listing(self.attempts)
        prune_older_attempt_meshes(self.attempts, self.current, self.work)
        self.assertEqual(listing(self.attempts), before)

    def test_a_mesh_the_scene_names_as_its_source_is_kept(self):
        write(self.work, {'scene.json': json.dumps({'bodies': [
            {'id': 'a', 'source': ATTEMPTS + 'build-e/geometry/body.obj'}]}).encode()})
        prune_older_attempt_meshes(self.attempts, self.current, self.work)
        self.assertTrue((self.attempts / 'build-e/geometry/body.obj').is_file())
        self.assertFalse((self.attempts / 'build-e/geometry/source_components/body-solid.obj').exists())

    def test_unreadable_capture_never_fails_the_command(self):
        (self.attempts / 'check-a/capture.json').write_bytes(b'{interrupted')
        (self.attempts / 'build-c/capture.json').unlink()
        prune_older_attempt_meshes(self.attempts, self.current, self.work)
        self.assertEqual(self.meshes(), ['build-b', 'build-b', 'check-d', 'check-d'])
        prune_older_attempt_meshes(self.attempts / 'absent', self.current, self.work)

    @unittest.skipIf(os.name == 'nt', 'POSIX links')
    def test_links_inside_the_draft_are_left_alone(self):
        outside = self.work.parent / 'outside'
        write(outside, {'geometry/body.obj': b'OTHER'})
        os.symlink(outside, self.attempts / 'build-linked')
        os.symlink(outside / 'geometry', self.attempts / 'build-e/geometry/nested')
        prune_older_attempt_meshes(self.attempts, self.current, self.work)
        self.assertEqual((outside / 'geometry/body.obj').read_bytes(), b'OTHER')


if __name__ == '__main__':
    unittest.main()
