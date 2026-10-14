"""Full-cadence Slim trace storage: JSONL samples, uint64 offsets and float64 columns.

Only the existing inline trace and this program-owned disk representation are
supported. No physical state is synthesized, reduced or resampled here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
import copy
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import struct
import threading
import time

import numpy as np

from ..mechanism.io import write_json_atomic

TRACE_SCHEMA = "slim_physical_trace_v2"
STORAGE_SCHEMA = "slim_trace_storage_v1"
PREFIX_SCHEMA = "slim_trace_prefix_v1"
_FLOAT = np.dtype("<f8")
_OFFSET = np.dtype("<u8")


class TraceStorageError(ValueError):
    pass


def _drop_clean_file_cache(stream):
    # D's cgroup accounts file cache too. Advise only on this trace's files.
    if hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def file_digest(path: Path, *, drop_cache=False) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
        if drop_cache:
            _drop_clean_file_cache(stream)
    return value.hexdigest()


def _files(trace_path, storage):
    parent = Path(trace_path).resolve().parent
    from ..snapshot_paths import workspace_for, resolve
    workspace = workspace_for(trace_path)
    return {name: (resolve(workspace, record['path'], relative_to=parent) if workspace else
                   (parent / record['path']).resolve()) for name, record in storage['files'].items()}


class SampleRows(Sequence):
    def __init__(self, sample_path, index_path, count, indices=None, *, sample_byte_end=None):
        self.sample_path, self.index_path = Path(sample_path), Path(index_path)
        self.count = count
        self.indices = range(count) if indices is None else indices
        self.sample_byte_end = sample_byte_end

    def __len__(self):
        return len(self.indices)

    def _offset(self, index):
        if index == self.count and self.sample_byte_end is not None:
            return self.sample_byte_end
        with self.index_path.open("rb") as stream:
            stream.seek(index * 8)
            value = stream.read(8)
        if len(value) != 8:
            raise TraceStorageError("Sample offset is missing")
        return struct.unpack("<Q", value)[0]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return SampleRows(self.sample_path, self.index_path, self.count, self.indices[index],
                              sample_byte_end=self.sample_byte_end)
        source_index = self.indices[operator.index(index)]
        start, end = self._offset(source_index), self._offset(source_index + 1)
        if self.sample_byte_end is not None and not 0 <= start < end <= self.sample_byte_end:
            raise TraceStorageError("Sample exceeds its confirmed prefix boundary")
        with self.sample_path.open("rb") as stream:
            stream.seek(start)
            encoded = stream.read(end - start)
        if len(encoded) != end - start:
            raise TraceStorageError("Recorded sample is truncated")
        return json.loads(encoded)

    def __iter__(self):
        if not self.indices:
            return
        if self.indices.step != 1:
            for index in range(len(self)):
                yield self[index]
            return
        with self.sample_path.open("rb") as stream:
            stream.seek(self._offset(self.indices.start))
            for _ in self.indices:
                remaining = (-1 if self.sample_byte_end is None else self.sample_byte_end-stream.tell())
                encoded = stream.readline(remaining) if remaining != 0 else b""
                if not encoded:
                    raise TraceStorageError("Recorded sample sequence is truncated")
                yield json.loads(encoded)


class SignalColumns(Mapping):
    def __init__(self, path, names, count, *, stride=None):
        self.path, self.names, self.count = Path(path), tuple(names), count
        self.columns = {name: index for index, name in enumerate(self.names)}
        self.stride = count if stride is None else stride

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def __getitem__(self, name):
        index = self.columns[name]
        # Map one contiguous column, not an all-columns matrix. Temporary
        # measurement views release their mappings when the expression ends.
        return np.memmap(self.path, dtype=_FLOAT, mode="r", offset=index * self.stride * 8, shape=(self.count,))


class TraceView(Mapping):
    def __init__(self, metadata, trace_path):
        self.metadata = metadata
        self.trace_path = Path(trace_path).resolve()
        storage = metadata["storage"]
        files = _files(self.trace_path, storage)
        self.samples = SampleRows(files["samples"], files["offsets"], storage["recorded_samples"],
                                  sample_byte_end=storage.get("sample_byte_end"))
        self.signals = SignalColumns(files["signals"], storage["signal_names"], storage["recorded_samples"],
                                     stride=storage.get("column_stride_samples"))

    def __len__(self):
        return len(self.metadata) + 2

    def __iter__(self):
        yield from self.metadata
        yield "samples"
        yield "signals"

    def __getitem__(self, key):
        if key == "samples":
            return self.samples
        if key == "signals":
            return self.signals
        return self.metadata[key]


def _verify_storage(trace_path, storage):
    if storage.get("schema_version") != STORAGE_SCHEMA or storage.get("dtype") != "<f8":
        raise TraceStorageError("Unsupported trace storage representation")
    count = storage["recorded_samples"]
    names = storage["signal_names"]
    if count < 1 or count > storage["expected_samples"] or not names or len(names) != len(set(names)):
        raise TraceStorageError("Invalid recorded sample/channel inventory")
    files = _files(trace_path, storage)
    if set(files) != {"samples", "offsets", "signals"}:
        raise TraceStorageError("Trace side-file inventory is incomplete")
    expected_sizes = {"offsets": (count + 1) * 8, "signals": count * len(names) * 8}
    for name, path in files.items():
        record = storage["files"][name]
        if path.stat().st_size != record["bytes"] or path.stat().st_size != expected_sizes.get(name, record["bytes"]):
            raise TraceStorageError(f"Trace side-file size mismatch: {name}")
        if file_digest(path, drop_cache=True) != record["sha256"]:
            raise TraceStorageError(f"Trace side-file binding mismatch: {name}")
    offsets = np.memmap(files["offsets"], dtype=_OFFSET, mode="r", shape=(count + 1,))
    if int(offsets[0]) != 0 or int(offsets[-1]) != files["samples"].stat().st_size or np.any(offsets[1:] <= offsets[:-1]):
        raise TraceStorageError("Invalid sample offsets or incomplete JSONL rows")


def open_trace(path):
    path = Path(path).resolve()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != TRACE_SCHEMA:
        if "samples" in metadata and "signals" in metadata:
            return metadata  # Actual previously produced inline traces.
        raise TraceStorageError("Not a readable Slim physical trace")
    _verify_storage(path, metadata["storage"])
    return TraceView(metadata, path)


def trace_prefix_path(trace_path):
    return Path(trace_path).resolve().with_suffix(".data") / "prefix.json"


def open_trace_prefix(path):
    """Read only fsync-confirmed rows; this is never acceptance or solver state.

    Accept either the intended trace path (which may not exist after a kill)
    or its data-directory prefix.json. Unconfirmed tails remain untouched.
    Unlike open_trace, this recovery view has no full-file content attestation.
    """
    path = Path(path).resolve()
    prefix_path = path if path.name == "prefix.json" else trace_prefix_path(path)
    if not prefix_path.is_file():
        raise TraceStorageError("No readable confirmed trace prefix")
    metadata = json.loads(prefix_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != PREFIX_SCHEMA:
        raise TraceStorageError("Unsupported trace prefix record")
    from ..snapshot_paths import workspace_for, resolve
    workspace = workspace_for(prefix_path)
    trace_path = resolve(workspace, metadata['trace_path']) if workspace else Path(metadata['trace_path']).resolve()
    if trace_prefix_path(trace_path) != prefix_path:
        raise TraceStorageError("Trace prefix does not belong to its data directory")
    storage = metadata["storage"]
    count, capacity = storage["recorded_samples"], storage["expected_samples"]
    names, end = storage["signal_names"], storage["sample_byte_end"]
    if (storage.get("schema_version") != PREFIX_SCHEMA or storage.get("dtype") != "<f8"
            or not 0 < count <= capacity or storage.get("column_stride_samples") != capacity
            or storage.get("offset_count") != count or not names or len(names) != len(set(names)) or end <= 0):
        raise TraceStorageError("Invalid confirmed trace prefix inventory")
    files = _files(trace_path, storage)
    if set(files) != {"samples", "offsets", "signals"}:
        raise TraceStorageError("Trace prefix side-file inventory is incomplete")
    required = {"samples":end, "offsets":count*8, "signals":((len(names)-1)*capacity+count)*8}
    if any(files[name].stat().st_size < size for name,size in required.items()):
        raise TraceStorageError("Confirmed trace prefix data is truncated")
    offsets = np.memmap(files["offsets"], dtype=_OFFSET, mode="r", shape=(count,))
    if int(offsets[0]) != 0 or int(offsets[-1]) >= end or np.any(offsets[1:] <= offsets[:-1]):
        raise TraceStorageError("Invalid confirmed trace prefix offsets")
    # Always derive these flags here; a recovery descriptor cannot grant PASS.
    metadata.update(status="UNVERIFIED_PREFIX", physical_pass=False,
                    recovery_scope="Confirmed recorded rows only; no solver continuation or uninterrupted equivalence",
                    content_attestation="Flush boundary only; no full-file hash verification")
    return TraceView(metadata, trace_path)


def trace_artifacts(path):
    path = Path(path).resolve()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") == TRACE_SCHEMA:
        return [path, *_files(path, metadata["storage"]).values()]
    return [path]


def release_trace_cache(trace):
    if isinstance(trace, TraceView):
        for path in _files(trace.trace_path, trace.metadata["storage"]).values():
            with path.open("rb") as stream:
                _drop_clean_file_cache(stream)


def reuse_trace(source, destination):
    """Rebase a verified cache reference; preserve all existing source data files."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    trace = open_trace(source)
    if isinstance(trace, TraceView):
        metadata = copy.deepcopy(trace.metadata)
        files = _files(source, metadata["storage"])
        for name, path in files.items():
            metadata["storage"]["files"][name]["path"] = Path(os.path.relpath(path, destination.parent)).as_posix()
    else:
        metadata = trace.copy()
    metadata["reused_from"] = str(source)
    write_json_atomic(destination, metadata)
    return TraceView(metadata, destination) if isinstance(trace, TraceView) else metadata


class TraceWriter:
    def __init__(self, trace_path, expected_samples, chunk_rows=1024, *, flush_interval_seconds=1.0):
        if expected_samples < 1 or chunk_rows < 1:
            raise ValueError("Positive sample capacity and recording block size required")
        if not math.isfinite(flush_interval_seconds) or flush_interval_seconds<=0:
            raise ValueError("Positive finite trace flush interval required")
        self.trace_path = Path(trace_path).resolve()
        self.directory = self.trace_path.with_suffix(".data")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.capacity = int(expected_samples)
        self.chunk_rows = min(int(chunk_rows), self.capacity)
        self.paths = {"samples": self.directory / "samples.jsonl", "offsets": self.directory / "offsets.u64",
                      "signals": self.directory / "signals.f64"}
        self.handles = ExitStack()
        try:
            self.samples_file = self.handles.enter_context(self.paths["samples"].open("xb"))
            self.offsets_file = self.handles.enter_context(self.paths["offsets"].open("xb"))
            self.signals_file = self.handles.enter_context(self.paths["signals"].open("x+b"))
        except Exception:
            self.handles.close()
            raise
        self.count = self.flushed = self.pending = 0
        self.names = None
        self.buffer = None
        self.storage = None
        self.flush_interval_seconds=float(flush_interval_seconds)
        self._flush_lock=threading.RLock()
        self._flush_stop=threading.Event()
        self._flush_thread=None
        self._flush_error=None
        self._last_flush=time.monotonic()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            if self.count and self.storage is None:
                self.finish_data()  # Preserve an explicitly counted prefix on ordinary errors.
        finally:
            self._stop_flusher()
            with self._flush_lock:self.handles.close()

    def append(self, sample, signals):
        with self._flush_lock:
            self._raise_flush_error()
            self._append_locked(sample,signals)
            if self._flush_thread is None:
                self._flush_thread=threading.Thread(target=self._flush_loop,name="slim-trace-flush",daemon=True)
                self._flush_thread.start()

    def _append_locked(self, sample, signals):
        if self.storage is not None or self.count >= self.capacity:
            raise TraceStorageError("Recording is finished or exceeds the solver-step capacity")
        names = tuple(signals)
        if self.names is None:
            if not names:
                raise TraceStorageError("No actual signal channels supplied")
            self.names = names
            self.buffer = np.empty((self.chunk_rows, len(names)), dtype=_FLOAT)
            self.signals_file.truncate(self.capacity * len(names) * 8)
        if names == self.names:
            values = tuple(signals.values())
        elif set(names) == set(self.names):
            values = tuple(signals[name] for name in self.names)
        else:
            raise TraceStorageError("An actual signal channel is missing or changed during recording")
        row = np.asarray(values, dtype=_FLOAT)
        if not np.isfinite(row).all():
            raise TraceStorageError("Nonfinite actual signal value")
        if "time_s" in signals and sample["time_s"] != signals["time_s"]:
            raise TraceStorageError("Sample and signal times differ")
        encoded = (json.dumps(sample, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        self.offsets_file.write(struct.pack("<Q", self.samples_file.tell()))
        self.samples_file.write(encoded)
        self.buffer[self.pending] = row
        self.pending += 1
        self.count += 1
        # Publish t=0 before returning to a potentially slow native step.
        # Keep the original row boundaries even after this first short flush
        # or an interval flush; the signal buffer never exceeds chunk_rows.
        if self.count==1 or self.count % self.chunk_rows==0 or time.monotonic()-self._last_flush>=self.flush_interval_seconds:
            self._flush()

    def _flush(self):
        with self._flush_lock:
            self._raise_flush_error()
            self._flush_locked()

    def _raise_flush_error(self):
        if self._flush_error is not None:
            raise TraceStorageError(f"Timed trace flush failed: {self._flush_error}") from self._flush_error

    def _flush_loop(self):
        try:
            while True:
                with self._flush_lock:
                    if self._flush_stop.is_set() or self.storage is not None or self.samples_file.closed:return
                    delay=(max(0.,self.flush_interval_seconds-(time.monotonic()-self._last_flush))
                           if self.pending else self.flush_interval_seconds)
                if self._flush_stop.wait(delay):return
                with self._flush_lock:
                    if self._flush_stop.is_set() or self.storage is not None or self.samples_file.closed:return
                    if self.pending and time.monotonic()-self._last_flush>=self.flush_interval_seconds:self._flush_locked()
        except Exception as exc:
            with self._flush_lock:self._flush_error=exc
            self._flush_stop.set()

    def _stop_flusher(self):
        self._flush_stop.set()
        thread=self._flush_thread
        if thread is not None and thread is not threading.current_thread():thread.join()

    def _flush_locked(self):
        if not self.pending:
            return
        for index in range(len(self.names)):
            self.signals_file.seek((index * self.capacity + self.flushed) * 8)
            self.signals_file.write(self.buffer[:self.pending, index].tobytes())
        # Do not retain dirty mappings of the whole multi-gigabyte array.
        # Flush one bounded recording block, making file-cache pages reclaimable.
        for stream in (self.samples_file, self.offsets_file, self.signals_file):
            stream.flush()
            os.fsync(stream.fileno())
            _drop_clean_file_cache(stream)
        self.flushed += self.pending
        self.pending = 0
        # Publish only after all three streams have reached this boundary.
        # No final offset sentinel exists yet; sample_byte_end supplies it.
        write_json_atomic(trace_prefix_path(self.trace_path), {
            "schema_version":PREFIX_SCHEMA, "trace_path":str(self.trace_path),
            "storage":{"schema_version":PREFIX_SCHEMA, "dtype":"<f8", "layout":"signal_columns",
                "expected_samples":self.capacity, "recorded_samples":self.flushed,
                "column_stride_samples":self.capacity, "offset_count":self.flushed,
                "sample_byte_end":self.samples_file.tell(), "signal_names":list(self.names),
                "files":{name:{"path":Path(os.path.relpath(path,self.trace_path.parent)).as_posix()}
                         for name,path in self.paths.items()}}})
        self._last_flush=time.monotonic()

    def finish_data(self):
        # Join outside the lock needed by the background writer. Packing and
        # final hashes then see a single, stopped producer of all three files.
        self._stop_flusher()
        with self._flush_lock:
            self._raise_flush_error()
            return self._finish_data_locked()

    def _finish_data_locked(self):
        if self.storage is not None:
            return self.storage
        if not self.count:
            raise TraceStorageError("No solver samples were recorded")
        self._flush()
        self.offsets_file.write(struct.pack("<Q", self.samples_file.tell()))
        if self.count != self.capacity:
            # Preserve the published capacity-stride prefix even if killed
            # during final packing. Only shortened runs need this compact copy.
            compact_path = self.directory / "signals.compact.f64"
            with compact_path.open("xb") as compact:
                for column in range(len(self.names)):
                    copied = 0
                    length = self.count * 8
                    while copied < length:
                        size = min(1024 * 1024, length - copied)
                        self.signals_file.seek(column * self.capacity * 8 + copied)
                        block = self.signals_file.read(size)
                        if len(block) != size:
                            raise TraceStorageError("Actual signal block is incomplete")
                        compact.write(block)
                        copied += size
                compact.flush()
                os.fsync(compact.fileno())
            self.paths["signals"] = compact_path
        for stream in (self.samples_file, self.offsets_file, self.signals_file):
            stream.flush()
            os.fsync(stream.fileno())
        self.handles.close()
        self.buffer = None
        self.storage = {"schema_version": STORAGE_SCHEMA, "dtype": "<f8", "layout": "signal_columns",
                        "expected_samples": self.capacity, "recorded_samples": self.count,
                        "signal_names": list(self.names),
                        "files": {name: {"path": Path(os.path.relpath(path, self.trace_path.parent)).as_posix(),
                                         "bytes": path.stat().st_size, "sha256": file_digest(path, drop_cache=True)}
                                  for name, path in self.paths.items()}}
        # This data-only description is recoverable evidence, never a PASS.
        write_json_atomic(self.directory / "data.json", {"schema_version": "slim_recorded_data_v1",
                          "trace_path": str(self.trace_path), "storage": self.storage})
        return self.storage

    def view(self, metadata):
        if "samples" in metadata or "signals" in metadata:
            raise TraceStorageError("Inline recording arrays must not enter the trace manifest")
        value = {**metadata, "schema_version": TRACE_SCHEMA, "storage": self.finish_data()}
        return TraceView(value, self.trace_path)

    def commit(self, metadata):
        view = self.view(metadata)
        write_json_atomic(self.trace_path, view.metadata)
        return view.metadata
