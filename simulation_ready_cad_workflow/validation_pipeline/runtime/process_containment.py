from __future__ import annotations

import os
import signal
import subprocess
from typing import Any


WINDOWS_CREATE_SUSPENDED = 0x00000004
WINDOWS_JOB_LIMIT_JOB_MEMORY = 0x00000200
WINDOWS_JOB_LIMIT_KILL_ON_CLOSE = 0x00002000


class ProcessContainment:
    """Bind a Windows process tree before it can launch descendants."""

    def __init__(self) -> None:
        self._handle: object | None = None
        self._kernel32: object | None = None
        self._ntdll: object | None = None
        self._process_handle: object | None = None
        self._process_pid: int | None = None
        self._resumed = False

    def __enter__(self) -> ProcessContainment:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self.close()

    def attach_and_resume(
        self,
        process: subprocess.Popen[Any],
        *,
        max_job_memory_bytes: int | None,
    ) -> None:
        self._validate_memory_limit(max_job_memory_bytes)
        if os.name != "nt":
            return
        self.attach_suspended(
            process,
            max_job_memory_bytes=max_job_memory_bytes,
        )
        self.resume_suspended(process)

    @staticmethod
    def _validate_memory_limit(max_job_memory_bytes: int | None) -> None:
        if (
            max_job_memory_bytes is not None
            and (
                not isinstance(max_job_memory_bytes, int)
                or isinstance(max_job_memory_bytes, bool)
                or max_job_memory_bytes <= 0
            )
        ):
            raise ValueError(
                "max_job_memory_bytes must be a positive integer or None"
            )

    def attach_suspended(
        self,
        process: subprocess.Popen[Any],
        *,
        max_job_memory_bytes: int | None,
    ) -> None:
        """Attach a child without allowing a gated process to execute."""

        self._validate_memory_limit(max_job_memory_bytes)
        process_pid = int(process.pid)
        if process_pid <= 0:
            raise OSError("process containment received an invalid PID")
        if self._process_pid is not None:
            raise OSError("process containment is already attached")
        self._process_pid = process_pid
        self._resumed = False
        if os.name != "nt":
            return

        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [
            ctypes.c_void_p,
            wintypes.LPCWSTR,
        ]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        ntdll.NtResumeProcess.restype = wintypes.LONG

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._handle = handle
        self._kernel32 = kernel32
        try:
            limits = ExtendedLimitInformation()
            limits.BasicLimitInformation.LimitFlags = (
                WINDOWS_JOB_LIMIT_KILL_ON_CLOSE
            )
            if max_job_memory_bytes is not None:
                limits.BasicLimitInformation.LimitFlags |= (
                    WINDOWS_JOB_LIMIT_JOB_MEMORY
                )
                limits.JobMemoryLimit = max_job_memory_bytes
            if not kernel32.SetInformationJobObject(
                handle,
                9,
                ctypes.byref(limits),
                ctypes.sizeof(limits),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            raw_process_handle = getattr(process, "_handle", None)
            if raw_process_handle is None:
                raise OSError("Windows child process handle is unavailable")
            process_handle = wintypes.HANDLE(int(raw_process_handle))
            if not kernel32.AssignProcessToJobObject(
                handle,
                process_handle,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            self._ntdll = ntdll
            self._process_handle = process_handle
        except BaseException:
            self.close()
            raise

    def resume_suspended(
        self,
        process: subprocess.Popen[Any],
    ) -> None:
        """Release exactly the child previously attached while suspended."""

        process_pid = int(process.pid)
        if (
            self._process_pid != process_pid
            or self._resumed
        ):
            raise OSError(
                "process containment has no matching suspended child"
            )
        if os.name != "nt":
            os.kill(process_pid, signal.SIGCONT)
            self._resumed = True
            return
        ntdll = self._ntdll
        process_handle = self._process_handle
        if ntdll is None or process_handle is None:
            raise OSError("Windows suspended process handle is unavailable")
        resume_status = int(ntdll.NtResumeProcess(process_handle))
        if resume_status != 0:
            raise OSError(
                "NtResumeProcess failed with NTSTATUS "
                f"0x{resume_status & 0xFFFFFFFF:08x}"
            )
        self._resumed = True

    def close(self) -> None:
        if self._handle is None:
            self._process_pid = None
            self._process_handle = None
            self._ntdll = None
            self._resumed = False
            return
        import ctypes

        kernel32 = self._kernel32
        if kernel32 is None or not kernel32.CloseHandle(self._handle):
            raise ctypes.WinError(ctypes.get_last_error())
        self._handle = None
        self._kernel32 = None
        self._ntdll = None
        self._process_handle = None
        self._process_pid = None
        self._resumed = False
