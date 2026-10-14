"""Advisory whole-file locks on an open descriptor, on POSIX and Windows.

The Loop's coordination files (event log, run locks, launch locks) were
guarded with ``fcntl.flock`` directly, which does not exist on Windows and made
every module that imported it unusable there.  ``lock_file`` / ``unlock_file``
keep the same contract: an exclusive lock on the whole file that is released
explicitly or when the descriptor is closed, with a non-blocking variant that
raises ``BlockingIOError`` when the lock is held elsewhere.

POSIX uses ``flock``.  Windows uses ``msvcrt.locking`` on the first byte of
the file (byte ranges may extend past EOF, so an empty lock file works).  The
Windows call is not truly blocking - it retries for about ten seconds - so
``lock_file`` retries it until the lock is acquired.
"""

from __future__ import annotations

import os
import time

if os.name == "nt":  # pragma: no cover - exercised on Windows only
    import msvcrt

    _LOCK_BYTES = 1
    _RETRY_SLEEP_SECONDS = 0.05

    def lock_file(descriptor: int, *, blocking: bool = True) -> None:
        while True:
            position = os.lseek(descriptor, 0, os.SEEK_CUR)
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, _LOCK_BYTES)
                return
            except OSError as error:
                if not blocking:
                    raise BlockingIOError(
                        error.errno, "file is locked by another process"
                    ) from error
                time.sleep(_RETRY_SLEEP_SECONDS)
            finally:
                os.lseek(descriptor, position, os.SEEK_SET)

    def unlock_file(descriptor: int) -> None:
        position = os.lseek(descriptor, 0, os.SEEK_CUR)
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, _LOCK_BYTES)
        except OSError:
            pass  # closing the descriptor releases the range anyway
        finally:
            os.lseek(descriptor, position, os.SEEK_SET)

else:
    import fcntl

    def lock_file(descriptor: int, *, blocking: bool = True) -> None:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(descriptor, flags)

    def unlock_file(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
