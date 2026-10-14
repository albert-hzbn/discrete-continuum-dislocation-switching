"""Portable stand-in for the POSIX ``resource`` module used by the benchmark drivers.

Only ``getrusage(RUSAGE_SELF|RUSAGE_CHILDREN).ru_maxrss`` is used by the benchmarks.
On POSIX the real module is used. On Windows the peak working set of the current
process is returned in KiB, the Linux ``ru_maxrss`` unit, so callers that scale by
1024 on non-Darwin systems report bytes. Child-process peaks are not available on
Windows and are reported as 0.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

try:
    from resource import RUSAGE_CHILDREN, RUSAGE_SELF, getrusage  # noqa: F401
except ImportError:
    RUSAGE_SELF = 0
    RUSAGE_CHILDREN = -1

    def _peak_working_set_bytes() -> int:
        if sys.platform != "win32":
            return 0
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else 0

    def getrusage(who: int) -> SimpleNamespace:
        peak = _peak_working_set_bytes() if who == RUSAGE_SELF else 0
        return SimpleNamespace(ru_maxrss=peak // 1024)
