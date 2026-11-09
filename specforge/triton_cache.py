"""A Triton cache manager that gives every process its own cache directory.

Two processes that JIT-compile the same kernel at the same moment both write it
into the shared ``TRITON_CACHE_DIR`` and read it back. On the cluster's NFS
scratch one of them can then hit ``FileNotFoundError`` on a ``.ptx``/``.json``
the other is replacing, and its SGLang scheduler exits; that killed one server
in each of two regen runs, once two hours in. The tensor-parallel ranks of one
capture server inherit one environment, so a per-server directory cannot keep
them apart; this manager does::

    TRITON_CACHE_MANAGER=specforge.triton_cache:PerProcessFileCacheManager

puts each process's entries under ``$TRITON_CACHE_DIR/pid<pid>/``. Every
process compiles its own kernels, which ranks starting on a cold per-job cache
do anyway. The managed capture servers use it whenever more than one SGLang
process shares the job's cache (``launch_plan``).
"""

from __future__ import annotations

import os

from triton import knobs
from triton.runtime.cache import FileCacheManager

#: ``TRITON_CACHE_MANAGER`` value selecting :class:`PerProcessFileCacheManager`.
PER_PROCESS_CACHE_MANAGER = "specforge.triton_cache:PerProcessFileCacheManager"


class PerProcessFileCacheManager(FileCacheManager):
    """:class:`FileCacheManager` rooted at ``<cache dir>/pid<pid>``."""

    def __init__(self, key, override=False, dump=False):
        if override or dump:
            # the dump and override directories are read, not raced on
            super().__init__(key, override=override, dump=dump)
            return
        root = knobs.cache.dir
        if not root:
            raise RuntimeError("Could not create or locate cache dir")
        self.key = key
        self.cache_dir = os.path.join(root, f"pid{os.getpid()}", key)
        self.lock_path = os.path.join(self.cache_dir, "lock")
        os.makedirs(self.cache_dir, exist_ok=True)
