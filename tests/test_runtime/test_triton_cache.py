# coding=utf-8
"""The per-process Triton cache manager of the managed capture servers."""

import os
import tempfile
import unittest
from unittest import mock

from triton import knobs
from triton.runtime.cache import get_cache_manager

from specforge.triton_cache import PER_PROCESS_CACHE_MANAGER, PerProcessFileCacheManager


class TestPerProcessCache(unittest.TestCase):
    def test_entries_land_under_the_process_own_directory(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            os.environ,
            {"TRITON_CACHE_DIR": root, "TRITON_CACHE_MANAGER": PER_PROCESS_CACHE_MANAGER},
        ):
            self.assertIs(knobs.cache.manager_class, PerProcessFileCacheManager)
            manager = get_cache_manager("ab" * 32)
            self.assertIsInstance(manager, PerProcessFileCacheManager)
            own = os.path.join(root, f"pid{os.getpid()}")
            self.assertEqual(os.path.dirname(manager.cache_dir), own)
            path = manager.put(b"\x00ptx", "kernel.ptx")
            self.assertTrue(path.startswith(own + os.sep))
            self.assertEqual(manager.get_file("kernel.ptx"), path)
            manager.put_group("kernel.json", {"kernel.ptx": path})
            self.assertEqual(manager.get_group("kernel.json"), {"kernel.ptx": path})
            # nothing is written into the shared root itself
            self.assertEqual(os.listdir(root), [f"pid{os.getpid()}"])
            # a second manager in this process finds the same entry
            self.assertEqual(get_cache_manager("ab" * 32).get_file("kernel.ptx"), path)


if __name__ == "__main__":
    unittest.main()
