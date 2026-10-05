# coding=utf-8
"""managed_local refuses an SGLang whose spec-capture patch predates the
multimodal split-storage fix (training would run ~20x slower without it)."""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from unittest import mock

from specforge import launch_plan


def _fake_sglang(root: str, with_fix: bool) -> str:
    package = os.path.join(root, "sglang")
    managers = os.path.join(package, "srt", "managers")
    os.makedirs(managers)
    body = "def get_new_expanded_mm_items(items):\n    return items\n"
    if with_fix:
        body = "def _own_split_storage(items):\n    return items\n\n\n" + body
    with open(os.path.join(managers, "mm_utils.py"), "w", encoding="utf-8") as handle:
        handle.write(body)
    return package


class SplitStorageFixDetectionTest(unittest.TestCase):
    def _detect(self, spec):
        with mock.patch("specforge.launch_plan.importlib.util.find_spec", return_value=spec):
            return launch_plan._sglang_has_split_storage_fix()

    def test_detects_the_fix(self):
        with tempfile.TemporaryDirectory() as root:
            spec = types.SimpleNamespace(submodule_search_locations=[_fake_sglang(root, True)])
            self.assertIs(True, self._detect(spec))

    def test_detects_an_old_patch(self):
        with tempfile.TemporaryDirectory() as root:
            spec = types.SimpleNamespace(submodule_search_locations=[_fake_sglang(root, False)])
            self.assertIs(False, self._detect(spec))

    def test_unknown_layout_does_not_block(self):
        self.assertIsNone(self._detect(object()))  # what the other preflight tests mock
        self.assertIsNone(self._detect(None))
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(self._detect(types.SimpleNamespace(submodule_search_locations=[root])))

    def test_preflight_rejects_an_old_patch(self):
        plan = mock.Mock(managed_root=os.path.join(tempfile.gettempdir(), "specforge-preflight-absent"), managed_ports=())
        with (
            mock.patch("specforge.launch_plan.shutil.which", return_value="/usr/bin/mooncake_master"),
            mock.patch("specforge.launch_plan.importlib.util.find_spec", return_value=object()),
            mock.patch("specforge.launch_plan._sglang_has_split_storage_fix", return_value=False),
            self.assertRaisesRegex(RuntimeError, "split-storage fix"),
        ):
            launch_plan._managed_preflight(plan)


if __name__ == "__main__":
    unittest.main()
