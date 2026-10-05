# coding=utf-8
"""The spec-capture patch makes every split multimodal feature own its storage.

SGLang splits a bundled pixel_values tensor into one item per image/frame. As
views, N items pickle to N x the whole tensor (O(N^2) bytes) on the in-band
transport the --skip-tokenizer-init capture server uses: a 48-frame video was
a 49.8 GB message and ~98 s per capture request. The patched
get_new_expanded_mm_items copies each slice, so the message is ~1x the data.
"""

from __future__ import annotations

import pickle
import unittest


def _patched_mm_utils():
    try:
        from sglang.srt.managers import mm_utils
    except Exception:  # sglang missing or not importable here
        return None
    return mm_utils if hasattr(mm_utils, "_own_split_storage") else None


class SplitFeaturesOwnStorageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mm_utils = _patched_mm_utils()
        if cls.mm_utils is None:
            raise unittest.SkipTest(
                "installed sglang lacks the spec-capture split-storage fix; run "
                "scripts/apply_sglang_spec_capture_patch.sh"
            )

    def _bundled_item(self, frames: int, patches: int):
        import torch
        from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

        pixel_values = torch.randn(frames * patches, 64)
        item = MultimodalDataItem(modality=Modality.IMAGE)
        item.set("feature", pixel_values)
        item.set("image_grid_thw", torch.tensor([[1, 2, patches // 2]] * frames))
        tokens = patches // 4
        item.offsets = [(10 + i * (tokens + 2), 10 + i * (tokens + 2) + tokens - 1) for i in range(frames)]
        return item, pixel_values

    def test_split_items_own_their_storage(self):
        import torch

        frames, patches = 12, 40
        item, pixel_values = self._bundled_item(frames, patches)
        items = self.mm_utils.get_new_expanded_mm_items([item])
        self.assertEqual(frames, len(items))
        storages = {it.feature.untyped_storage().data_ptr() for it in items}
        self.assertEqual(frames, len(storages))
        for index, it in enumerate(items):
            self.assertEqual(it.feature.untyped_storage().nbytes(), it.feature.numel() * it.feature.element_size())
            self.assertTrue(torch.equal(it.feature, pixel_values[index * patches:(index + 1) * patches]))

    def test_pickle_is_linear_in_frames(self):
        frames, patches = 12, 40
        item, pixel_values = self._bundled_item(frames, patches)
        items = self.mm_utils.get_new_expanded_mm_items([item])
        data = pixel_values.numel() * pixel_values.element_size()
        size = len(pickle.dumps([it.feature for it in items], protocol=4))
        # views would pickle to frames x data; owned slices to ~1x (+ headers)
        self.assertLess(size, 1.2 * data)

    def test_model_specific_data_slices_are_owned_too(self):
        import torch

        frames, patches = 6, 40
        item, _ = self._bundled_item(frames, patches)
        grid = item.model_specific_data["image_grid_thw"]
        items = self.mm_utils.get_new_expanded_mm_items([item])
        for index, it in enumerate(items):
            value = it.model_specific_data["image_grid_thw"]
            self.assertEqual(value.untyped_storage().nbytes(), value.numel() * value.element_size())
            self.assertTrue(torch.equal(value.reshape(-1), grid[index].reshape(-1)))

    def test_shm_transport_servers_keep_views(self):
        """On the /dev/shm path every view is copied into shm anyway: no clone."""
        from unittest import mock

        args = type("Args", (), {"skip_tokenizer_init": False})()
        frames, patches = 6, 40
        item, pixel_values = self._bundled_item(frames, patches)
        with (
            mock.patch.object(self.mm_utils, "get_global_server_args", return_value=args),
            mock.patch.object(self.mm_utils, "_get_is_default_transport", return_value=False),
        ):
            items = self.mm_utils.get_new_expanded_mm_items([item])
        base = pixel_values.untyped_storage().data_ptr()
        self.assertTrue(all(it.feature.untyped_storage().data_ptr() == base for it in items))

    def test_capture_server_clones(self):
        from unittest import mock

        args = type("Args", (), {"skip_tokenizer_init": True})()
        item, pixel_values = self._bundled_item(6, 40)
        with (
            mock.patch.object(self.mm_utils, "get_global_server_args", return_value=args),
            mock.patch.object(self.mm_utils, "_get_is_default_transport", return_value=False),
        ):
            items = self.mm_utils.get_new_expanded_mm_items([item])
        self.assertEqual(6, len({it.feature.untyped_storage().data_ptr() for it in items}))

    def test_whole_tensor_items_are_left_alone(self):
        import torch

        tensor = torch.randn(8, 4)
        holder = type("Item", (), {})()
        holder.feature = tensor
        holder.precomputed_embeddings = None
        self.mm_utils._own_split_storage([holder])
        self.assertIs(tensor, holder.feature)  # owns its storage already: no copy


if __name__ == "__main__":
    unittest.main()
