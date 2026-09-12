# coding=utf-8
"""MMFlash's collator: the DFlash padding contract plus ``visual_score``."""

import unittest

import torch

from specforge.algorithms.common.dflash_family_data import (
    build_collator,
    build_mmflash_collator,
)
from specforge.data.visual_score import SCALE, TEXT_ONLY_SENTINEL


def _sample(length, visual=None):
    feature = {
        "input_ids": torch.arange(1, length + 1).view(1, -1),
        "loss_mask": torch.ones(1, length, dtype=torch.long),
        "hidden_states": torch.ones(1, length, 4),
    }
    if visual is not None:
        feature["visual_score"] = torch.tensor([visual], dtype=torch.long)
    return feature


class TestMMFlashCollator(unittest.TestCase):
    def test_without_channel_equals_dflash_collator(self):
        features = [_sample(3), _sample(5)]
        mm = build_mmflash_collator()(features)
        base = build_collator()(features)
        self.assertEqual(set(mm), set(base))
        for key in base:
            torch.testing.assert_close(mm[key], base[key])
        self.assertNotIn("visual_score", mm)

    def test_channel_is_padded_with_the_sentinel_and_dequantised(self):
        image = _sample(3, visual=[0, SCALE // 2, SCALE])
        text = _sample(5, visual=[TEXT_ONLY_SENTINEL] * 5)
        short_text = _sample(2, visual=[TEXT_ONLY_SENTINEL] * 2)
        batch = build_mmflash_collator()([image, text, short_text])
        score = batch["visual_score"]
        self.assertEqual(score.dtype, torch.float32)
        self.assertEqual(tuple(score.shape), (3, 5))
        torch.testing.assert_close(score[0], torch.tensor([0.0, 0.5, 1.0, -1.0, -1.0]))
        # the has-image flag the model derives: only the first row
        self.assertEqual((score >= 0).any(dim=1).tolist(), [True, False, False])
        # loss_mask padding is still zero
        self.assertEqual(batch["loss_mask"][0].tolist(), [1, 1, 1, 0, 0])

    def test_mixed_presence_is_an_error(self):
        with self.assertRaises(KeyError):
            build_mmflash_collator()([_sample(3, visual=[0, 0, 0]), _sample(3)])

    def test_bad_shape_is_an_error(self):
        bad = _sample(3)
        bad["visual_score"] = torch.zeros(3, dtype=torch.long)
        with self.assertRaises(ValueError):
            build_mmflash_collator()([bad])


if __name__ == "__main__":
    unittest.main()
