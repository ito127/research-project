"""CPU unit tests of the text-removal helpers, the attention-mask edits and the skip check of
scripts/remove_text.py (no model weights needed).

    python -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "diffusers/src")]

from textremoval.fill import extend_boundary_colors, fill_from_boundary  # noqa: E402
from textremoval.guidance import TextRemovalController, replace_with_source  # noqa: E402
from textremoval.masks import dilate_mask, pixel_mask_to_latent, select_text_mask, tight_box  # noqa: E402


class MaskTests(unittest.TestCase):
    def test_largest_component_and_box(self):
        mask = np.zeros((16, 16), bool)
        mask[1:3, 1:3] = True          # 4 px
        mask[8:12, 5:10] = True        # 20 px
        largest = select_text_mask(mask, "largest_text")
        self.assertEqual(int(largest.sum()), 20)
        self.assertEqual(tight_box(largest), [5, 8, 10, 12])
        self.assertTrue(np.array_equal(select_text_mask(mask, "all_text"), mask))
        with self.assertRaises(ValueError):
            select_text_mask(np.zeros((4, 4), bool), "all_text")

    def test_latent_cells_and_dilation(self):
        mask = np.zeros((32, 32), bool)
        mask[9, 17] = True
        cells = pixel_mask_to_latent(mask)
        self.assertEqual(tuple(cells.shape), (1, 1, 4, 4))
        self.assertTrue(bool(cells[0, 0, 1, 2]) and int(cells.sum()) == 1)
        self.assertEqual(int(dilate_mask(cells, 1).sum()), 9)


class FillTests(unittest.TestCase):
    def test_fill_keeps_outside_and_uses_boundary_colour(self):
        array = np.full((40, 40, 3), (10, 200, 30), np.uint8)
        array[15:25, 15:25] = (255, 255, 255)        # "text"
        image = Image.fromarray(array)
        mask = np.zeros((40, 40), bool)
        mask[15:25, 15:25] = True
        filled = np.asarray(fill_from_boundary(image, mask, dilation_px=2, sigma=12.0, gradual_distance_px=60.0))
        self.assertTrue((filled[15:25, 15:25] == (10, 200, 30)).all())
        self.assertTrue(np.array_equal(filled[:10], array[:10]))
        # Image A: the text is kept, the outside takes the text's boundary colour.
        image_a = np.asarray(extend_boundary_colors(image, mask, 12.0, gradual_distance_px=60.0))
        self.assertTrue(np.array_equal(image_a[15:25, 15:25], array[15:25, 15:25]))


class GuidanceTests(unittest.TestCase):
    def test_replace_makes_clean_estimate_equal_source(self):
        generator = torch.Generator().manual_seed(0)
        velocity = torch.randn(1, 3, 16, 6, 6, generator=generator)
        latents = torch.randn(1, 3, 16, 6, 6, generator=generator)
        source = torch.randn(1, 16, 6, 6, generator=generator)
        keep = torch.ones(1, 1, 6, 6)
        keep[..., :2, :2] = 0
        sigma = torch.tensor(0.7)
        corrected, fraction = replace_with_source(velocity, latents, sigma, source, keep, 0.0)
        clean = latents[:, 1] - 0.7 * corrected[:, 1]
        inside = keep.bool().expand_as(clean)
        self.assertTrue(torch.allclose(clean[inside], source[inside], atol=1e-5))
        self.assertTrue(torch.equal(corrected[:, 1][~inside], velocity[:, 1][~inside]))
        self.assertTrue(torch.equal(corrected[:, [0, 2]], velocity[:, [0, 2]]))
        self.assertEqual(fraction, 1.0)

    def test_controller_blocks_previous_mask_on_layer_zero(self):
        text = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
        text[..., 2:4, 2:4] = True
        box = torch.zeros_like(text)
        box[..., 0:6, 0:6] = True
        controller = TextRemovalController(text_latent=text, box=box, z_grey=torch.zeros(1, 16, 8, 8),
                                           source=torch.zeros(1, 16, 8, 8), num_layers=3)
        maps = controller.token_maps(0)
        self.assertEqual(tuple(maps.shape), (1, 3, 4, 4))
        self.assertTrue(bool(maps[0, 0, 1, 1]) and int(maps.sum()) == 1)
        # Step 1: M_1 from the text layer's clean estimate (non-grey inside the box only).
        latents = torch.zeros(1, 3, 16, 8, 8)
        latents[:, 2, :, 4:8, 4:8] = 5.0
        controller.correct(0, torch.tensor(0.9), latents, torch.zeros_like(latents))
        controller.correct(1, torch.tensor(0.8), latents, torch.zeros_like(latents))
        self.assertTrue(torch.equal(controller.keep_masks[0], text[0, 0]))
        expected = torch.zeros(8, 8, dtype=torch.bool)
        expected[4:6, 4:6] = True
        self.assertTrue(torch.equal(controller.keep_masks[1], expected))
        self.assertTrue(torch.equal(controller.token_maps(2)[0, 0], torch.tensor(
            [[False] * 4, [False] * 4, [False, False, True, False], [False] * 4])))


class AttentionMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.custom_model_mmdit import CustomFluxTransformer2DModel
        cls.model = CustomFluxTransformer2DModel

    def test_reference_layer_and_blocking(self):
        model = self.model
        boxes = [(0, 0, 64, 64), (0, 0, 64, 64), (16, 16, 48, 48)]    # 4x4, 4x4, 2x2 tokens
        t_text = 3
        base = model.build_additive_spatial_roi_mask_v2(model, boxes, height=8, width=8, t_text_length=t_text,
                                                        device="cpu", dtype=torch.float32)
        mask = model.extend_mask_with_foreground_reference(base, boxes, t_text, boxes[0])
        layer0, background, text_layer = slice(3, 19), slice(19, 35), slice(35, 39)
        reference = slice(39, 55)
        self.assertEqual(tuple(mask.shape), (55, 55))
        # The text layer no longer sees layer 0 but sees the reference tokens inside its box only.
        self.assertTrue((mask[text_layer, layer0] < -1e8).all())
        visible = (mask[text_layer, reference] == 0).float().sum(dim=1)
        self.assertTrue(torch.equal(visible, torch.full((4,), 4.0)))
        # Nobody but the text layer and the reference itself reads the reference tokens.
        self.assertTrue((mask[:35, reference] < -1e8).all())
        # Blocking: background queries lose the marked layer-0 tokens; other rows are unchanged.
        maps = torch.zeros(1, 3, 4, 4, dtype=torch.bool)
        maps[0, 0, 1, 2] = True
        blocked = model.block_tokens_from_background(model, mask, maps, boxes, t_text)
        column = 3 + 1 * 4 + 2
        self.assertTrue((blocked[background, column] < -1e30).all())
        changed = (blocked != mask).nonzero()
        self.assertTrue(((changed[:, 0] >= 19) & (changed[:, 0] < 35) & (changed[:, 1] == column)).all())


class SkipCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import remove_text
        cls.cli = remove_text

    def test_missing_outputs(self):
        import tempfile

        cli = self.cli
        names = cli.expected_outputs(paste_back=True, save_intermediates=False)
        self.assertEqual(names, [*cli.RESULT_FILES, "result.png"])
        self.assertEqual(len(cli.expected_outputs(paste_back=False, save_intermediates=True)),
                         len(cli.RESULT_FILES) + len(cli.INTERMEDIATE_FILES))
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            self.assertEqual(cli.missing_outputs(out_dir / "absent", names), names)
            for name in cli.RESULT_FILES:
                (out_dir / name).write_bytes(b"x")
            self.assertEqual(cli.missing_outputs(out_dir, names), ["result.png"])
            (out_dir / "result.png").touch()          # empty: an interrupted write
            self.assertEqual(cli.missing_outputs(out_dir, names), ["result.png"])
            (out_dir / "result.png").write_bytes(b"x")
            self.assertEqual(cli.missing_outputs(out_dir, names), [])


if __name__ == "__main__":
    unittest.main()
