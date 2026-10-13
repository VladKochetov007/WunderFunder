from __future__ import annotations

import unittest

from scripts.team_member_export_lib import infer_architecture, iter_chunks, select_row_groups


def gru_shapes(hidden: int, layers: int, head_width: int) -> dict[str, tuple[int, ...]]:
    shapes = {
        "mean": (211,),
        "std": (211,),
        "inp.weight": (hidden, 211),
        "inp.bias": (hidden,),
        "head.0.weight": (head_width, 2 * hidden),
        "head.0.bias": (head_width,),
        "head.2.weight": (2, head_width),
        "head.2.bias": (2,),
    }
    for layer in range(layers):
        shapes.update(
            {
                f"gru.weight_ih_l{layer}": (3 * hidden, hidden),
                f"gru.weight_hh_l{layer}": (3 * hidden, hidden),
                f"gru.bias_ih_l{layer}": (3 * hidden,),
                f"gru.bias_hh_l{layer}": (3 * hidden,),
            }
        )
    return shapes


class MemberArchitectureTest(unittest.TestCase):
    def test_canonical_and_merged_heads_are_inferred(self) -> None:
        canonical = infer_architecture(gru_shapes(128, 2, 128))
        self.assertEqual((canonical.hidden, canonical.layers, canonical.head_width), (128, 2, 128))

    def test_head_width_is_preserved_for_merged_team_member(self) -> None:
        architecture = infer_architecture(gru_shapes(256, 2, 512))
        self.assertEqual((architecture.hidden, architecture.head_width), (256, 512))

    def test_noncontiguous_or_wrong_width_gru_is_rejected(self) -> None:
        shapes = gru_shapes(128, 2, 128)
        for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            shapes[f"gru.{kind}_l2"] = shapes.pop(f"gru.{kind}_l1")
        with self.assertRaisesRegex(ValueError, "contiguous"):
            infer_architecture(shapes)

        shapes = gru_shapes(128, 1, 128)
        shapes["gru.weight_hh_l0"] = (384, 127)
        with self.assertRaisesRegex(ValueError, "weight_hh_l0"):
            infer_architecture(shapes)

    def test_unsupported_recurrent_tensor_is_rejected(self) -> None:
        shapes = gru_shapes(128, 1, 128)
        shapes["gru.weight_ih_l0_reverse"] = (384, 128)
        with self.assertRaisesRegex(ValueError, "direction"):
            infer_architecture(shapes)


class RowGroupSelectionTest(unittest.TestCase):
    def test_default_and_bounded_group_ranges(self) -> None:
        self.assertEqual(select_row_groups(0, 2, 1873), (0, 1))
        self.assertEqual(select_row_groups(1871, None, 1873), (1871, 1872))

    def test_empty_and_out_of_range_selections_are_rejected(self) -> None:
        for first, count in ((1873, 1), (0, 0), (1872, 2), (-1, 1)):
            with self.subTest(first=first, count=count), self.assertRaises(ValueError):
                select_row_groups(first, count, 1873)


class RecurrentChunkTest(unittest.TestCase):
    def test_full_sequence_uses_ordered_stateful_chunks(self) -> None:
        chunks = iter_chunks(20_000, 500)
        self.assertEqual(len(chunks), 40)
        self.assertEqual(chunks[:2], ((0, 500), (500, 1000)))
        self.assertEqual(chunks[-1], (19_500, 20_000))
        self.assertTrue(all(start < stop for start, stop in chunks))
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], 20_000)
        self.assertTrue(
            all(left[1] == right[0] for left, right in zip(chunks, chunks[1:], strict=False))
        )

    def test_short_final_chunk_and_invalid_bounds(self) -> None:
        self.assertEqual(iter_chunks(11, 4), ((0, 4), (4, 8), (8, 11)))
        for length, width in ((0, 1), (5, 0), (5, 6)):
            with self.subTest(length=length, width=width), self.assertRaises(ValueError):
                iter_chunks(length, width)


if __name__ == "__main__":
    unittest.main()
