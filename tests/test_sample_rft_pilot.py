import unittest

from rft.prompt import NATURAL_COT_PROMPT_VERSION
from scripts.sample_rft_pilot import select_pilot_rows


def make_row(order: int, pair_index: int, side: str) -> dict:
    pair = f"order-{order}-pair-{pair_index}"
    return {
        "global_sample_id": f"{pair}-{side}",
        "global_pair_id": pair,
        "question_order": order,
        "intervention_type": side,
        "process_prompt_version": NATURAL_COT_PROMPT_VERSION,
        "process_target": {"tom_order": order},
    }


class SampleRFTPilotTest(unittest.TestCase):
    @staticmethod
    def rows(pair_count: int = 4) -> list[dict]:
        return [
            make_row(order, pair_index, side)
            for order in (1, 2, 3)
            for pair_index in range(pair_count)
            for side in ("hidden", "observed")
        ]

    def test_selects_balanced_complete_pairs_deterministically(self):
        rows = self.rows()
        selected, manifest = select_pilot_rows(
            rows, pairs_per_order=2, seed=7, num_samples_per_prompt=16
        )
        reversed_selected, reversed_manifest = select_pilot_rows(
            reversed(rows), pairs_per_order=2, seed=7, num_samples_per_prompt=16
        )
        self.assertEqual(selected, reversed_selected)
        self.assertEqual(manifest, reversed_manifest)
        self.assertEqual(len(selected), 12)
        self.assertEqual(manifest["pair_count"], 6)
        self.assertEqual(manifest["expected_candidate_count"], 192)
        self.assertEqual(
            manifest["order_counts"], {"1": 4, "2": 4, "3": 4}
        )
        self.assertEqual(
            manifest["intervention_counts"], {"hidden": 6, "observed": 6}
        )
        self.assertTrue(
            all(count == 2 for count in manifest["bucket_counts"].values())
        )

    def test_rejects_incomplete_or_insufficient_pairs(self):
        rows = self.rows(pair_count=1)
        rows.pop()
        with self.assertRaisesRegex(ValueError, "Incomplete observed/hidden pair"):
            select_pilot_rows(rows, pairs_per_order=1)

        with self.assertRaisesRegex(ValueError, "only 1 eligible pairs"):
            select_pilot_rows(self.rows(pair_count=1), pairs_per_order=2)


if __name__ == "__main__":
    unittest.main()
