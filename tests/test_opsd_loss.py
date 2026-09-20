import importlib.util
import unittest


HAS_OPSD_RUNTIME = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "trl")
)

if HAS_OPSD_RUNTIME:
    import torch
    import torch.nn.functional as F

    from opsd.trainer import OPSDTrainer


@unittest.skipUnless(HAS_OPSD_RUNTIME, "isolated OPSD runtime is not installed")
class OPSDLossTest(unittest.TestCase):
    def setUp(self):
        self.student = torch.tensor(
            [[[2.0, 0.0, -1.0], [0.0, 1.0, -0.5]]], dtype=torch.float64
        )
        self.teacher = torch.tensor(
            [[[0.0, 2.0, -1.0], [1.5, -0.5, 0.0]]], dtype=torch.float64
        )

    def _per_token_forward_kl(self):
        student_log_probs = F.log_softmax(self.student, dim=-1)
        teacher_log_probs = F.log_softmax(self.teacher, dim=-1)
        return F.kl_div(
            student_log_probs,
            teacher_log_probs,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)

    def test_forward_kl_sums_vocabulary_before_token_reduction(self):
        labels = torch.tensor([[1, 1]])
        expected = self._per_token_forward_kl().mean()
        actual = OPSDTrainer.generalized_jsd_loss(
            self.student,
            self.teacher,
            labels=labels,
            beta=0,
            temperature=1.0,
            token_clip=None,
        )
        torch.testing.assert_close(actual, expected)
        self.assertGreaterEqual(actual.item(), 0.0)

    def test_token_clip_applies_after_vocabulary_sum(self):
        labels = torch.tensor([[1, -100]])
        clip = 0.05
        expected = self._per_token_forward_kl()[0, 0].clamp(max=clip)
        actual = OPSDTrainer.generalized_jsd_loss(
            self.student,
            self.teacher,
            labels=labels,
            beta=0,
            temperature=1.0,
            token_clip=clip,
        )
        torch.testing.assert_close(actual, expected)

    def test_position_ids_ignore_left_padding(self):
        attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
        expected = torch.tensor([[0, 0, 0, 1, 2], [0, 0, 1, 2, 3]])
        actual = OPSDTrainer._position_ids_from_attention_mask(attention_mask)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
