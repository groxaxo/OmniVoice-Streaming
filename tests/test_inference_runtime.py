import unittest
from unittest.mock import patch

import torch

from omnivoice.models.omnivoice import (
    _build_block_mask_document_ids,
    _build_inference_attention_mask,
    _mask_mod_packed,
)
from omnivoice.utils.common import configure_cuda_inference, resolve_inference_dtype


class InferenceRuntimeTests(unittest.TestCase):
    def test_resolve_inference_dtype_prefers_bfloat16_on_ampere(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 6)),
        ):
            self.assertEqual(resolve_inference_dtype("cuda:0"), torch.bfloat16)

    def test_resolve_inference_dtype_falls_back_to_float16_on_pre_ampere_cuda(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        ):
            self.assertEqual(resolve_inference_dtype("cuda:0"), torch.float16)

    def test_resolve_inference_dtype_uses_float32_off_cuda(self) -> None:
        self.assertEqual(resolve_inference_dtype("cpu"), torch.float32)
        self.assertEqual(resolve_inference_dtype("mps"), torch.float32)

    def test_build_block_mask_document_ids_keeps_padding_self_only(self) -> None:
        document_ids = _build_block_mask_document_ids(
            [4, 2],
            max_seq_len=4,
            device=torch.device("cpu"),
        )
        expected = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, -1, -2],
            ],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(document_ids.cpu(), expected))

        self.assertTrue(_mask_mod_packed(document_ids, 1, None, 0, 1))
        self.assertFalse(_mask_mod_packed(document_ids, 1, None, 0, 2))
        self.assertFalse(_mask_mod_packed(document_ids, 1, None, 2, 3))
        self.assertTrue(_mask_mod_packed(document_ids, 1, None, 2, 2))

    # ------------------------------------------------------------------
    # Pass 2 tests
    # ------------------------------------------------------------------

    def test_build_block_mask_document_ids_vectorized_matches_expected(self) -> None:
        """Vectorised implementation must produce the same output for varied inputs."""
        for lengths, max_seq_len, expected_rows in [
            ([4, 2], 4, [[0, 0, 0, 0], [0, 0, -1, -2]]),
            ([1, 1, 1], 3, [[0, -1, -2], [0, -1, -2], [0, -1, -2]]),
            ([3, 3], 3, [[0, 0, 0], [0, 0, 0]]),
        ]:
            with self.subTest(lengths=lengths):
                result = _build_block_mask_document_ids(
                    lengths, max_seq_len, device=torch.device("cpu")
                )
                expected = torch.tensor(expected_rows, dtype=torch.int32)
                self.assertTrue(torch.equal(result, expected))

    def test_build_inference_attention_mask_keeps_padding_self_only(self) -> None:
        mask = _build_inference_attention_mask(
            c_lens=[5],
            target_lens=[3],
            max_seq_len=5,
            device=torch.device("cpu"),
        )

        self.assertEqual(mask.shape, (2, 1, 5, 5))

        cond = mask[0, 0]
        uncond = mask[1, 0]

        self.assertTrue(torch.all(cond[:5, :5]))
        self.assertTrue(torch.all(uncond[:3, :3]))
        self.assertFalse(uncond[3, :3].any())
        self.assertFalse(uncond[4, :4].any())
        self.assertTrue(uncond[3, 3])
        self.assertTrue(uncond[4, 4])

    def test_combined_max_argmax_matches_separate_calls(self) -> None:
        """log_probs.max(dim=-1) must return the same values as separate max/argmax."""
        torch.manual_seed(0)
        log_probs = torch.randn(1, 8, 16, 512)
        combined_scores, combined_tokens = log_probs.max(dim=-1)
        separate_tokens = log_probs.argmax(dim=-1)
        separate_scores = log_probs.max(dim=-1)[0]
        self.assertTrue(torch.equal(combined_tokens, separate_tokens))
        self.assertTrue(torch.equal(combined_scores, separate_scores))

    def test_configure_cuda_inference_sets_cudnn_benchmark(self) -> None:
        """configure_cuda_inference must enable cudnn.benchmark on CUDA devices."""
        import torch.backends.cudnn as cudnn
        original = cudnn.benchmark
        try:
            cudnn.benchmark = False
            with patch(
                "omnivoice.utils.common.resolve_device_string", return_value="cuda:0"
            ):
                configure_cuda_inference("cuda:0")
            self.assertTrue(cudnn.benchmark)
        finally:
            cudnn.benchmark = original

    def test_voice_prompt_lru_cache_evicts_oldest(self) -> None:
        """LRU cache must evict the least-recently-used entry when full."""
        from omnivoice.openai_tts_server import _VoicePromptLRUCache

        cache = _VoicePromptLRUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Access 'a' to make it recently used
        _ = cache.get("a")
        # Adding a 4th entry should evict 'b' (oldest unused)
        cache["d"] = 4
        self.assertIsNone(cache.get("b"), "'b' should have been evicted")
        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.get("c"), 3)
        self.assertEqual(cache.get("d"), 4)
        self.assertEqual(len(cache), 3)

    def test_voice_prompt_lru_cache_clear(self) -> None:
        from omnivoice.openai_tts_server import _VoicePromptLRUCache

        cache = _VoicePromptLRUCache(maxsize=10)
        cache["x"] = object()
        cache.clear()
        self.assertEqual(len(cache), 0)


if __name__ == "__main__":
    unittest.main()
