"""CUDA regressions for visibility-aware SegPaged KV loads.

Compare the optimized kernel, the retained full-scan kernel, and a separately
computed FP32 dense reference. These are numerical tests, not timing tests.
Run directly with unittest or pytest; CUDA/Triton absence is an explicit skip.
"""

import importlib.util
from importlib import import_module
from pathlib import Path
import sys
import types
import unittest

import torch


_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/mem_cache/head_kv"
)
_PACKAGE_NAME = "_segpaged_page_skip_testpkg"
if _PACKAGE_NAME not in sys.modules:
    _package = types.ModuleType(_PACKAGE_NAME)
    _package.__path__ = [str(_SOURCE)]
    sys.modules[_PACKAGE_NAME] = _package
_public_attention = import_module(f"{_PACKAGE_NAME}.attention").paged_attention
_HAS_CUDA_TRITON = (
    torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
)


def _dense_fp32_reference(q, k, v, slots, lengths, positions, qpos, windows, sinks, group, causal):
    """Test-only concatenation; policies use logical positions, not page order."""
    q, k, v = (tensor.float().cpu() for tensor in (q, k, v))
    slots, lengths, positions, qpos, windows, sinks = (
        tensor.cpu() for tensor in (slots, lengths, positions, qpos, windows, sinks)
    )
    result = torch.zeros_like(q)
    for head in range(q.shape[0]):
        kv_head = head // group
        keys, values, logical_positions = [], [], []
        for page in range(slots.shape[1]):
            count = int(lengths[kv_head, page])
            if count:
                slot = int(slots[kv_head, page])
                keys.append(k[slot, :count])
                values.append(v[slot, :count])
                logical_positions.append(positions[kv_head, page, :count])
        if not keys:
            continue
        keys, values, logical_positions = map(torch.cat, (keys, values, logical_positions))
        visible = torch.ones((q.shape[1], keys.shape[0]), dtype=torch.bool)
        if causal:
            visible &= logical_positions[None, :] <= qpos[head, :, None]
        window, sink = int(windows[kv_head]), int(sinks[kv_head])
        if window > 0:
            visible &= (
                (logical_positions[None, :] >= qpos[head, :, None] - window + 1)
                | (logical_positions[None, :] < sink)
            )
        scores = q[head] @ keys.T * q.shape[-1] ** -0.5
        scores.masked_fill_(~visible, -torch.inf)
        nonempty = visible.any(dim=-1)
        result[head, nonempty] = torch.softmax(scores[nonempty], dim=-1) @ values
    return result


def _case(dtype, *, pages, page_size, queries):
    generator = torch.Generator().manual_seed(321)
    heads, group, dim = 2, 3, 13
    # Noncontiguous Q, K, V and descriptors exercise the launcher's strides.
    q = torch.randn(heads * group, dim, queries, generator=generator).to(
        device="cuda", dtype=dtype
    ).transpose(1, 2)
    k, v = (
        torch.randn(heads * pages, dim, page_size, generator=generator).to(
            device="cuda", dtype=dtype
        ).transpose(1, 2)
        for _ in range(2)
    )
    slots = torch.randperm(heads * pages, generator=generator).reshape(heads, pages)
    lengths = torch.full_like(slots, page_size, dtype=torch.int32)
    # Holes between live pages, partial tails, and distinct occurrences of one
    # physical page must not be mistaken for sorted, contiguous token ranges.
    slots[:, 1], lengths[:, 1] = -1, 0
    lengths[:, -1] = max(1, page_size // 2)
    slots[0, -1] = slots[0, 0]
    positions = torch.stack([
        torch.randperm(pages * page_size, generator=generator).reshape(pages, page_size)
        for _ in range(heads)
    ])
    positions.masked_fill_(torch.arange(page_size)[None, None, :] >= lengths[..., None], -1)
    qpos = torch.linspace(0, pages * page_size - 1, queries, dtype=torch.int64)
    qpos = qpos[None, :].expand(heads * group, -1).clone()
    qpos[1] += 2  # Different query positions inside the same GQA group.
    slots = torch.stack((slots, slots), dim=-1).to("cuda")[..., 0]
    positions = torch.stack((positions, positions), dim=-1).to("cuda")[..., 0]
    return (
        q, k, v, slots, lengths.to("cuda"), positions, qpos.to("cuda"),
        torch.tensor([9, 0], dtype=torch.int64, device="cuda"),
        torch.tensor([4, 0], dtype=torch.int64, device="cuda"),
    )


class SegPagedPageSkipAPITest(unittest.TestCase):
    def test_public_option_requires_boolean_and_preserves_cpu_result(self):
        q, k = torch.zeros(1, 1, 4), torch.zeros(1, 2, 4)
        v = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        args = (q, k, v, torch.tensor([[0]]), torch.tensor([[2]]), torch.tensor([[[0, 1]]]))
        options = {"query_positions": torch.tensor([1])}
        for invalid in (None, 1, "yes", torch.tensor(True)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(TypeError, "must be bool"):
                _public_attention(*args, **options, skip_invisible=invalid)
        for enabled in (False, True):
            actual = _public_attention(*args, **options, skip_invisible=enabled)
            torch.testing.assert_close(actual, v.mean(dim=1, keepdim=True))

    def test_public_split_count_requires_bounded_integer(self):
        q, k = torch.zeros(1, 1, 4), torch.zeros(1, 2, 4)
        v = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        args = (q, k, v, torch.tensor([[0]]), torch.tensor([[2]]), torch.tensor([[[0, 1]]]))
        options = {"query_positions": torch.tensor([1])}
        for invalid in (False, True, 0, -1, 129, 1.0, "8", torch.tensor(8)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "num_splits"):
                _public_attention(*args, **options, num_splits=invalid)
        for splits in (None, 1, 8, 32, 128):
            actual = _public_attention(*args, **options, num_splits=splits)
            torch.testing.assert_close(actual, v.mean(dim=1, keepdim=True))


@unittest.skipUnless(_HAS_CUDA_TRITON, "CUDA and Triton are required")
class SegPagedPageSkipTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = import_module(f"{_PACKAGE_NAME}.triton_attention")
        cls.kernel = staticmethod(module.triton_paged_attention)

    def _compare(self, case, *, causal, num_splits=None):
        before = tuple(tensor.clone() for tensor in case)
        expected = _dense_fp32_reference(*case, group=3, causal=causal)
        outputs = [
            self.kernel(
                *case, 3, case[0].shape[-1] ** -0.5, causal,
                skip_invisible=skip, num_splits=num_splits,
            )
            for skip in (False, True)
        ]
        tolerance = {
            torch.float32: 3e-5,
            torch.float16: 3e-3,
            torch.bfloat16: 2e-2,
        }[case[0].dtype]
        for output in outputs:
            self.assertTrue(torch.isfinite(output).all().item())
            torch.testing.assert_close(output.float().cpu(), expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(outputs[1], outputs[0], rtol=tolerance, atol=tolerance)
        for tensor, snapshot in zip(case, before):
            torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
        return outputs[1]

    def test_irregular_positions_gqa_masks_dtypes_and_partitions(self):
        # 4 pages uses a single partition; 19 pages exercises split-softmax.
        # 64-token pages also exercise more than one token tile per page.
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for causal in (False, True):
                for pages, page_size, queries in ((4, 7, 5), (19, 64, 3)):
                    with self.subTest(dtype=dtype, causal=causal, pages=pages, page_size=page_size):
                        case = _case(dtype, pages=pages, page_size=page_size, queries=queries)
                        self._compare(case, causal=causal)

    def test_all_masked_and_empty_partitions_return_zero(self):
        case = _case(torch.float32, pages=19, page_size=7, queries=1)
        case[5].add_(1000)  # All active keys lie strictly after the query.
        case[6].zero_()
        case[7].fill_(3)
        case[8].zero_()
        # Keep one head completely empty while the other has masked live pages.
        case[3][1].fill_(-1)
        case[4][1].zero_()
        result = self._compare(case, causal=True)
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_public_dispatch_sorted_windows_sink_overlap_and_empty_partitions(self):
        case = _case(torch.float32, pages=19, page_size=7, queries=3)
        case[5].copy_(torch.arange(133, device="cuda").reshape(1, 19, 7))
        case[6].copy_(torch.tensor([1, 2, 132], device="cuda"))
        expected = self._compare(case, causal=True)
        for enabled in (False, True):
            actual = _public_attention(
                *case[:6], query_positions=case[6], num_q_per_kv=3,
                windows=case[7], sinks=case[8], causal=True,
                backend="triton", skip_invisible=enabled,
            )
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)

    def test_empty_queries_and_zero_pages(self):
        case = _case(torch.float16, pages=4, page_size=7, queries=0)
        self.assertEqual(self._compare(case, causal=True).shape, (6, 0, 13))
        case = list(_case(torch.float16, pages=4, page_size=7, queries=2))
        case[3], case[4], case[5] = case[3][:, :0], case[4][:, :0], case[5][:, :0]
        result = self._compare(tuple(case), causal=True)
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_explicit_split_counts_visible_and_all_masked(self):
        # 19 pages verifies that 32/128 are capped; 129 pages executes 128
        # partitions, including unused trailing partitions and page-tail masks.
        for pages in (19, 129):
            for splits in (1, 8, 32, 128):
                for all_masked in (False, True):
                    with self.subTest(pages=pages, splits=splits, all_masked=all_masked):
                        case = _case(torch.float32, pages=pages, page_size=7, queries=2)
                        case[5].copy_(torch.arange(pages * 7, device="cuda").reshape(1, pages, 7))
                        case[6].copy_(torch.tensor([1, pages * 7 - 1], device="cuda"))
                        if all_masked:
                            case[5].add_(10000)
                            case[6].zero_()
                            case[8].zero_()
                        expected = self._compare(case, causal=True, num_splits=splits)
                        actual = _public_attention(
                            *case[:6], query_positions=case[6], num_q_per_kv=3,
                            windows=case[7], sinks=case[8], causal=True,
                            backend="triton", num_splits=splits,
                        )
                        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
                        if all_masked:
                            self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))


if __name__ == "__main__":
    unittest.main()
