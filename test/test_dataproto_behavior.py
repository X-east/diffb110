"""
DataProto 行为测试：验证 select_idxs / slice / concat 等核心操作。

覆盖：
    - select_idxs: torch.Tensor, np.ndarray, list, 空索引
    - slice: start/end/step 组合
    - concat: CUDA tensor + list/tuple/numpy 元数据组合
    - repeat: interleave / stack
    - 往返: select_idxs + concat 一致性
    - 边界: 空索引, 单元素

注意：from_dict 会自动将 non_tensors 转为 np.array(dtype=object)，
因此所有 non_tensor_batch 值在比较时需转为 list 或使用 numpy 断言。
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from ddpo.runtime.protocol import DataProto

# 尝试 CUDA，不可用时回退 CPU
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_cuda_tensor(*shape, **kwargs):
    """创建 CUDA 或 CPU tensor（根据可用设备）。"""
    return torch.randn(*shape, **kwargs, device=_DEVICE)


def _to_list(value):
    """将 numpy array 或 tuple 转为 Python list，方便断言比较。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


class TestDataProtoBasics(unittest.TestCase):
    """基础创建与属性。"""

    def test_from_dict_basic(self):
        dp = DataProto.from_dict(
            tensors={"x": torch.randn(10, 3)},
            non_tensors={"meta": ["a"] * 10},
        )
        self.assertEqual(len(dp), 10)

    def test_from_dict_no_non_tensors(self):
        dp = DataProto.from_dict(tensors={"x": torch.randn(5, 4)})
        self.assertEqual(len(dp), 5)
        self.assertEqual(dp.non_tensor_batch, {})

    def test_from_dict_mixed_device(self):
        dp = DataProto.from_dict(tensors={"x": _make_cuda_tensor(8, 2)})
        self.assertEqual(len(dp), 8)
        self.assertEqual(dp.batch["x"].device.type, _DEVICE.type)

    def test_from_single_dict_list(self):
        """from_single_dict 含 list 非 tensor 数据。"""
        dp = DataProto.from_single_dict({
            "x": torch.randn(3, 2),
            "lst": ["a", "b", "c"],
        })
        self.assertEqual(len(dp), 3)
        self.assertListEqual(_to_list(dp.non_tensor_batch["lst"]), ["a", "b", "c"])

    def test_chunk_basic(self):
        """chunk 基本测试。"""
        dp = DataProto.from_dict(
            tensors={"x": torch.randn(6, 2)},
            non_tensors={"label": ["a", "b", "c", "d", "e", "f"]},
        )
        chunks = dp.chunk(3)
        self.assertEqual(len(chunks), 3)
        for c in chunks:
            self.assertEqual(len(c), 2)
        all_labels = []
        for c in chunks:
            all_labels.extend(_to_list(c.non_tensor_batch["label"]))
        self.assertListEqual(all_labels, ["a", "b", "c", "d", "e", "f"])

    def test_chunk_preserves_device(self):
        dp = DataProto.from_dict(tensors={"x": _make_cuda_tensor(6, 2)})
        chunks = dp.chunk(2)
        for c in chunks:
            self.assertEqual(c.batch["x"].device.type, _DEVICE.type)


class TestSelectIdxs(unittest.TestCase):
    """select_idxs 行为测试。"""

    def setUp(self):
        self.dp = DataProto.from_dict(
            tensors={"x": _make_cuda_tensor(10, 4)},
            non_tensors={"lst": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]},
        )

    def test_select_torch_tensor(self):
        idxs = torch.tensor([0, 2, 4])
        result = self.dp.select_idxs(idxs)
        self.assertEqual(len(result), 3)
        self.assertListEqual(_to_list(result.non_tensor_batch["lst"]), ["a", "c", "e"])

    def test_select_numpy_array(self):
        idxs = np.array([1, 3, 5])
        result = self.dp.select_idxs(idxs)
        self.assertEqual(len(result), 3)
        self.assertListEqual(_to_list(result.non_tensor_batch["lst"]), ["b", "d", "f"])

    def test_select_python_list(self):
        idxs = [0, 5, 9]
        result = self.dp.select_idxs(idxs)
        self.assertEqual(len(result), 3)
        self.assertListEqual(_to_list(result.non_tensor_batch["lst"]), ["a", "f", "j"])

    def test_select_single_index(self):
        result = self.dp.select_idxs(torch.tensor([0]))
        self.assertEqual(len(result), 1)
        self.assertListEqual(_to_list(result.non_tensor_batch["lst"]), ["a"])

    def test_select_empty(self):
        result = self.dp.select_idxs(torch.tensor([], dtype=torch.int32))
        self.assertEqual(len(result), 0)
        self.assertEqual(len(result.non_tensor_batch["lst"]), 0)

    def test_select_preserves_tensor_device(self):
        idxs = torch.tensor([0, 1, 2])
        result = self.dp.select_idxs(idxs)
        self.assertEqual(result.batch["x"].device.type, _DEVICE.type)


class TestSlice(unittest.TestCase):
    """slice 方法行为测试。"""

    def setUp(self):
        self.dp = DataProto.from_dict(
            tensors={"x": _make_cuda_tensor(10, 3)},
            non_tensors={"items": list(range(10))},
        )

    def test_slice_basic(self):
        result = self.dp.slice(2, 7)
        self.assertEqual(len(result), 5)
        self.assertListEqual(_to_list(result.non_tensor_batch["items"]), list(range(2, 7)))

    def test_slice_from_start(self):
        result = self.dp.slice(end=3)
        self.assertEqual(len(result), 3)
        self.assertListEqual(_to_list(result.non_tensor_batch["items"]), [0, 1, 2])

    def test_slice_to_end(self):
        result = self.dp.slice(7)
        self.assertEqual(len(result), 3)
        self.assertListEqual(_to_list(result.non_tensor_batch["items"]), [7, 8, 9])

    def test_slice_with_step(self):
        result = self.dp.slice(0, 10, 2)
        self.assertEqual(len(result), 5)
        self.assertListEqual(_to_list(result.non_tensor_batch["items"]), [0, 2, 4, 6, 8])

    def test_slice_via_getitem(self):
        result = self.dp[3:8]
        self.assertEqual(len(result), 5)
        self.assertListEqual(_to_list(result.non_tensor_batch["items"]), list(range(3, 8)))

    def test_slice_preserves_device(self):
        result = self.dp.slice(0, 5)
        self.assertEqual(result.batch["x"].device.type, _DEVICE.type)


class TestConcat(unittest.TestCase):
    """concat 静态方法行为测试。"""

    def _make_dp(self, start, n):
        return DataProto.from_dict(
            tensors={"x": _make_cuda_tensor(n, 2)},
            non_tensors={"label": [f"item_{i}" for i in range(start, start + n)]},
        )

    def test_concat_two(self):
        a = self._make_dp(0, 5)
        b = self._make_dp(5, 3)
        result = DataProto.concat([a, b])
        self.assertEqual(len(result), 8)
        self.assertListEqual(
            _to_list(result.non_tensor_batch["label"]),
            [f"item_{i}" for i in range(8)],
        )

    def test_concat_single(self):
        a = self._make_dp(0, 5)
        result = DataProto.concat([a])
        self.assertEqual(len(result), 5)
        self.assertListEqual(
            _to_list(result.non_tensor_batch["label"]),
            [f"item_{i}" for i in range(5)],
        )

    def test_concat_three(self):
        parts = [self._make_dp(i * 3, 3) for i in range(3)]
        result = DataProto.concat(parts)
        self.assertEqual(len(result), 9)

    def test_concat_preserves_device(self):
        a = self._make_dp(0, 3)
        b = self._make_dp(3, 4)
        result = DataProto.concat([a, b])
        self.assertEqual(result.batch["x"].device.type, _DEVICE.type)

    def test_concat_list_metadata(self):
        """验证 list 元数据正确拼接（内部转为 numpy array）。"""
        a = DataProto.from_dict(
            tensors={"x": _make_cuda_tensor(2, 2)},
            non_tensors={"tags": ["t1", "t2"]},
        )
        b = DataProto.from_dict(
            tensors={"x": _make_cuda_tensor(3, 2)},
            non_tensors={"tags": ["t3", "t4", "t5"]},
        )
        result = DataProto.concat([a, b])
        self.assertListEqual(
            _to_list(result.non_tensor_batch["tags"]),
            ["t1", "t2", "t3", "t4", "t5"],
        )

    def test_concat_tuple_metadata(self):
        """验证 tuple 元数据正确拼接。"""
        a = DataProto.from_dict(
            tensors={"x": _make_cuda_tensor(2, 2)},
            non_tensors={"ids": (1, 2)},
        )
        b = DataProto.from_dict(
            tensors={"x": _make_cuda_tensor(3, 2)},
            non_tensors={"ids": (3, 4, 5)},
        )
        result = DataProto.concat([a, b])
        self.assertListEqual(
            _to_list(result.non_tensor_batch["ids"]),
            [1, 2, 3, 4, 5],
        )

    def test_concat_numpy_metadata(self):
        """验证 np.ndarray 元数据正确拼接。"""
        a = DataProto.from_dict(
            tensors={"x": _make_cuda_tensor(2, 2)},
            non_tensors={"arr": np.array([0.1, 0.2])},
        )
        b = DataProto.from_dict(
            tensors={"x": _make_cuda_tensor(3, 2)},
            non_tensors={"arr": np.array([0.3, 0.4, 0.5])},
        )
        result = DataProto.concat([a, b])
        np.testing.assert_array_almost_equal(
            result.non_tensor_batch["arr"], np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        )


class TestRepeat(unittest.TestCase):
    """repeat 方法行为测试。"""

    def setUp(self):
        self.dp = DataProto.from_dict(
            tensors={"x": _make_cuda_tensor(3, 2)},
            non_tensors={"label": ["a", "b", "c"]},
        )

    def test_repeat_interleave(self):
        result = self.dp.repeat(repeat_times=2, interleave=True)
        self.assertEqual(len(result), 6)
        self.assertListEqual(
            _to_list(result.non_tensor_batch["label"]),
            ["a", "a", "b", "b", "c", "c"],
        )

    def test_repeat_stack(self):
        result = self.dp.repeat(repeat_times=2, interleave=False)
        self.assertEqual(len(result), 6)
        self.assertListEqual(
            _to_list(result.non_tensor_batch["label"]),
            ["a", "b", "c", "a", "b", "c"],
        )

    def test_repeat_preserves_device(self):
        result = self.dp.repeat(repeat_times=2)
        self.assertEqual(result.batch["x"].device.type, _DEVICE.type)


class TestRoundTrip(unittest.TestCase):
    """select_idxs + concat 往返测试。"""

    def test_select_concat_roundtrip(self):
        dp = DataProto.from_dict(
            tensors={"x": _make_cuda_tensor(10, 3)},
            non_tensors={"tag": [f"t{i}" for i in range(10)]},
        )
        a = dp.select_idxs(torch.tensor([0, 1, 2, 3]))
        b = dp.select_idxs(torch.tensor([4, 5, 6, 7, 8, 9]))
        self.assertEqual(len(a), 4)
        self.assertEqual(len(b), 6)

        merged = DataProto.concat([a, b])
        self.assertEqual(len(merged), 10)
        self.assertListEqual(
            _to_list(merged.non_tensor_batch["tag"]),
            [f"t{i}" for i in range(10)],
        )

        # 验证 tensor 值一致
        torch.testing.assert_close(merged.batch["x"], dp.batch["x"])


if __name__ == "__main__":
    unittest.main()
