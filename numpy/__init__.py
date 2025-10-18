"""Minimal compatibility layer for the handful of NumPy features used in tests."""
from __future__ import annotations

import builtins
import math
from typing import Any, List, Sequence, Union, overload

Number = Union[int, float]
float32 = float


class ndarray:
    """Array container supporting a tiny subset of NumPy semantics."""

    def __init__(self, data: Any) -> None:
        self._data = _to_nested(data)

    @property
    def ndim(self) -> int:
        return _ndim(self._data)

    @property
    def size(self) -> int:
        return len(_flatten(self._data))

    def astype(self, dtype: type, copy: bool = True) -> "ndarray":
        if dtype not in (float32, float):
            raise TypeError("Only float32 casts are supported")
        casted = _apply_unary(self._data, float)
        if copy:
            return ndarray(casted)
        self._data = casted
        return self

    def reshape(self, *shape: int) -> "ndarray":
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])  # type: ignore[assignment]
        if len(shape) == 1 and shape[0] == -1:
            return ndarray(_flatten(self._data))
        if shape == (-1,):
            return ndarray(_flatten(self._data))
        raise ValueError("Only reshape(-1) is supported in the compatibility layer")

    def tolist(self):
        return _clone(self._data)

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __mul__(self, other: Any) -> "ndarray":
        return ndarray(_apply_binary(self._data, other, lambda a, b: a * b))

    def __rmul__(self, other: Any) -> "ndarray":
        return self.__mul__(other)

    def __add__(self, other: Any) -> "ndarray":
        return ndarray(_apply_binary(self._data, other, lambda a, b: a + b))

    def __radd__(self, other: Any) -> "ndarray":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "ndarray":
        return ndarray(_apply_binary(self._data, other, lambda a, b: a - b))

    def __rsub__(self, other: Any) -> "ndarray":
        return ndarray(_apply_binary(other, self._data, lambda a, b: a - b))

    def __truediv__(self, other: Any) -> "ndarray":
        return ndarray(_apply_binary(self._data, other, lambda a, b: a / b))

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"ndarray({self._data!r})"


@overload
def array(data: ndarray, dtype: type | None = ...) -> ndarray: ...


@overload
def array(data: Sequence[Number], dtype: type | None = ...) -> ndarray: ...


@overload
def array(data: Sequence[Sequence[Number]], dtype: type | None = ...) -> ndarray: ...


def array(data, dtype: type | None = None) -> ndarray:
    values = data.tolist() if isinstance(data, ndarray) else _to_nested(data)
    if dtype in (float32, float):
        values = _apply_unary(values, float)
    return ndarray(values)


def linspace(start: Number, stop: Number, num: int, endpoint: bool = True) -> ndarray:
    if num <= 0:
        raise ValueError("num must be positive")
    if num == 1:
        return ndarray([float(stop if endpoint else start)])
    step = (stop - start) / (num - (1 if endpoint else 0))
    values = [start + step * i for i in range(num)]
    if endpoint:
        values[-1] = stop
    return ndarray(values)


def stack(arrays: Sequence[Union[ndarray, Sequence[Number]]], axis: int = 0) -> ndarray:
    converted = [
        arr.tolist() if isinstance(arr, ndarray) else _to_nested(arr)
        for arr in arrays
    ]
    if not converted:
        return ndarray([])
    if axis == 0:
        return ndarray(converted)
    if axis == 1:
        length = len(converted[0])
        if any(len(item) != length for item in converted):
            raise ValueError("All arrays must share the same length")
        return ndarray([[arr[i] for arr in converted] for i in range(length)])
    raise ValueError("Only axis 0 or 1 are supported")


def mean(
    arr: Union[ndarray, Sequence[Number], Sequence[Sequence[Number]]],
    axis: int | None = None,
):
    data = arr.tolist() if isinstance(arr, ndarray) else _to_nested(arr)
    if axis is None:
        flat = _flatten(data)
        return sum(flat) / len(flat) if flat else 0.0
    if axis == 0:
        if not data:
            return ndarray([])
        cols = len(data[0])
        return ndarray([
            sum(row[i] for row in data) / len(data)
            for i in range(cols)
        ])
    if axis == 1:
        return ndarray([
            (sum(row) / len(row)) if row else 0.0
            for row in data
        ])
    raise ValueError("Only axis 0 or 1 are supported")


def abs(arr):
    data = arr.tolist() if isinstance(arr, ndarray) else _to_nested(arr)
    return ndarray(_apply_unary(data, builtins.abs))


def max(arr):
    data = arr.tolist() if isinstance(arr, ndarray) else _to_nested(arr)
    flat = _flatten(data)
    if not flat:
        raise ValueError("max() arg is an empty sequence")
    return builtins.max(flat)


def sin(arr):
    data = arr.tolist() if isinstance(arr, ndarray) else _to_nested(arr)
    return ndarray(_apply_unary(data, math.sin))


def clip(arr, minimum: Number, maximum: Number):
    data = arr.tolist() if isinstance(arr, ndarray) else _to_nested(arr)
    return ndarray([
        max(minimum, min(maximum, value))
        for value in data
    ])


def _to_nested(value: Any):
    if isinstance(value, ndarray):
        return _clone(value._data)
    if isinstance(value, (list, tuple)):
        return [_to_nested(item) for item in value]
    return float(value)


def _clone(value: Any):
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return value


def _ndim(value: Any) -> int:
    if isinstance(value, list) and value:
        return 1 + _ndim(value[0])
    if isinstance(value, list):
        return 1
    return 0


def _flatten(value: Any) -> List[float]:
    if isinstance(value, list):
        result: List[float] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [float(value)]


def _apply_unary(value: Any, func):
    if isinstance(value, list):
        return [_apply_unary(item, func) for item in value]
    return func(value)


def _apply_binary(left: Any, right: Any, func):
    left_data = left.tolist() if isinstance(left, ndarray) else _to_nested(left)
    right_data = right.tolist() if isinstance(right, ndarray) else _to_nested(right)
    if isinstance(left_data, list) and isinstance(right_data, list):
        if len(left_data) != len(right_data):
            raise ValueError("Operand shapes must match")
        return [
            _apply_binary(l_item, r_item, func)
            for l_item, r_item in zip(left_data, right_data)
        ]
    if isinstance(left_data, list):
        return [_apply_binary(item, right_data, func) for item in left_data]
    if isinstance(right_data, list):
        return [_apply_binary(left_data, item, func) for item in right_data]
    return func(left_data, right_data)


pi = math.pi
__all__ = [
    "ndarray",
    "array",
    "linspace",
    "stack",
    "mean",
    "abs",
    "max",
    "sin",
    "clip",
    "float32",
    "pi",
]
