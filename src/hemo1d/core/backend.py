from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import ufl
import math


@runtime_checkable
class Backend(Protocol):
    """Shared math surface for backend-agnostic code."""

    pi: float

    def vector(self, values: Any) -> Any: ...

    def matrix(self, values: Any) -> Any: ...

    def sqrt(self, value: Any) -> Any: ...

    def dot(self, left: Any, right: Any) -> Any: ...

    def inner(self, left: Any, right: Any) -> Any: ...


class NumpyBackend:
    pi = np.pi

    def vector(self, values: Any) -> Any:
        return np.asarray(values)

    def matrix(self, values: Any) -> Any:
        return np.asarray(values)

    def sqrt(self, value: Any) -> Any:
        return np.sqrt(value)

    def dot(self, left: Any, right: Any) -> Any:
        return np.dot(left, right)

    def inner(self, left: Any, right: Any) -> Any:
        return np.inner(left, right)


class UFLBackend:
    pi = math.pi

    def vector(self, values: Any) -> Any:
        return ufl.as_vector(values)

    def matrix(self, values: Any) -> Any:
        return ufl.as_matrix(values)

    def sqrt(self, value: Any) -> Any:
        return ufl.sqrt(value)

    def dot(self, left: Any, right: Any) -> Any:
        return ufl.dot(left, right)

    def inner(self, left: Any, right: Any) -> Any:
        return ufl.inner(left, right)
    

NP_BACKEND = NumpyBackend()
UFL_BACKEND = UFLBackend()