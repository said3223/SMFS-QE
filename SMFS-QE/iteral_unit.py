# iteral_unit.py
# -*- coding: utf-8 -*-
"""
Iteral Unit — a minimal, framework‑agnostic building block that follows the diagram:

    Inputs:   S_n, U_n            ->  External semantic stimulus
    Context:  C                    ->  Conditioning signal
    Engine:   S_{n+} = f(A_n S_n + U_n, C)
    Outputs:  E_{n+1}             ->  Semantic emission
    State:    S_n                  ->  Persistent internal state

The module is deliberately typed and pluggable:
- You can pass any numeric container (NumPy arrays, PyTorch tensors, plain lists).
- Linear operator A_n may be a matrix-like object or a callable `A(x) -> x'`.
- Nonlinearity f(x, C) and emission g(S_{n+}, C) are user-supplied callables.
- No implicit threading / asyncio. Deterministic and testable.

Default behaviour (if you do not pass custom functions):
- Uses NumPy (if available) for vector math. Falls back to Python lists.
- f: tanh-like activation with optional context gating.
- g: linear projection with optional bias (learnable W_out, b_out if NumPy is available).

This file is pure-Python and can be embedded in bigger systems (e.g., CBC / SMFS-QE).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional, Tuple, Protocol, Union, List
from datetime import datetime, timezone
import math
import random

try:  # optional NumPy for convenience
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore


ArrayLike = Union[' _np.ndarray', List[float], Tuple[float, ...], Any]  # loose by design


class LinearOp(Protocol):
    def __call__(self, x: ArrayLike) -> ArrayLike: ...


class TransformFn(Protocol):
    def __call__(self, x: ArrayLike, ctx: Any) -> ArrayLike: ...


class EmitFn(Protocol):
    def __call__(self, s_next: ArrayLike, ctx: Any) -> Any: ...


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


# ---------------------------- Math helpers ----------------------------------

def _to_array(x: ArrayLike) -> ArrayLike:
    if _np is not None:
        if isinstance(x, _np.ndarray):
            return x
        return _np.asarray(x, dtype=_np.float64)
    # Fallback: keep lists/tuples as-is
    return list(x) if isinstance(x, (list, tuple)) else x


def _matmul(a: ArrayLike, x: ArrayLike) -> ArrayLike:
    if _np is not None:
        return _np.matmul(_to_array(a), _to_array(x))
    # naive list dot for 2Dx1D
    if isinstance(a, list) and isinstance(x, list):
        return [sum(ai * xi for ai, xi in zip(row, x)) for row in a]
    # last resort: assume callable or identity
    try:
        return a @ x  # type: ignore
    except Exception:
        return x


def _add(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    if _np is not None:
        return _to_array(x) + _to_array(y)
    if isinstance(x, list) and isinstance(y, list):
        return [xi + yi for xi, yi in zip(x, y)]
    try:
        return x + y  # type: ignore
    except Exception:
        return y


def _tanh(x: ArrayLike) -> ArrayLike:
    if _np is not None:
        return _np.tanh(_to_array(x))
    if isinstance(x, list):
        return [math.tanh(v) for v in x]
    return math.tanh(x)  # type: ignore


# ---------------------------- Defaults --------------------------------------

def default_linear(A: Optional[ArrayLike]) -> LinearOp:
    if A is None:
        return lambda x: x
    if callable(A):
        return A  # user-supplied linear operator
    def op(x: ArrayLike) -> ArrayLike:
        return _matmul(A, x)
    return op


def default_f(x: ArrayLike, ctx: Any) -> ArrayLike:
    """tanh with soft context gating.

    If context is a scalar c, output = tanh(x) * sigma(c).
    If context is a vector, scales elementwise where possible.
    Otherwise context is ignored.
    """
    y = _tanh(x)
    if ctx is None:
        return y
    if _np is not None:
        try:
            if _np.isscalar(ctx):
                gate = 1.0 / (1.0 + math.exp(-float(ctx)))
                return y * gate  # type: ignore
            c = _np.asarray(ctx, dtype=_np.float64)
            # broadcast (min shape)
            return y * (1.0 / (1.0 + _np.exp(-c)))
        except Exception:
            return y
    # list fallback
    try:
        c = float(ctx)
        gate = 1.0 / (1.0 + math.exp(-c))
        if isinstance(y, list):
            return [gate * v for v in y]
        return gate * y  # type: ignore
    except Exception:
        return y


def default_g_factory(dim_out: Optional[int] = None) -> EmitFn:
    """Creates a simple linear emission `E = W_out * S + b` if NumPy exists; else identity.

    The parameters are random‑initialized for demonstration and can be replaced.
    """
    if _np is None:
        return lambda s_next, ctx: s_next

    # lazy parameters captured by closure
    W: Optional[_np.ndarray] = None
    b: Optional[_np.ndarray] = None

    def init_if_needed(s_next: ArrayLike) -> None:
        nonlocal W, b
        if W is not None:
            return
        s_arr = _to_array(s_next)
        in_dim = int(s_arr.shape[-1]) if hasattr(s_arr, 'shape') else len(s_arr)  # type: ignore
        out_dim = dim_out or in_dim
        rnd = _np.random.default_rng(42)
        W = rnd.normal(0, 1 / max(1, in_dim), size=(out_dim, in_dim))
        b = _np.zeros((out_dim,))

    def g(s_next: ArrayLike, ctx: Any) -> _np.ndarray:
        nonlocal W, b
        init_if_needed(s_next)
        assert W is not None and b is not None
        return _np.matmul(W, _to_array(s_next)) + b

    return g


# ---------------------------- Core types ------------------------------------

@dataclass(frozen=True)
class IteralInputs:
    S_n: ArrayLike                 # previous state (can be the unit's current state)
    U_n: ArrayLike                 # external semantic stimulus
    A_n: Optional[Union[ArrayLike, LinearOp]] = None  # linear transform
    C: Optional[Any] = None        # context


@dataclass(frozen=True)
class IteralOutputs:
    E_np1: Any                     # semantic emission E_{n+1}
    S_np: ArrayLike                # next state S_{n+}
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IteralUnit:
    """A stateful transformation engine implementing:

        S_{n+} = f(A_n S_n + U_n, C)
        E_{n+1} = g(S_{n+}, C)

    Parameters
    ----------
    state : ArrayLike
        Initial internal state S_0.
    f : TransformFn, optional
        Nonlinear transform (defaults to `default_f`).
    g : EmitFn, optional
        Emission function (defaults to linear projection if NumPy present; identity otherwise).
    name : str
        Human‑readable identifier (used in traces / logs).
    """

    state: ArrayLike
    f: TransformFn = field(default=default_f)
    g: EmitFn = field(default_factory=lambda: default_g_factory(dim_out=None))
    name: str = "iteral"

    # --------- public API ---------

    def step(self, inputs: IteralInputs) -> IteralOutputs:
        """Performs a single Iteral step and mutates internal state.

        Returns the emission and the new state.
        """
        lin = default_linear(inputs.A_n)
        Ax = lin(inputs.S_n)
        summed = _add(Ax, inputs.U_n)
        s_next = self.f(summed, inputs.C)
        e_next = self.g(s_next, inputs.C)
        self.state = s_next  # mutate internal state
        return IteralOutputs(
            E_np1=e_next,
            S_np=s_next,
            info={
                "ts": _now_ts(),
                "name": self.name,
                "had_numpy": _np is not None,
                "shapes": _infer_shapes(inputs.S_n, inputs.U_n, s_next, e_next),
            },
        )

    def reset(self, state: Optional[ArrayLike] = None) -> None:
        if state is not None:
            self.state = state

    # convenience sugar mirroring the formula components
    def apply(self, S_n: ArrayLike, U_n: ArrayLike, A_n: Optional[Union[ArrayLike, LinearOp]] = None,
              C: Optional[Any] = None) -> IteralOutputs:
        return self.step(IteralInputs(S_n=S_n, U_n=U_n, A_n=A_n, C=C))

    # stateless preview (does not mutate internal state)
    def preview(self, S_n: ArrayLike, U_n: ArrayLike, A_n: Optional[Union[ArrayLike, LinearOp]] = None,
                C: Optional[Any] = None) -> IteralOutputs:
        current = self.state
        out = self.apply(S_n, U_n, A_n, C)
        self.state = current
        return out


# ---------------------------- Utilities -------------------------------------

def _infer_shapes(S_n: Any, U_n: Any, S_np: Any, E_np1: Any) -> Dict[str, Any]:
    def shape_of(x: Any) -> Any:
        if _np is not None and isinstance(x, _np.ndarray):
            return tuple(int(d) for d in x.shape)
        if isinstance(x, (list, tuple)):
            return (len(x),)
        return type(x).__name__
    return {
        "S_n": shape_of(S_n),
        "U_n": shape_of(U_n),
        "S_n_plus": shape_of(S_np),
        "E_n_plus_1": shape_of(E_np1),
    }


# ---------------------------- Example usage ---------------------------------

def _example() -> None:  # pragma: no cover
    print("Running Iteral Unit demo (NumPy available:", _np is not None, ")")

    # Initial state (vector of size 4)
    S0 = _np.zeros(4) if _np is not None else [0.0, 0.0, 0.0, 0.0]

    # Linear operator (4x4), stimulus and context
    if _np is not None:
        A = _np.eye(4) * 0.9
        U = _np.array([0.1, 0.0, -0.1, 0.2])
        C = 0.5  # mild gating
    else:
        A = [[0.9,0,0,0],[0,0.9,0,0],[0,0,0.9,0],[0,0,0,0.9]]
        U = [0.1, 0.0, -0.1, 0.2]
        C = 0.5

    # Custom emission: project to 2 dims if NumPy available
    g = default_g_factory(dim_out=2) if _np is not None else (lambda s, c: s)

    unit = IteralUnit(state=S0, g=g, name="demo-iteral")

    out = unit.apply(S_n=unit.state, U_n=U, A_n=A, C=C)
    print("E_{n+1}:", out.E_np1)
    print("S_{n+} :", out.S_np)
    print("info   :", out.info)


if __name__ == "__main__":  # pragma: no cover
    _example()
