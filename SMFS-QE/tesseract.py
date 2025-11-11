# tesseract.py
# -*- coding: utf-8 -*-
"""
Base Tesseract (t¤) semantic unit.

This module defines a compact, framework-agnostic representation of an "atom of meaning"
with a small algebra for composition (⊗), fusion (⊕), projection (π), and scoring.
It is intentionally minimal and dependency-light, but will use NumPy if present.

Design goals
------------
- Deterministic identity and hashing (for SMFS-QE / semantic inode).
- Optional dense semantics vector; works without NumPy.
- Provenance and context are first-class fields.
- Pure-Python, easily embedded into larger SRIS subsystems (CBC, SMFS-QE, RIU).

Core objects
------------
- Tesseract: semantic unit with facets (content, context, affect, provenance).
- Link: typed relation between two tesseracts.
- TGraph: tiny container for tesseracts + links (no external graph deps).

Algebra (informal)
------------------
- compose (⊗): cross-meaning composition producing a new tesseract.
- fuse    (⊕): weighted merge / denoising of close meanings.
- project (π): view / slice / masking of a tesseract's vector & facets.
- score: similarity to a query vector or another tesseract.

All operations return new instances; inputs remain immutable (dataclasses frozen).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from datetime import datetime, timezone
from hashlib import blake2s
import json
import uuid
import math

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore

ArrayLike = Union[' _np.ndarray', List[float], Tuple[float, ...], Any]


# ---------------------------- helpers ---------------------------------------

def _utc_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _to_array(x: Optional[ArrayLike]) -> Optional[ArrayLike]:
    if x is None:
        return None
    if _np is not None:
        if isinstance(x, _np.ndarray):
            return x.astype(_np.float64, copy=False)
        return _np.asarray(x, dtype=_np.float64)
    return list(x) if isinstance(x, (list, tuple)) else x


def _as_list(x: Optional[ArrayLike]) -> Optional[List[float]]:
    if x is None:
        return None
    if _np is not None:
        return _np.asarray(x, dtype=_np.float64).ravel().tolist()
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    # last resort
    try:
        return list(x)  # type: ignore
    except Exception:
        return None


def _outer(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    if _np is not None:
        return _np.outer(_to_array(a), _to_array(b)).ravel()
    # naive outer for lists
    if isinstance(a, list) and isinstance(b, list):
        return [ai * bj for ai in a for bj in b]
    # fallback: pair tuple
    return (a, b)


def _add(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    if _np is not None:
        return _to_array(x) + _to_array(y)
    if isinstance(x, list) and isinstance(y, list):
        return [xi + yi for xi, yi in zip(x, y)]
    return x  # non-vector fallback


def _scale(x: ArrayLike, w: float) -> ArrayLike:
    if _np is not None:
        return float(w) * _to_array(x)
    if isinstance(x, list):
        return [w * xi for xi in x]
    return x


def _cosine(a: ArrayLike, b: ArrayLike) -> float:
    if _np is not None:
        A = _to_array(a)
        B = _to_array(b)
        na = float(_np.linalg.norm(A))
        nb = float(_np.linalg.norm(B))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(_np.dot(A, B) / (na * nb))
    if isinstance(a, list) and isinstance(b, list):
        num = sum(x*y for x, y in zip(a, b))
        da = math.sqrt(sum(x*x for x in a)) or 1.0
        db = math.sqrt(sum(y*y for y in b)) or 1.0
        return float(num / (da * db))
    return 0.0


def _canonical_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(obj: Dict[str, Any]) -> str:
    data = _canonical_json(obj).encode("utf-8", errors="ignore")
    return blake2s(data, digest_size=16).hexdigest()


# ---------------------------- core data -------------------------------------

@dataclass(frozen=True)
class Provenance:
    source: str = "unknown"
    author: str = "system"
    ts: int = field(default_factory=_utc_ts)
    notes: str = ""


@dataclass(frozen=True)
class Tesseract:
    """Minimal semantic unit (t¤).

    Fields
    ------
    tid : str
        Stable identifier (UUID by default).
    vec : Optional[ArrayLike]
        Optional dense vector (can be None for symbolic-only units).
    content : Dict[str, Any]
        Symbolic payload (tokens, structured meaning, etc.).
    context : Dict[str, Any]
        Conditioning metadata (domain, locale, task, etc.).
    affect : Dict[str, float]
        Optional affective tags, e.g., {"valence": +0.2, "arousal": 0.7}.
    prov : Provenance
        Creation provenance.
    tags : Tuple[str, ...]
        Searchable tags for SMFS-QE / indexing.
    """

    tid: str = field(default_factory=lambda: str(uuid.uuid4()))
    vec: Optional[ArrayLike] = None
    content: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    affect: Dict[str, float] = field(default_factory=dict)
    prov: Provenance = field(default_factory=Provenance)
    tags: Tuple[str, ...] = ()

    # ---------------- algebra ----------------

    def compose(self, other: "Tesseract", *, tag: str = "⊗") -> "Tesseract":
        """Cross-meaning composition producing a new tesseract.

        Vector part: outer product (flattened) if both have vectors.
        Symbolic part: pairs (self, other) under 'compose' key.
        """
        new_vec: Optional[ArrayLike] = None
        if self.vec is not None and other.vec is not None:
            new_vec = _outer(_to_array(self.vec), _to_array(other.vec))
        new_content = {
            "compose": {
                "lhs": self.content,
                "rhs": other.content,
            }
        }
        new_ctx = {**self.context, **other.context, "op": "compose"}
        new_tags = tuple(sorted(set(self.tags + other.tags + (tag,))))
        return replace(
            self,
            tid=str(uuid.uuid4()),
            vec=new_vec,
            content=new_content,
            context=new_ctx,
            affect={},
            prov=Provenance(source="compose", author="system"),
            tags=new_tags,
        )

    def fuse(self, other: "Tesseract", *, alpha: float = 0.5, tag: str = "⊕") -> "Tesseract":
        """Weighted merge of two close meanings. alpha in [0,1]."""
        a = max(0.0, min(1.0, float(alpha)))
        new_vec: Optional[ArrayLike] = None
        if self.vec is not None and other.vec is not None:
            new_vec = _add(_scale(_to_array(self.vec), a), _scale(_to_array(other.vec), 1.0 - a))
        new_content = {**other.content, **self.content, "_fused": True}
        new_ctx = {**self.context, **other.context, "op": "fuse", "alpha": a}
        new_aff = {**other.affect, **self.affect}
        new_tags = tuple(sorted(set(self.tags + other.tags + (tag,))))
        return replace(
            self,
            tid=str(uuid.uuid4()),
            vec=new_vec,
            content=new_content,
            context=new_ctx,
            affect=new_aff,
            prov=Provenance(source="fuse", author="system"),
            tags=new_tags,
        )

    def project(self, *, keys: Optional[Iterable[str]] = None, dims: Optional[Iterable[int]] = None,
                tag: str = "π") -> "Tesseract":
        """Projection on a subset of content keys / vector dims."""
        proj_content = self.content if keys is None else {k: v for k, v in self.content.items() if k in set(keys)}
        proj_vec: Optional[ArrayLike] = None
        if dims is None:
            proj_vec = self.vec
        else:
            if self.vec is not None:
                if _np is not None and isinstance(self.vec, _np.ndarray):
                    proj_vec = _to_array(self.vec)[list(dims)]
                elif isinstance(self.vec, list):
                    idx = list(dims)
                    proj_vec = [self.vec[i] for i in idx if 0 <= i < len(self.vec)]
        new_ctx = {**self.context, "op": "project"}
        new_tags = tuple(sorted(set(self.tags + (tag,))))
        return replace(
            self,
            tid=str(uuid.uuid4()),
            vec=proj_vec,
            content=proj_content,
            context=new_ctx,
            prov=Provenance(source="project", author="system"),
            tags=new_tags,
        )

    # ---------------- scoring & io ----------------

    def score(self, other: Union["Tesseract", ArrayLike]) -> float:
        """Cosine similarity if vectors exist, else 0."""
        if isinstance(other, Tesseract):
            if self.vec is None or other.vec is None:
                return 0.0
            return _cosine(self.vec, other.vec)
        # array-like
        if self.vec is None:
            return 0.0
        return _cosine(self.vec, other)

    def to_inode(self) -> Dict[str, Any]:
        """Compact, stable dict for SMFS-QE (semantic inode-like)."""
        obj = {
            "tid": self.tid,
            "ts": self.prov.ts,
            "src": self.prov.source,
            "author": self.prov.author,
            "tags": list(self.tags),
            "ctx": self.context,
            "aff": self.affect,
            "content": self.content,
            "vec": _as_list(self.vec),  # optional
        }
        obj["digest"] = blake2s(_canonical_json(obj).encode("utf-8"), digest_size=16).hexdigest()
        return obj

    @staticmethod
    def from_text(text: str, *, dim: int = 8, seed: int = 13, tags: Tuple[str, ...] = ()) -> "Tesseract":
        """Small helper to bootstrap a text unit with a deterministic pseudo-vector."""
        rnd = _np.random.default_rng(seed) if _np is not None else None
        if _np is not None:
            vec = rnd.normal(0, 1.0, size=(dim,))  # deterministic given seed
        else:
            # simple hash-to-vector fallback
            h = blake2s(text.encode("utf-8"), digest_size=dim).digest()
            vec = [((b / 255.0) * 2.0 - 1.0) for b in h]
        return Tesseract(
            vec=_to_array(vec),
            content={"text": text},
            context={},
            affect={},
            prov=Provenance(source="from_text", author="system"),
            tags=tags,
        )


@dataclass(frozen=True)
class Link:
    src: str
    dst: str
    rel: str = "related_to"
    w: float = 1.0
    ts: int = field(default_factory=_utc_ts)


@dataclass
class TGraph:
    """Tiny graph container for tesseracts and links (for prototypes/tests)."""
    nodes: Dict[str, Tesseract] = field(default_factory=dict)
    edges: List[Link] = field(default_factory=list)

    def add(self, t: Tesseract) -> None:
        self.nodes[t.tid] = t

    def link(self, a: Tesseract, b: Tesseract, rel: str = "related_to", w: float = 1.0) -> Link:
        e = Link(src=a.tid, dst=b.tid, rel=rel, w=w)
        self.edges.append(e)
        return e

    def neighbors(self, tid: str, *, rel: Optional[str] = None) -> List[Tesseract]:
        out: List[Tesseract] = []
        for e in self.edges:
            if e.src == tid and (rel is None or e.rel == rel):
                if e.dst in self.nodes:
                    out.append(self.nodes[e.dst])
        return out

    def find_by_tag(self, tag: str) -> List[Tesseract]:
        return [t for t in self.nodes.values() if tag in t.tags]


# ---------------------------- demo ------------------------------------------

def _demo() -> None:  # pragma: no cover
    print("NumPy available:", _np is not None)
    a = Tesseract.from_text("quantum", dim=6, seed=1, tags=("physics",))
    b = Tesseract.from_text("entanglement", dim=6, seed=2, tags=("physics","theory"))
    c = a.compose(b)
    d = a.fuse(b, alpha=0.7)
    p = c.project(dims=[0,2,4])

    print("score(a,b) =", a.score(b))
    print("compose     =", c.to_inode())
    print("fuse        =", d.to_inode())
    print("project     =", p.to_inode())

    g = TGraph()
    g.add(a); g.add(b); g.add(c); g.add(d); g.add(p)
    g.link(a, b, rel="implies", w=0.8)
    g.link(c, d, rel="derived_from", w=0.6)
    print("neighbors(a):", [n.content.get("text") for n in g.neighbors(a.tid)])

if __name__ == "__main__":  # pragma: no cover
    _demo()
