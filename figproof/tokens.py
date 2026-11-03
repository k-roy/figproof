"""figproof.tokens — the ONE token file, read in named layers.

Layers: `neutral` (the hue-free ramp: text and structure), `identity` (MAGESTIC molecular identities), `role` (RNA layer A: argument
roles), `mark` (RNA layer B: splice / polya / mod), `ground` (backgrounds and tints), `cover` (title slides and cover art only).
"""
from __future__ import annotations

import functools
import json
import os

PATH = os.environ.get("HOUSEFIG_TOKENS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokens.json")
COLOUR_LAYERS = ("identity", "role", "mark")


@functools.lru_cache(maxsize=None)
def load(path: str = PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def layer(name: str) -> dict:
    """{token: hex} for one layer."""
    L = load()["layers"]
    if name not in L:
        raise KeyError(f"unknown layer {name!r}; layers are {sorted(L)}")
    return {k: v for k, v in L[name].items() if isinstance(v, str)}


def colour(ref: str) -> str:
    """Hex for 'layer.token' (for example 'role.focal', 'neutral.ink'). Refuses an unknown name instead of defaulting."""
    lay, _, tok = ref.partition(".")
    toks = layer(lay)
    if tok not in toks:
        raise KeyError(f"{ref!r}: {lay} has {sorted(toks)}")
    return toks[tok]


@functools.lru_cache(maxsize=None)
def index() -> dict:
    """{HEX: [(layer, token), ...]} across every layer that holds plain colours."""
    out = {}
    for lay in ("neutral", "identity", "role", "mark", "ground"):
        for tok, hx in layer(lay).items():
            out.setdefault(hx.upper(), []).append((lay, tok))
    return out


def meanings_of(hx: str) -> list:
    """Every (layer, token) a hex names; empty for a colour that is not a token."""
    return list(index().get(hx.upper(), []))


def never_together() -> list:
    ""
    cl = load()["cross_layer"]
    return [(r["hexes"][0].upper(), r["hexes"][1].upper(), r["pair"][0], r["pair"][1], r["worst_de"]) for r in cl["never_together"]]


def declarable(include_declared: bool = False) -> list:
    ""
    cl = load()["cross_layer"]
    return [(r["hexes"][0].upper(), r["hexes"][1].upper(), r["pair"][0], r["pair"][1], r["worst_de"]) for r in cl["declarable"]
            if include_declared or not r.get("declared")]


def typography() -> dict:
    return load()["typography"]


def mode_spec(mode: str) -> dict:
    M = load()["modes"]
    if mode not in M:
        raise KeyError(f"unknown mode {mode!r}; modes are {sorted(M)}")
    return M[mode]
