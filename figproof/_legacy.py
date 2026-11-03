"""Where figproof's drawing and QA modules come from."""
from __future__ import annotations

import importlib
import os
import sys

QA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa")


def _load(name):
    if QA not in sys.path:
        sys.path.insert(0, QA)
    mod = importlib.import_module(name)
    where = os.path.dirname(os.path.abspath(mod.__file__))
    if os.path.realpath(where) != os.path.realpath(QA):
        raise ImportError(f"figproof: {name!r} resolved to {where}, not {QA}; another module of that name is earlier on sys.path")
    return mod


def _own(name):
    return importlib.import_module(f"figproof.{name}")


def panels():
    return _own("panels")


def glyphs():
    return _own("glyphs")


def molecules():
    return _own("molecules")


def figstyle():
    return _load("figstyle")


def figqa():
    return _load("figqa")
