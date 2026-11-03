"""figqa.py -- detect (and where possible auto-fix) overlapping text / legends /
data / panels in matplotlib figures.

Core value: a deterministic overlap DETECTOR that walks rendered artists after
``fig.canvas.draw()`` (Agg backend), computes pixel bounding boxes via
``artist.get_window_extent(renderer)``, and reports every pair that overlaps
beyond a tolerance -- so figures stop shipping with overlaps that previously had
to be caught by eye (render -> crop -> inspect -> nudge).

Public API
----------
    find_overlaps(fig, include_data=True, tol_px=2, ignore=None) -> list[Overlap]
    report(overlaps) -> str
    adjust_labels(ax, texts=None, objects=None, ...)            # adjustText fixer
    place_legend_clear(ax, handles=None, labels=None, ...)      # legend placement
    apply_nature_style() -> dict                                # Nature/publication rcParams (canonical)
    nature_figsize(kind='single', h=1.0) -> (w_in, h_in)        # Nature column widths
    good_defaults() -> dict                                     # delegates to apply_nature_style()

CLI
---
    python figqa.py path/to/make_fig.py
        execs/imports the script, grabs the Figure(s), runs the detector,
        prints the report, exits non-zero if overlaps remain.

Dependencies
------------
    matplotlib, numpy (required) + adjustText (optional, ``pip install adjustText``).
    adjustText is soft-imported: the DETECTOR works without it; only the
    ``adjust_labels`` fixer needs it and it degrades gracefully (raises a clear
    error / no-ops) if absent.
"""

from __future__ import annotations

import importlib.util
import os
import runpy
import sys
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import matplotlib

if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
    try:
        matplotlib.use("Agg")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Annotation, Text

try:
    from adjustText import adjust_text as _adjust_text

    _HAVE_ADJUSTTEXT = True
except Exception:
    _adjust_text = None
    _HAVE_ADJUSTTEXT = False


@dataclass
class ArtistBox:
    """A rendered artist plus its pixel-space bounding box and identity."""

    artist: object
    kind: str
    label: str
    x0: float
    y0: float
    x1: float
    y1: float
    ax_id: str

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class Overlap:
    a: ArtistBox
    b: ArtistBox
    inter_px: float
    frac_smaller: float
    ax_id: str
    manual_fix: bool = False

    def as_dict(self) -> dict:
        return {
            "axes": self.ax_id,
            "a_kind": self.a.kind,
            "a_label": self.a.label,
            "a_bbox_px": [round(self.a.x0, 1), round(self.a.y0, 1), round(self.a.x1, 1), round(self.a.y1, 1)],
            "b_kind": self.b.kind,
            "b_label": self.b.label,
            "b_bbox_px": [round(self.b.x0, 1), round(self.b.y0, 1), round(self.b.x1, 1), round(self.b.y1, 1)],
            "intersection_px2": round(self.inter_px, 1),
            "frac_of_smaller": round(self.frac_smaller, 3),
            "manual_fix_needed": self.manual_fix,
        }


def _bbox_intersection(a: ArtistBox, b: ArtistBox) -> float:
    """Pixel-area of the intersection of two boxes (0 if disjoint)."""
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _ax_identifier(ax) -> str:
    if ax is None:
        return "<figure>"
    try:
        ss = ax.get_subplotspec()
        if ss is not None:
            r0, r1, c0, c1 = ss.rowspan.start, ss.rowspan.stop, ss.colspan.start, ss.colspan.stop
            return f"axes[r{r0}:{r1},c{c0}:{c1}]"
    except Exception:
        pass
    title = ""
    try:
        title = ax.get_title()
    except Exception:
        pass
    return f"axes({title!r})" if title else f"axes@{hex(id(ax))[-6:]}"


def _text_identity(t: Text) -> str:
    s = t.get_text()
    s = s.replace("\n", "\\n")
    if len(s) > 40:
        s = s[:37] + "..."
    return f"text:{s!r}"


def _safe_extent(artist, renderer):
    """Return (x0,y0,x1,y1) px or None if degenerate / unavailable."""
    try:
        bb = artist.get_window_extent(renderer=renderer)
    except TypeError:
        try:
            bb = artist.get_window_extent()
        except Exception:
            return None
    except Exception:
        return None
    if bb is None:
        return None
    x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
    if not all(np.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if (x1 - x0) <= 0.5 or (y1 - y0) <= 0.5:
        return None
    return x0, y0, x1, y1


def _in_view_tick_labels(axis) -> List[Text]:
    """Tick-label Texts matplotlib will ACTUALLY draw.

    Same bug class as the ``axison`` phantom-tick case handled in
    ``_collect_text_boxes``, one level down: a tick whose location falls outside
    the axis view interval keeps a Text artist reporting ``get_visible() ==
    True``, but ``Axis.draw`` filters it out and never renders it. Measuring
    those bboxes yields phantom overlaps -- e.g. a default ``ax.plot([0,1],
    [0,1])`` has ``xlim=(-0.05, 1.05)`` with locator ticks ``[-0.2 ... 1.2]``,
    so the undrawn ``-0.2`` label collides with the neighbouring panel's real
    tick labels. Falls back to the unfiltered list on any surprise.
    """
    try:
        lo, hi = sorted(axis.get_view_interval())
    except Exception:
        return list(axis.get_ticklabels())
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return list(axis.get_ticklabels())
    eps = (hi - lo) * 1e-9
    try:
        ticks = list(axis.get_major_ticks()) + list(axis.get_minor_ticks())
    except Exception:
        return list(axis.get_ticklabels())
    out: List[Text] = []
    for tk in ticks:
        try:
            loc = tk.get_loc()
        except Exception:
            continue
        if loc is None or not np.isfinite(loc) or loc < lo - eps or loc > hi + eps:
            continue
        for lab in (getattr(tk, "label1", None), getattr(tk, "label2", None)):
            if isinstance(lab, Text) and lab.get_visible():
                out.append(lab)
    return out


def _artist_px(artist, ax, pts, offsets=False):
    """Map an artist's OWN coordinates to pixels through the ARTIST's transform.

    🔴 B11 — the phantom-glyph-overlap bug. This function exists because four sites
    used ``ax.transData.transform(pts)`` on an artist's raw ``xdata``/``ydata``.
    That is correct only when the artist's transform *is* ``transData``.

    ``offsets=True`` selects a collection's OFFSET transform, which is a different
    transform from ``get_transform()`` and is the one matplotlib uses to place offsets.
    
    """
    tr = None
    try:
        if offsets:
            getter = (getattr(artist, "get_offset_transform", None)
                      or getattr(artist, "get_transOffset", None))
            tr = getter() if getter is not None else None
        else:
            tr = artist.get_transform()
    except Exception:
        tr = None
    if tr is None:
        tr = ax.transData
    return np.asarray(tr.transform(pts), dtype=float)


def _clip_ext_to_axes(ext, artist, ax, renderer):
    """Intersect a CLIPPED artist's extent with the axes' drawing region.

    ``get_window_extent`` reports an artist's geometric extent, not its drawn
    (clipped) extent. A ``barh`` starting at x=0 on a LOG x-axis transforms to
    x0 ~ -1.9e5 px -- a left-unbounded half-plane. matplotlib clips it to the
    axes, so nothing is drawn there, but pairing on the raw extent makes that
    bar "overlap" every artist to its left, including artists in OTHER panels.
    Only artists with ``clip_on`` are clipped; deliberately-outside artists
    (panel letters, annotations with ``clip_on=False``) are left untouched.
    Returns None if the artist is entirely outside its axes (nothing drawn).
    """
    if ext is None:
        return None
    try:
        if not artist.get_clip_on():
            return ext
    except Exception:
        return ext
    try:
        cb = artist.get_clip_box()
        if cb is None:
            cb = ax.get_window_extent(renderer)
        cx0, cy0, cx1, cy1 = cb.x0, cb.y0, cb.x1, cb.y1
    except Exception:
        return ext
    if not all(np.isfinite(v) for v in (cx0, cy0, cx1, cy1)):
        return ext
    x0, y0, x1, y1 = ext
    x0, x1 = max(x0, min(cx0, cx1)), min(x1, max(cx0, cx1))
    y0, y1 = max(y0, min(cy0, cy1)), min(y1, max(cy0, cy1))
    if (x1 - x0) <= 0.5 or (y1 - y0) <= 0.5:
        return None
    return x0, y0, x1, y1


def _text_is_meaningful(t: Text) -> bool:
    if not t.get_visible():
        return False
    try:
        s = t.get_text()
    except Exception:
        return False
    if s is None or str(s).strip() == "":
        return False
    return True


def _collect_text_boxes(ax, renderer, ax_id) -> List[ArtistBox]:
    boxes: List[ArtistBox] = []
    texts: List[Text] = []
    for attr in ("title", "_left_title", "_right_title"):
        t = getattr(ax, attr, None)
        if isinstance(t, Text):
            texts.append(t)
    if getattr(ax, "axison", True):
        for axis in (ax.xaxis, ax.yaxis):
            if not axis.get_visible():
                continue
            lbl = axis.get_label()
            if isinstance(lbl, Text):
                texts.append(lbl)
            for tick_text in _in_view_tick_labels(axis):
                texts.append(tick_text)
            ot = getattr(axis, "offsetText", None)
            if isinstance(ot, Text):
                texts.append(ot)
    for child in ax.texts:
        if isinstance(child, Text):
            texts.append(child)

    for t in texts:
        if not _text_is_meaningful(t):
            continue
        ext = _safe_extent(t, renderer)
        if ext is None:
            continue
        kind = "annotation" if isinstance(t, Annotation) else "text"
        boxes.append(ArtistBox(t, kind, _text_identity(t), *ext, ax_id=ax_id))
    return boxes


def _collect_legend_boxes(ax, renderer, ax_id) -> List[ArtistBox]:
    boxes = []
    legends = []
    lg = ax.get_legend()
    if lg is not None:
        legends.append(lg)
    for child in getattr(ax, "artists", []):
        if isinstance(child, Legend) and child not in legends:
            legends.append(child)
    for lg in legends:
        if not lg.get_visible():
            continue
        ext = _safe_extent(lg, renderer)
        if ext is None:
            continue
        boxes.append(ArtistBox(lg, "legend", "legend", *ext, ax_id=ax_id))
    return boxes


def _collect_data_boxes(ax, renderer, ax_id) -> List[ArtistBox]:
    """Data artists. Tight geometry where cheap (scatter offsets, patch paths);
    Line2D is coarse (whole data span) -> documented limitation."""
    boxes = []
    for coll in ax.collections:
        if not coll.get_visible():
            continue
        if isinstance(coll, PathCollection):
            try:
                offsets = coll.get_offsets()
                if offsets is not None and len(offsets) > 0:
                    pix = _artist_px(coll, ax, np.asarray(offsets), offsets=True)
                    pix = pix[np.isfinite(pix).all(axis=1)]
                    if len(pix) > 0:
                        x0, y0 = pix[:, 0].min(), pix[:, 1].min()
                        x1, y1 = pix[:, 0].max(), pix[:, 1].max()
                        pad = 4.0
                        boxes.append(
                            ArtistBox(coll, "collection", f"scatter(n={len(pix)})",
                                      x0 - pad, y0 - pad, x1 + pad, y1 + pad, ax_id=ax_id))
                        continue
            except Exception:
                pass
        ext = _clip_ext_to_axes(_safe_extent(coll, renderer), coll, ax, renderer)
        if ext is not None:
            boxes.append(ArtistBox(coll, "collection", "collection", *ext, ax_id=ax_id))

    for p in ax.patches:
        if not p.get_visible():
            continue
        if _effective_patch_alpha(p) <= 0.0:
            try:
                import matplotlib.colors as _mc
                _ec = p.get_edgecolor()
                if _ec is None or float(_mc.to_rgba(_ec)[3]) <= 0.0 or float(p.get_linewidth()) <= 0:
                    continue
            except Exception:
                continue
        ext = _clip_ext_to_axes(_safe_extent(p, renderer), p, ax, renderer)
        if ext is None:
            continue
        boxes.append(ArtistBox(p, "patch", f"patch:{type(p).__name__}", *ext, ax_id=ax_id))

    for ln in ax.lines:
        if not ln.get_visible():
            continue
        try:
            xdata = ln.get_xdata()
            ydata = ln.get_ydata()
            if xdata is None or len(xdata) == 0:
                continue
            pts = np.column_stack([np.asarray(xdata, float), np.asarray(ydata, float)])
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) == 0:
                continue
            pix = _artist_px(ln, ax, pts)
            pix = pix[np.isfinite(pix).all(axis=1)]
            if len(pix) == 0:
                continue
            x0, y0 = pix[:, 0].min(), pix[:, 1].min()
            x1, y1 = pix[:, 0].max(), pix[:, 1].max()
            if (x1 - x0) <= 0.5 and (y1 - y0) <= 0.5:
                continue
            ext = _clip_ext_to_axes((x0, y0, x1, y1), ln, ax, renderer)
            if ext is None:
                continue
            boxes.append(ArtistBox(ln, "line", f"line(n={len(pix)})", *ext, ax_id=ax_id))
        except Exception:
            continue
    return boxes


def _default_should_ignore(a: ArtistBox, b: ArtistBox) -> bool:
    """Conservative suppression of intended overlaps. False positives are
    acceptable per spec; silent misses are not -- so only suppress the clearly
    intended 'label fully inside a contrasting patch' case."""
    text_box = patch_box = None
    if a.kind in ("text", "annotation") and b.kind == "patch":
        text_box, patch_box = a, b
    elif b.kind in ("text", "annotation") and a.kind == "patch":
        text_box, patch_box = b, a
    if text_box is None:
        return False
    contained = (
        patch_box.x0 - 1 <= text_box.x0 and text_box.x1 <= patch_box.x1 + 1
        and patch_box.y0 - 1 <= text_box.y0 and text_box.y1 <= patch_box.y1 + 1
    )
    if not contained:
        return False
    try:
        import matplotlib.colors as mcolors

        tcol = np.array(mcolors.to_rgb(text_box.artist.get_color()))
        face = patch_box.artist.get_facecolor()
        if face is None:
            return False
        if hasattr(face, "__len__") and len(face) and hasattr(face[0], "__len__"):
            face = face[0]
        fcol = np.array(mcolors.to_rgb(face))

        alpha = _effective_patch_alpha(patch_box.artist)
        bg = np.array([1.0, 1.0, 1.0])
        try:
            _ax = getattr(patch_box.artist, "axes", None)
            _src = _ax if _ax is not None else patch_box.artist.get_figure()
            if _src is not None:
                bg = np.array(mcolors.to_rgb(_src.get_facecolor()))
        except Exception:
            pass
        fcol = alpha * fcol + (1.0 - alpha) * bg

        def lum(c):
            return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]

        return abs(lum(tcol) - lum(fcol)) > 0.25
    except Exception:
        return False


_DATA_KINDS = {"line", "collection", "patch"}


def _pair_relevant(a: ArtistBox, b: ArtistBox, include_data: bool) -> bool:
    ka, kb = a.kind, b.kind
    text_kinds = {"text", "annotation"}
    if ka in text_kinds and kb in text_kinds:
        return True
    if "legend" in (ka, kb):
        other = kb if ka == "legend" else ka
        if other in text_kinds or other == "legend":
            return True
        return include_data
    if not include_data:
        return False
    if (ka in text_kinds and kb in _DATA_KINDS) or (kb in text_kinds and ka in _DATA_KINDS):
        return True
    return False


def _is_movable(box: ArtistBox) -> bool:
    """Whether figqa's fixers could in principle relocate this artist.

    Legends -> place_legend_clear. Plain Text -> adjust_labels. Annotations are
    treated as NOT auto-movable: they are anchored to a data point (xy) and often
    carry an arrow, so relocating them silently breaks the schematic's meaning.
    Annotation-vs-annotation collisions are exactly the 'manual fix needed' class.
    """
    return box.kind in ("legend", "text")


def find_overlaps(
    fig,
    include_data: bool = True,
    tol_px: float = 2.0,
    frac_tol: float = 0.05,
    ignore: Optional[Callable[[ArtistBox, ArtistBox], bool]] = None,
) -> List[Overlap]:
    """Render ``fig`` (Agg) and return every artist pair overlapping beyond
    tolerance.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    include_data : bool
        Include data artists (scatter/patch/line) so text-over-data is caught.
    tol_px : float
        An overlap counts only if the intersection exceeds ``tol_px`` pixels on
        BOTH axes (rejects hairline edge touches).
    frac_tol : float
        Minimum fraction of the smaller box that must be covered.
    ignore : callable(a, b) -> bool
        Predicate to suppress intended overlaps. Defaults to a conservative
        label-inside-contrasting-patch suppressor. Pass ``ignore=lambda a,b: False``
        to disable all suppression.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if ignore is None:
        ignore = _default_should_ignore

    all_boxes: List[ArtistBox] = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        ax_id = _ax_identifier(ax)
        all_boxes += _collect_text_boxes(ax, renderer, ax_id)
        all_boxes += _collect_legend_boxes(ax, renderer, ax_id)
        if include_data:
            all_boxes += _collect_data_boxes(ax, renderer, ax_id)
    for t in getattr(fig, "texts", []):
        if isinstance(t, Text) and _text_is_meaningful(t):
            ext = _safe_extent(t, renderer)
            if ext is not None:
                all_boxes.append(ArtistBox(t, "text", _text_identity(t), *ext, ax_id="<figure>"))
    for lg in getattr(fig, "legends", []):
        if isinstance(lg, Legend) and lg.get_visible():
            ext = _safe_extent(lg, renderer)
            if ext is not None:
                all_boxes.append(ArtistBox(lg, "legend", "legend", *ext, ax_id="<figure>"))

    overlaps: List[Overlap] = []
    n = len(all_boxes)
    min_side = float(tol_px)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = all_boxes[i], all_boxes[j]
            if a.artist is b.artist:
                continue
            if not _pair_relevant(a, b, include_data):
                continue
            if ignore(a, b):
                continue
            inter = _bbox_intersection(a, b)
            if inter <= 0:
                continue
            ix = min(a.x1, b.x1) - max(a.x0, b.x0)
            iy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ix <= min_side or iy <= min_side:
                continue
            smaller_area = min(a.area, b.area)
            frac = inter / smaller_area if smaller_area > 0 else 0.0
            if frac < frac_tol:
                continue
            cross = a.ax_id != b.ax_id
            ax_ctx = "cross-axes(%s|%s)" % (a.ax_id, b.ax_id) if cross else a.ax_id
            manual = not (_is_movable(a) or _is_movable(b))
            overlaps.append(Overlap(a, b, inter, frac, ax_ctx, manual_fix=manual))
    overlaps.sort(key=lambda o: o.inter_px, reverse=True)
    return overlaps


def report(overlaps: Sequence[Overlap]) -> str:
    if not overlaps:
        return "figproof.qa: OK -- no overlaps detected."
    lines = [f"figproof.qa: {len(overlaps)} overlap(s) detected.", ""]
    by_ctx = {}
    for o in overlaps:
        by_ctx.setdefault(o.ax_id, []).append(o)
    for ctx, group in by_ctx.items():
        lines.append(f"== {ctx} == ({len(group)} overlap(s))")
        for o in group:
            tag = "  [MANUAL FIX NEEDED]" if o.manual_fix else ""
            lines.append(
                f"  {o.a.kind} {o.a.label}  <->  {o.b.kind} {o.b.label}{tag}"
            )
            lines.append(
                f"      inter={o.inter_px:.0f}px^2  frac_of_smaller={o.frac_smaller:.0%}  "
                f"A_px=[{o.a.x0:.0f},{o.a.y0:.0f},{o.a.x1:.0f},{o.a.y1:.0f}]  "
                f"B_px=[{o.b.x0:.0f},{o.b.y0:.0f},{o.b.x1:.0f},{o.b.y1:.0f}]"
            )
        lines.append("")
    n_manual = sum(1 for o in overlaps if o.manual_fix)
    if n_manual:
        lines.append(
            f"Note: {n_manual} overlap(s) flagged MANUAL FIX NEEDED -- both artists are "
            f"data/schematic and cannot be auto-relocated; adjust the figure script by hand."
        )
    return "\n".join(lines).rstrip()


NATURE_CYCLE_ORDER = ["blue", "orange", "bluish_green", "vermillion",
                      "sky_blue", "reddish_purple", "yellow", "black"]

NATURE_COLUMN_WIDTHS_IN = {
    "single": 3.50, "1col": 3.50, "1": 3.50,
    "onehalf": 4.80, "1.5col": 4.80, "1.5": 4.80,
    "double": 7.20, "2col": 7.20, "2": 7.20,
}


def apply_nature_style():
    """Apply a Nature/publication style to ``matplotlib.rcParams`` and
    return the applied dict. **This is the canonical style to call for every new
    figure.**

    What it sets (a compact, print-ready style):

    - **Type:** sans-serif at a 7 pt base (Arial -> Helvetica -> Helvetica Neue
      -> DejaVu Sans), with a small type scale (title 8, label 7, ticks 6,
      legend 6 / legend-title 6.5).
    - **Axes/spines:** 0.5 pt axes, ``0.15`` edge/tick colour, only the LEFT and
      BOTTOM spines (top/right off), outward ticks (major 2.5/0.5, minor
      1.5/0.4).
    - **Lines/markers/patches:** 1.0 pt lines, markersize 4, 0.5 pt marker edge
      and patch edge (all >= the 0.5 pt ``line_width`` policy floor).
    - **Grid:** OFF by default, but pre-styled (0.4 pt, ``0.9`` grey, alpha 0.6)
      for scripts that opt in via ``ax.grid(True)``.
    - **Legend:** frameless, tight handle/label spacing.
    - **Vector/editable text:** TrueType-42 in PDF/PS and ``svg.fonttype='none'``
      so text stays editable in Illustrator/Inkscape.
    - **Figure/savefig:** white backgrounds, ``figure.dpi=150``,
      ``savefig.dpi=400``, ``bbox='tight'``, ``constrained_layout`` on.
    - **Colour cycle:** the Okabe-Ito colour-blind-safe palette, reordered so a
      saturated colour (blue) leads and black is last -- the SAME hex values the
      ``palette_compliance`` policy check treats as approved.

    Font guard: if none of Arial/Helvetica/Helvetica Neue are installed this
    WARNs and relies on the DejaVu Sans fallback (matplotlib always ships it).
    It NEVER hard-fails.

    Returns
    -------
    dict
        The ``rcParams`` key -> value mapping that was applied (with
        ``axes.prop_cycle`` as a ``cycler`` object), so callers can inspect it or
        restore afterwards (e.g. inside ``matplotlib.rc_context``).
    """
    from cycler import cycler

    cycle_hex = [OKABE_ITO[k] for k in NATURE_CYCLE_ORDER]

    sans_stack = ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"]
    try:
        from matplotlib import font_manager

        installed = {f.name for f in font_manager.fontManager.ttflist}
        if not (installed & {"Arial", "Helvetica", "Helvetica Neue"}):
            warnings.warn(
                "apply_nature_style: Arial/Helvetica/Helvetica Neue not installed; "
                "falling back to DejaVu Sans. Install Arial for the exact Nature "
                "default font.", stacklevel=2)
    except Exception:
        pass

    nature = {
        "font.family": "sans-serif",
        "font.sans-serif": sans_stack,
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 6.5,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "0.15",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.color": "0.15",
        "ytick.color": "0.15",
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "lines.markeredgewidth": 0.5,
        "patch.linewidth": 0.5,
        "axes.grid": False,
        "grid.linewidth": 0.4,
        "grid.color": "0.9",
        "grid.alpha": 0.6,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.constrained_layout.use": True,
        "axes.prop_cycle": cycler(color=cycle_hex),
    }
    matplotlib.rcParams.update(nature)
    return dict(nature)


def nature_figsize(kind="single", h=1.0):
    """Return a ``(width_in, height_in)`` figure size at a Nature column width.

    Widths (Nature Research): ``'single'``/``'1col'`` = 3.50 in (89 mm),
    ``'onehalf'``/``'1.5col'`` = 4.80 in, ``'double'``/``'2col'`` = 7.20 in
    (183 mm).

    Examples
    --------
    >>> nature_figsize('single')          # (3.5, 2.16)  golden single column
    >>> nature_figsize('double', h=0.5)   # (7.2, 2.22)  wide short banner
    
    """
    w = NATURE_COLUMN_WIDTHS_IN.get(str(kind).lower())
    if w is None:
        raise ValueError(
            f"nature_figsize: unknown kind {kind!r}; use 'single'/'onehalf'/'double' "
            f"(aliases '1col'/'1.5col'/'2col').")
    golden = 0.6180339887498949
    return (w, round(w * golden * float(h), 4))


def good_defaults():
    """Apply the Nature/publication style and return the applied rcParams.

    **Backward-compatible upgrade (delegates to ``apply_nature_style()``).**
    Historically this set only ``constrained_layout`` + a slightly-smaller font;
    it now delegates to :func:`apply_nature_style` so that EVERY existing caller
    of ``good_defaults()`` transparently gets the full publication style
    (compact 7 pt type, thin 0.5 pt left/bottom-spine axes, Okabe-Ito
    ``prop_cycle``, editable vector text, white backgrounds, constrained layout).

    The return contract is unchanged: it returns the dict of rcParams that was
    applied, so callers can inspect or restore it. New code should prefer calling
    :func:`apply_nature_style` (and :func:`nature_figsize`) directly.
    """
    return apply_nature_style()


def adjust_labels(ax, texts=None, objects=None, **kwargs):
    """Repel text labels off each other and off data using adjustText.

    Soft-degrades: if adjustText is not installed, raises a clear RuntimeError
    (the DETECTOR does not need adjustText; only this fixer does).

    Parameters
    ----------
    ax : Axes
    texts : list[Text] or None
        Texts to move; defaults to ``ax.texts``.
    objects : list of artists to repel FROM (e.g. legend, scatter) or None.
    **kwargs : forwarded to adjustText.adjust_text (e.g. arrowprops).
    """
    if not _HAVE_ADJUSTTEXT:
        raise RuntimeError(
            "adjust_labels requires adjustText (pip install adjustText). "
            "The detector works without it; only the fixer needs it."
        )
    if texts is None:
        texts = list(ax.texts)
    texts = [t for t in texts if _text_is_meaningful(t)]
    if not texts:
        return 0
    fig = ax.get_figure()
    fig.canvas.draw()
    kw = dict(
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5),
    )
    if objects is not None:
        kw["objects"] = objects
    kw.update(kwargs)
    try:
        return _adjust_text(texts, **kw)
    except ValueError as e:
        if "finite" in str(e) and "objects" in kw:
            kw.pop("objects", None)
            return _adjust_text(texts, **kw)
        raise


def place_legend_clear(
    ax,
    handles=None,
    labels=None,
    candidates=None,
    include_data=True,
    **legend_kwargs,
):
    """Place a legend at the candidate location with least overlap.

    Tries inside corners AND outside-axes anchors, renders each, scores total
    overlap (legend vs text + data) via the detector, and keeps the minimum.

    Returns the chosen ``Legend``.
    """
    if candidates is None:
        candidates = [
            dict(loc="upper right"),
            dict(loc="upper left"),
            dict(loc="lower left"),
            dict(loc="lower right"),
            dict(loc="center left", bbox_to_anchor=(1.02, 0.5)),
            dict(loc="upper left", bbox_to_anchor=(1.02, 1.0)),
            dict(loc="lower center", bbox_to_anchor=(0.5, -0.28)),
            dict(loc="upper center", bbox_to_anchor=(0.5, 1.18)),
        ]
    fig = ax.get_figure()

    def _make(loc_kw):
        old = ax.get_legend()
        if old is not None:
            old.remove()
        kw = dict(legend_kwargs)
        kw.update(loc_kw)
        if handles is not None:
            if labels is not None:
                lg = ax.legend(handles, labels, **kw)
            else:
                lg = ax.legend(handles=handles, **kw)
        else:
            lg = ax.legend(**kw)
        return lg

    best = None
    best_score = None
    best_kw = None
    for loc_kw in candidates:
        lg = _make(loc_kw)
        if lg is None:
            continue
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ext = _safe_extent(lg, renderer)
        if ext is None:
            continue
        lbox = ArtistBox(lg, "legend", "legend", *ext, ax_id=_ax_identifier(ax))
        ax_id = _ax_identifier(ax)
        targets = _collect_text_boxes(ax, renderer, ax_id)
        if include_data:
            targets += _collect_data_boxes(ax, renderer, ax_id)
        score = 0.0
        for tb in targets:
            score += _bbox_intersection(lbox, tb)
        outside = "bbox_to_anchor" in loc_kw
        if best_score is None or score < best_score - 1e-6 or (
            abs(score - (best_score or 0)) < 1e-6 and not outside and best_kw and "bbox_to_anchor" in best_kw
        ):
            best, best_score, best_kw = lg, score, loc_kw
    if best_kw is not None:
        best = _make(best_kw)
    return best


def _load_figures_from_script(path: str):
    """Exec a .py figure script and return the list of live Figures.

    Strategy:
      1. If the module defines ``build()`` returning a Figure, use that.
      2. Otherwise run the module top-level and collect plt.get_fignums().
    Patches plt.show(), plt.savefig() and Figure.savefig() to no-ops so the script doesn't block or
    write, and guards against plt.close() destroying the figure before we capture it.
    """
    from matplotlib.figure import Figure as _Figure
    abspath = os.path.abspath(path)
    _orig_show, _orig_close, _orig_savefig = plt.show, plt.close, plt.savefig
    plt.show = lambda *a, **k: None
    plt.savefig = lambda *a, **k: None
    plt.close = lambda *a, **k: None
    _orig_fig_savefig = _Figure.savefig
    _Figure.savefig = lambda self, *a, **k: None

    figs = []
    try:
        spec = importlib.util.spec_from_file_location("_figqa_target", abspath)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_figqa_target"] = mod
            try:
                spec.loader.exec_module(mod)
            except SystemExit:
                pass
            build = getattr(mod, "build", None)
            if callable(build):
                result = build()
                if result is not None and hasattr(result, "axes"):
                    figs = [result]
            if not figs:
                figs = [plt.figure(n) for n in plt.get_fignums()]
        if not figs:
            runpy.run_path(abspath, run_name="__main__")
            figs = [plt.figure(n) for n in plt.get_fignums()]
    finally:
        plt.show, plt.close, plt.savefig = _orig_show, _orig_close, _orig_savefig
        _Figure.savefig = _orig_fig_savefig
    return figs


OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

MIN_LINEWIDTH_PT = 0.5

GRAYSCALE_DELTAE_MIN = 20.0


def _policy_record(status, summary, **extra):
    rec = {"status": status, "summary": summary}
    rec.update(extra)
    return rec


def _safe_check(fn, *args, **kwargs):
    """Run a check function; degrade ANY internal exception to a SKIP record so a
    check can never raise into the aggregated report."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return _policy_record("SKIP", f"check error: {type(e).__name__}: {e}",
                              detail=[], suggestions=["(internal error; check skipped)"])


def _to_rgb01(color):
    import matplotlib.colors as mcolors
    return tuple(float(c) for c in mcolors.to_rgb(color))


def _srgb_to_lab(rgb):
    """sRGB (0..1) -> CIELAB (D65). Standard, deterministic, dep-free."""
    def _lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (_lin(c) for c in rgb)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x, y, z = x / xn, y / yn, z / zn

    def _f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fx, fy, fz = _f(x), _f(y), _f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    bb = 200 * (fy - fz)
    return (L, a, bb)


def _delta_e76(lab1, lab2):
    return float(np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(lab1, lab2))))


def _nearest_okabe(rgb):
    """Return (name, hex, deltaE) of the nearest Okabe-Ito color in CIELAB."""
    lab = _srgb_to_lab(rgb)
    best = None
    for name, hx in OKABE_ITO.items():
        d = _delta_e76(lab, _srgb_to_lab(_to_rgb01(hx)))
        if best is None or d < best[2]:
            best = (name, hx, d)
    return best


import re as _re

_PANEL_LETTER_RE = _re.compile(r"^[A-Za-z][\)\.]?$")


def check_panel_letters(fig):
    """Find short panel letters (a, b, A), 'a)' ...) near each axes' top-left.

    Reports per-axes whether a letter is present and where it sits relative to
    the axes top-left margin. WARN if a multi-panel figure has axes lacking a
    letter, or a letter overlaps panel content."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = [ax for ax in fig.axes if ax.get_visible()]
    if len(axes) < 2:
        return _policy_record("PASS", "single-panel figure; panel letters not required",
                              detail=[], suggestions=[])

    candidates = []
    for t in getattr(fig, "texts", []):
        if isinstance(t, Text) and _text_is_meaningful(t) and _PANEL_LETTER_RE.match(t.get_text().strip()):
            candidates.append((t, None))
    for ax in axes:
        for t in ax.texts:
            if isinstance(t, Text) and _text_is_meaningful(t) and _PANEL_LETTER_RE.match(t.get_text().strip()):
                candidates.append((t, ax))

    detail = []
    missing = []
    overlapping = []
    for ax in axes:
        ax_id = _ax_identifier(ax)
        try:
            abb = ax.get_window_extent(renderer=renderer)
        except Exception:
            continue
        tlx, tly = abb.x0, abb.y1
        found = None
        for t, owner in candidates:
            ext = _safe_extent(t, renderer)
            if ext is None:
                continue
            cx = 0.5 * (ext[0] + ext[2])
            cy = 0.5 * (ext[1] + ext[3])
            near_x = (tlx - 0.30 * abb.width) <= cx <= (tlx + 0.30 * abb.width)
            near_y = (tly - 0.18 * abb.height) <= cy <= (tly + 0.22 * abb.height)
            if near_x and near_y:
                found = (t, ext)
                break
        if found is None:
            missing.append(ax_id)
            detail.append(f"{ax_id}: NO panel letter near top-left")
        else:
            t, ext = found
            letter = t.get_text().strip()
            in_content = (abb.x0 + 2 <= ext[0] and ext[2] <= abb.x1 - 2
                          and abb.y0 + 2 <= ext[1] and ext[3] <= abb.y1 - 2)
            place = "inside-content (crowds panel)" if in_content else "margin (good)"
            if in_content:
                overlapping.append(ax_id)
            detail.append(f"{ax_id}: letter {letter!r} present, {place}")

    suggestions = []
    if missing:
        suggestions.append(
            "Add a bold panel letter (a, b, c... margin top-left, clip_on=False) to: "
            + ", ".join(missing))
    if overlapping:
        suggestions.append(
            "Move panel letter into the MARGIN (e.g. ax.text(-0.08, 1.05, 'a', "
            "transform=ax.transAxes, fontweight='bold', clip_on=False)) for: "
            + ", ".join(overlapping))
    status = "WARN" if (missing or overlapping) else "PASS"
    summary = (f"{len(axes)} panels; {len(missing)} missing letter, "
               f"{len(overlapping)} crowding content")
    return _policy_record(status, summary, detail=detail, suggestions=suggestions)


def _font_cmap_chars(font_path):
    """Return the set of unicode codepoints (ints) covered by a font file, or
    None if it cannot be parsed. Handles .ttc via fontNumber=0."""
    from fontTools.ttLib import TTFont, TTLibError

    try:
        try:
            tt = TTFont(font_path, fontNumber=0, lazy=True)
        except TTLibError:
            tt = TTFont(font_path, lazy=True)
        cmap = tt.getBestCmap()
        chars = set(cmap.keys()) if cmap else set()
        tt.close()
        return chars
    except Exception:
        return None


_MATHTEXT_RE = _re.compile(r"\$[^$]*\$")


def check_glyph_coverage(fig):
    """For every text artist, verify each (non-mathtext, non-whitespace) char is
    in the resolved font family's cmap chain. WARN on missing glyphs (mojibake
    risk: →, ×, δ, ′, ≈, CJK, emoji ...)."""
    try:
        from matplotlib import font_manager
    except Exception:
        return _policy_record("SKIP", "matplotlib.font_manager unavailable",
                              detail=[], suggestions=[])

    fig.canvas.draw()
    texts = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for t in _iter_all_axes_text(ax):
            texts.append(t)
    for t in getattr(fig, "texts", []):
        if isinstance(t, Text):
            texts.append(t)

    cmap_cache = {}

    def _covered(ch, fp):
        """Is char `ch` covered by ANY font in the family fallback chain?"""
        families = fp.get_family()
        if isinstance(families, str):
            families = [families]
        try:
            paths = []
            for fam in families:
                fp2 = fp.copy()
                fp2.set_family(fam)
                paths.append(font_manager.findfont(fp2, fallback_to_default=True))
            paths.append(font_manager.findfont(fp, fallback_to_default=True))
        except Exception:
            return True
        cp = ord(ch)
        any_resolved = False
        for path in paths:
            if path not in cmap_cache:
                cmap_cache[path] = _font_cmap_chars(path)
            chars = cmap_cache[path]
            if chars is None:
                continue
            any_resolved = True
            if cp in chars:
                return True
        return not any_resolved

    detail = []
    missing_total = 0
    for t in texts:
        if not _text_is_meaningful(t):
            continue
        s = t.get_text()
        s = _MATHTEXT_RE.sub("", s)
        fp = t.get_fontproperties()
        missing = []
        for ch in s:
            if ch.isspace() or ch in "\n\t":
                continue
            if not _covered(ch, fp):
                if ch not in missing:
                    missing.append(ch)
        if missing:
            missing_total += len(missing)
            disp = " ".join(f"{c!r}(U+{ord(c):04X})" for c in missing)
            detail.append(f"{_text_identity(t)}: missing {disp}")

    if missing_total:
        return _policy_record(
            "WARN", f"{missing_total} char(s) not in resolved font cmap (mojibake risk)",
            detail=detail,
            suggestions=["Use a fallback font that has the glyph (e.g. DejaVu Sans), "
                         "or draw the glyph; set rcParams['pdf.fonttype']=42 for vector embed."])
    return _policy_record("PASS", "all text glyphs covered by resolved font(s)",
                          detail=[], suggestions=[])


def _iter_all_axes_text(ax):
    """Yield all Text artists on an axes (titles, axis/tick labels, free text)."""
    for attr in ("title", "_left_title", "_right_title"):
        t = getattr(ax, attr, None)
        if isinstance(t, Text):
            yield t
    if getattr(ax, "axison", True):
        for axis in (ax.xaxis, ax.yaxis):
            if not axis.get_visible():
                continue
            lbl = axis.get_label()
            if isinstance(lbl, Text):
                yield lbl
            for tk in axis.get_ticklabels():
                yield tk
    for t in ax.texts:
        if isinstance(t, Text):
            yield t
    lg = ax.get_legend()
    if lg is not None:
        for t in lg.get_texts():
            yield t


def check_provenance_footer(fig):
    """Detect a small grey fig.text near the bottom of the figure (the required
    provenance footer). WARN if absent."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    import matplotlib.colors as mcolors

    fh = fig.bbox.height
    candidates = []
    for t in getattr(fig, "texts", []):
        if not isinstance(t, Text) or not _text_is_meaningful(t):
            continue
        ext = _safe_extent(t, renderer)
        if ext is None:
            continue
        cy = 0.5 * (ext[1] + ext[3])
        if cy > 0.18 * fh:
            continue
        try:
            fs = t.get_fontsize()
        except Exception:
            fs = 10
        try:
            r, g, b = mcolors.to_rgb(t.get_color())
            is_grey = (max(r, g, b) - min(r, g, b) < 0.12) and (0.25 < (r + g + b) / 3 < 0.75)
        except Exception:
            is_grey = False
        candidates.append((t, fs, is_grey))

    strong = [c for c in candidates if c[1] <= 7.5 and c[2]]
    if strong:
        return _policy_record("PASS", "provenance footer detected (small grey bottom text)",
                              detail=[f"{_text_identity(c[0])} ({c[1]:.1f}pt)" for c in strong],
                              suggestions=[])
    if candidates:
        return _policy_record(
            "WARN", "bottom text present but not styled as a provenance footer",
            detail=[f"{_text_identity(c[0])} ({c[1]:.1f}pt, grey={c[2]})" for c in candidates],
            suggestions=["Style the footer small+grey: fig.text(0.5, 0.01, '<source · script · date>', "
                         "ha='center', fontsize=5.5, color='0.45')"])
    return _policy_record(
        "WARN", "no provenance footer found",
        detail=[],
        suggestions=["Add a provenance footer: fig.text(0.5, 0.01, '<data source/accession · "
                     "processing · script.py · '+datetime.date.today().isoformat()+'>', "
                     "ha='center', fontsize=5.5, color='0.45')"])


def _artist_color(artist):
    """Best-effort representative color of an artist as an rgb tuple, or None."""
    import matplotlib.colors as mcolors
    for getter in ("get_color", "get_facecolor", "get_edgecolor"):
        fn = getattr(artist, getter, None)
        if fn is None:
            continue
        try:
            c = fn()
        except Exception:
            continue
        if c is None:
            continue
        try:
            if isinstance(c, str):
                first = c
            elif hasattr(c, "__len__") and len(c) and hasattr(c[0], "__len__"):
                first = c[0]
            else:
                first = c
            rgb = mcolors.to_rgb(first)
            return rgb
        except Exception:
            try:
                rgb = mcolors.to_rgb(c)
                return rgb
            except Exception:
                continue
    return None


def check_cross_panel_color(fig):
    """Same legend-entry / label TEXT mapped to DIFFERENT colors across axes ->
    WARN. Emits a per-panel semantic->color table for author review."""
    import matplotlib.colors as mcolors

    fig.canvas.draw()
    sem = {}
    table = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        ax_id = _ax_identifier(ax)
        lg = ax.get_legend()
        if lg is None:
            continue
        labels = [t.get_text() for t in lg.get_texts()]
        handles = lg.legend_handles if hasattr(lg, "legend_handles") else getattr(lg, "legendHandles", [])
        for lbl, h in zip(labels, handles):
            if not lbl or lbl.startswith("_"):
                continue
            rgb = _artist_color(h)
            if rgb is None:
                continue
            hx = mcolors.to_hex(rgb)
            sem.setdefault(lbl, []).append((ax_id, hx))
            table.append((ax_id, lbl, hx))

    detail = [f"{a} | {l} -> {h}" for (a, l, h) in table]
    inconsistent = []
    for lbl, entries in sem.items():
        hexes = set(h for _, h in entries)
        if len(hexes) > 1 and len(set(a for a, _ in entries)) > 1:
            inconsistent.append((lbl, sorted(hexes)))

    if inconsistent:
        return _policy_record(
            "WARN",
            f"{len(inconsistent)} label(s) use different colors across panels",
            detail=detail,
            suggestions=[f"Unify color for {lbl!r} across panels (found {hxs})"
                         for lbl, hxs in inconsistent]
            + ["Cross-panel correspondence is load-bearing: a recurring element "
               "must use the SAME color in every panel."])
    return _policy_record(
        "PASS",
        "no cross-panel legend-label color conflicts" if table else "no multi-panel legends to compare",
        detail=detail, suggestions=[])


def check_panel_spacing(fig, min_gap_px=8.0):
    """Minimum pixel gap between adjacent axes content regions. WARN if two axes
    are closer than min_gap_px (and not overlapping -- overlap is the detector's
    job)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = [ax for ax in fig.axes if ax.get_visible()]
    if len(axes) < 2:
        return _policy_record("PASS", "single panel; spacing not applicable",
                              detail=[], suggestions=[])
    boxes = []
    for ax in axes:
        try:
            bb = ax.get_window_extent(renderer=renderer)
            boxes.append((_ax_identifier(ax), bb))
        except Exception:
            continue
    detail = []
    too_close = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (ia, ba), (ib, bbx) = boxes[i], boxes[j]
            v_overlap = min(ba.y1, bbx.y1) - max(ba.y0, bbx.y0)
            h_overlap = min(ba.x1, bbx.x1) - max(ba.x0, bbx.x0)
            gaps = []
            if v_overlap > 0:
                gx = max(ba.x0, bbx.x0) - min(ba.x1, bbx.x1)
                if gx >= 0:
                    gaps.append(("horizontal", gx))
            if h_overlap > 0:
                gy = max(ba.y0, bbx.y0) - min(ba.y1, bbx.y1)
                if gy >= 0:
                    gaps.append(("vertical", gy))
            for orient, g in gaps:
                if g < min_gap_px:
                    too_close.append((ia, ib, orient, g))
                    detail.append(f"{ia} <-> {ib}: {orient} gap {g:.1f}px (< {min_gap_px:.0f}px)")
    if too_close:
        return _policy_record(
            "WARN", f"{len(too_close)} adjacent-axes pair(s) closer than {min_gap_px:.0f}px",
            detail=detail,
            suggestions=["Increase gridspec wspace/hspace or use constrained_layout / "
                         "fig.tight_layout() to give panels room."])
    return _policy_record("PASS", f"all adjacent panels >= {min_gap_px:.0f}px apart",
                          detail=[], suggestions=[])


def check_line_width(fig, min_pt=MIN_LINEWIDTH_PT):
    """Flag Line2D / Patch edge widths below the print threshold (points)."""
    fig.canvas.draw()
    thin = []
    detail = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        ax_id = _ax_identifier(ax)
        for ln in ax.lines:
            if not ln.get_visible():
                continue
            try:
                lw = float(ln.get_linewidth())
            except Exception:
                continue
            if lw <= 0:
                continue
            ls = ln.get_linestyle()
            if ls in ("none", "None", " ", ""):
                continue
            if lw < min_pt:
                thin.append(lw)
                detail.append(f"{ax_id}: Line2D lw={lw:.2f}pt (< {min_pt}pt)")
        for p in ax.patches:
            if not p.get_visible():
                continue
            try:
                lw = float(p.get_linewidth())
            except Exception:
                continue
            ec = p.get_edgecolor()
            try:
                import matplotlib.colors as mcolors
                visible_edge = (ec is not None) and (len(ec) < 4 or ec[3] > 0)
            except Exception:
                visible_edge = True
            if lw > 0 and visible_edge and lw < min_pt:
                thin.append(lw)
                detail.append(f"{ax_id}: {type(p).__name__} edge lw={lw:.2f}pt (< {min_pt}pt)")
    if thin:
        return _policy_record(
            "WARN", f"{len(thin)} line/edge width(s) below {min_pt}pt print floor",
            detail=detail,
            suggestions=[f"Set linewidth >= {min_pt}pt (0.176mm) so lines survive print/downscale."])
    return _policy_record("PASS", f"all drawn line/edge widths >= {min_pt}pt",
                          detail=[], suggestions=[])


def _sample_figure_colors(fig):
    """Collect a deduped set of representative colors actually used by data /
    legend artists. Returns list of (hex, source_label)."""
    import matplotlib.colors as mcolors

    seen = {}
    def _add(rgb, src):
        if rgb is None:
            return
        try:
            hx = mcolors.to_hex(rgb)
        except Exception:
            return
        if hx == "#ffffff":
            return
        seen.setdefault(hx, src)

    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for ln in ax.lines:
            if ln.get_visible() and ln.get_linestyle() not in ("none", "None"):
                _add(_artist_color(ln), "line")
        for p in ax.patches:
            if p.get_visible():
                _add(_artist_color(p), "patch")
        for coll in ax.collections:
            if not coll.get_visible():
                continue
            try:
                fc = coll.get_facecolor()
                if fc is not None and len(fc):
                    _add(mcolors.to_rgb(fc[0]), "collection")
            except Exception:
                pass
        lg = ax.get_legend()
        if lg is not None:
            handles = lg.legend_handles if hasattr(lg, "legend_handles") else getattr(lg, "legendHandles", [])
            for h in handles:
                _add(_artist_color(h), "legend")
    return [(hx, src) for hx, src in seen.items()]


def check_palette_compliance(fig, palette=None, tol_deltae=12.0):
    """Sample artist colors; flag those not close to an approved palette
    (Okabe-Ito by default) and suggest the nearest approved color. Pure-stdlib
    (CIELAB ΔE) -> always active, no optional dep."""
    pal = palette or OKABE_ITO
    pal_lab = {name: _srgb_to_lab(_to_rgb01(hx)) for name, hx in pal.items()}
    colors = _sample_figure_colors(fig)
    if not colors:
        return _policy_record("PASS", "no sampled data/legend colors", detail=[], suggestions=[])
    detail = []
    off = []
    for hx, src in colors:
        lab = _srgb_to_lab(_to_rgb01(hx))
        best = min(pal_lab.items(), key=lambda kv: _delta_e76(lab, kv[1]))
        d = _delta_e76(lab, best[1])
        ok = d <= tol_deltae
        detail.append(f"{hx} ({src}) -> nearest {best[0]} {pal[best[0]]} ΔE={d:.1f}{'' if ok else '  OFF'}")
        if not ok:
            off.append((hx, src, best[0], pal[best[0]], d))
    if off:
        return _policy_record(
            "WARN", f"{len(off)}/{len(colors)} colors off the approved palette",
            detail=detail,
            suggestions=[f"Replace {hx} ({src}) with {nm} {pnh} (ΔE={d:.1f})"
                         for hx, src, nm, pnh, d in off]
            + ["Use the Okabe-Ito CVD-safe palette for categorical color; "
               "viridis/cividis for sequential."])
    return _policy_record("PASS", f"all {len(colors)} colors close to approved palette",
                          detail=detail, suggestions=[])


_MACHADO = {
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779],
                        [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}


def _simulate_cvd_fallback(rgb, kind):
    v = np.clip(_MACHADO[kind] @ np.asarray(rgb, float), 0, 1)
    return tuple(v)


def check_colorblind(fig, min_deltae=15.0):
    """Simulate protan/deutan/tritan and flag color PAIRS whose separation
    collapses below min_deltae under simulation. Prefers `colorspacious`;
    falls back to a built-in Machado matrix; degrades to SKIP only on internal
    error. (Note: this check is always able to run via the fallback; the
    optional dep just improves accuracy.)"""
    colors = [hx for hx, _ in _sample_figure_colors(fig)]
    if len(colors) < 2:
        return _policy_record("PASS", "fewer than 2 colors; CVD pairing N/A",
                              detail=[], suggestions=[])

    sim_fn = None
    backend = "machado-fallback"
    try:
        import colorspacious

        def _cs(rgb, kind):
            cvd = {"name": "sRGB1+CVD",
                   "cvd_type": {"protan": "protanomaly", "deutan": "deuteranomaly",
                                "tritan": "tritanomaly"}[kind],
                   "severity": 100}
            out = colorspacious.cspace_convert(np.asarray(rgb), cvd, "sRGB1")
            return tuple(np.clip(out, 0, 1))
        sim_fn = _cs
        backend = "colorspacious"
    except Exception:
        sim_fn = _simulate_cvd_fallback

    rgbs = [_to_rgb01(hx) for hx in colors]
    detail = []
    risky = []
    for kind in ("protan", "deutan", "tritan"):
        sim = [sim_fn(c, kind) for c in rgbs]
        labs = [_srgb_to_lab(c) for c in sim]
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                d = _delta_e76(labs[i], labs[j])
                if d < min_deltae:
                    risky.append((kind, colors[i], colors[j], d))
                    detail.append(f"{kind}: {colors[i]} ~ {colors[j]} ΔE={d:.1f} (< {min_deltae})")
    if risky:
        return _policy_record(
            "WARN", f"{len(risky)} color pair(s) hard to distinguish under CVD ({backend})",
            detail=detail,
            suggestions=["Add redundant encoding (marker/linestyle/hatch) for CVD-collapsed pairs, "
                         "or pick more separated Okabe-Ito colors."],
            backend=backend)
    return _policy_record("PASS", f"colors distinguishable under simulated CVD ({backend})",
                          detail=detail, suggestions=[], backend=backend)


def check_grayscale(fig, min_deltae=GRAYSCALE_DELTAE_MIN):
    """Desaturate sampled colors, compute pairwise CIELAB ΔE on the grey values,
    flag any pair < ~20 (indistinguishable in B/W print). Pure-stdlib."""
    colors = [hx for hx, _ in _sample_figure_colors(fig)]
    if len(colors) < 2:
        return _policy_record("PASS", "fewer than 2 colors; grayscale pairing N/A",
                              detail=[], suggestions=[])
    rgbs = [_to_rgb01(hx) for hx in colors]
    def _grey(rgb):
        y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        return (y, y, y)
    labs = [_srgb_to_lab(_grey(c)) for c in rgbs]
    detail = []
    risky = []
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            d = _delta_e76(labs[i], labs[j])
            if d < min_deltae:
                risky.append((colors[i], colors[j], d))
                detail.append(f"{colors[i]} ~ {colors[j]} grey-ΔE={d:.1f} (< {min_deltae})")
    if risky:
        return _policy_record(
            "WARN", f"{len(risky)} color pair(s) indistinguishable in grayscale print",
            detail=detail,
            suggestions=["Spread the colors' LUMINANCE (not just hue), or add "
                         "marker/linestyle/hatch so they survive B/W printing."])
    return _policy_record("PASS", "all color pairs separable in grayscale", detail=detail,
                          suggestions=[])


_OCCLUSION_ALPHA_OPAQUE = 0.85
_OCCLUSION_TIP_FRAC = 0.55
_OCCLUSION_TIP_RADIUS_PX = 9.0


def _effective_patch_alpha(p):
    """Effective fill opacity of a Patch in [0,1]: combine get_alpha() with the
    alpha baked into the facecolor RGBA (mpl often bakes it there, not in
    get_alpha)."""
    import matplotlib.colors as mcolors
    a = 1.0
    try:
        ga = p.get_alpha()
        if ga is not None:
            a *= float(ga)
    except Exception:
        pass
    try:
        fc = p.get_facecolor()
        if fc is not None:
            rgba = mcolors.to_rgba(fc[0] if (hasattr(fc, "__len__") and len(fc) and hasattr(fc[0], "__len__")) else fc)
            a *= float(rgba[3])
    except Exception:
        pass
    return a


def _marker_tip_points_px(artist, ax):
    """Return a list of (px, py) pixel-space candidate tip/end points for a
    directional marker artist, or [] if it is not such a marker.

    Tests BOTH path extremities (head and tail): head identification across
    FancyArrow/FancyArrowPatch/annotation arrows is fiddly, and there is no
    'tail-in-box is fine' exemption, so probing both ends is safe and robust."""
    from matplotlib.patches import FancyArrow, FancyArrowPatch
    pts = []
    try:
        if isinstance(artist, FancyArrowPatch):
            pp = getattr(artist, "_posA_posB", None)
            trans = artist.get_transform()
            if pp is not None:
                for xy in pp:
                    pts.append(tuple(trans.transform(xy)))
            else:
                path = artist.get_path()
                v = trans.transform(path.vertices)
                pts.append(tuple(v[0]))
                pts.append(tuple(v[-1]))
        elif isinstance(artist, FancyArrow):
            trans = artist.get_transform()
            v = trans.transform(artist.get_path().vertices)
            v = v[np.isfinite(v).all(axis=1)]
            if len(v):
                pts.append(tuple(v[0]))
                d = np.hypot(v[:, 0] - v[0, 0], v[:, 1] - v[0, 1])
                pts.append(tuple(v[int(np.argmax(d))]))
    except Exception:
        return []
    out = []
    for p in pts:
        if not all(np.isfinite(c) for c in p):
            continue
        if not any(abs(p[0] - q[0]) < 1 and abs(p[1] - q[1]) < 1 for q in out):
            out.append((float(p[0]), float(p[1])))
    return out


def _line2d_end_marker_points_px(ln, ax):
    """Pixel points of the END markers of a Line2D that actually draws markers."""
    try:
        mk = ln.get_marker()
        if mk in (None, "None", "none", "", " "):
            return []
        xd = np.asarray(ln.get_xdata(), float)
        yd = np.asarray(ln.get_ydata(), float)
        pts = np.column_stack([xd, yd])
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) == 0:
            return []
        pix = _artist_px(ln, ax, pts)
        pix = pix[np.isfinite(pix).all(axis=1)]
        if len(pix) == 0:
            return []
        ends = [tuple(pix[0])]
        if len(pix) > 1:
            ends.append(tuple(pix[-1]))
        return [(float(x), float(y)) for x, y in ends]
    except Exception:
        return []


def _render_rgba(fig):
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf.astype(np.int16), w, h


def _tip_window(px, py, w, h, r):
    """Buffer-row/col slice for a square tip region around (px,py). Buffer row 0
    is the TOP of the figure, so the pixel y must be flipped."""
    ry = h - 1 - py
    x0 = int(max(0, np.floor(px - r))); x1 = int(min(w, np.ceil(px + r) + 1))
    y0 = int(max(0, np.floor(ry - r))); y1 = int(min(h, np.ceil(ry + r) + 1))
    return x0, x1, y0, y1


def check_occluded_marker(fig, diff_tol=16, tip_frac=_OCCLUSION_TIP_FRAC,
                          tip_radius_px=_OCCLUSION_TIP_RADIUS_PX):
    """Flag a small directional marker (arrowhead / Line2D end-marker)
    substantially hidden behind a higher-zorder, near-opaque Patch.

    Render-based toggle-diff: for each candidate, render the figure with the
    marker hidden (base), forced on top (footprint), and at its real zorder
    (visible). A tip region is OCCLUDED when most of its footprint pixels are NOT
    visible, the occluding artist there is a Patch with zorder above the marker
    and effective alpha >= 0.85. Restores all artist state before returning."""
    from matplotlib.patches import FancyArrow, FancyArrowPatch, Patch

    candidates = []
    occ_patches = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for p in ax.patches:
            if not p.get_visible():
                continue
            if isinstance(p, (FancyArrow, FancyArrowPatch)):
                tips = _marker_tip_points_px(p, ax)
                if tips:
                    candidates.append((p, ax, tips, p.get_zorder()))
            else:
                a = _effective_patch_alpha(p)
                occ_patches.append((p, ax, p.get_zorder(), a))
        for t in ax.texts:
            ap = getattr(t, "arrow_patch", None)
            if ap is not None and ap.get_visible():
                tips = _marker_tip_points_px(ap, ax)
                if tips:
                    candidates.append((ap, ax, tips, ap.get_zorder()))
        for ln in ax.lines:
            if not ln.get_visible():
                continue
            tips = _line2d_end_marker_points_px(ln, ax)
            if tips:
                candidates.append((ln, ax, tips, ln.get_zorder()))

    if not candidates or not occ_patches:
        fig.canvas.draw()
        return _policy_record("PASS", "no occludable directional markers", detail=[], suggestions=[])

    w, h = fig.canvas.get_width_height()
    detail = []
    occluded = []

    for artist, ax, tips, mz in candidates:
        relevant = []
        for (p, pax, pz, pa) in occ_patches:
            if pa < _OCCLUSION_ALPHA_OPAQUE:
                continue
            higher = (pz > mz) or (abs(pz - mz) < 1e-9)
            if not higher:
                continue
            try:
                pbb = p.get_window_extent()
            except Exception:
                pbb = None
            covers = False
            for (px, py) in tips:
                if pbb is not None and pbb.x0 - 2 <= px <= pbb.x1 + 2 and pbb.y0 - 2 <= py <= pbb.y1 + 2:
                    covers = True
                    break
            if covers:
                relevant.append((p, pa))
        if not relevant:
            continue

        orig_vis = artist.get_visible()
        orig_z = artist.get_zorder()
        try:
            artist.set_visible(False)
            fig.canvas.draw()
            base, _, _ = _render_rgba(fig)
            artist.set_visible(True)
            artist.set_zorder(max(orig_z, 10_000))
            fig.canvas.draw()
            top, _, _ = _render_rgba(fig)
            artist.set_zorder(orig_z)
            fig.canvas.draw()
            full, _, _ = _render_rgba(fig)
        finally:
            artist.set_zorder(orig_z)
            artist.set_visible(orig_vis)

        footprint = np.abs(top - base).sum(axis=2) > diff_tol
        visible = np.abs(full - base).sum(axis=2) > diff_tol
        hidden = footprint & ~visible

        for (px, py) in tips:
            x0, x1, y0, y1 = _tip_window(px, py, w, h, tip_radius_px)
            if x1 <= x0 or y1 <= y0:
                continue
            fp = footprint[y0:y1, x0:x1]
            nfp = int(fp.sum())
            if nfp < 4:
                continue
            nhid = int((hidden[y0:y1, x0:x1]).sum())
            frac = nhid / nfp if nfp else 0.0
            if frac >= tip_frac:
                occ = relevant[0][0]
                ax_id = _ax_identifier(ax)
                name = type(artist).__name__
                onm = type(occ).__name__
                occluded.append((ax_id, name, onm, frac, int(px), int(py)))
                detail.append(
                    f"{ax_id}: {name} tip @({int(px)},{int(py)}) {frac:.0%} hidden behind "
                    f"opaque {onm} (zorder {orig_z} < occluder)")
                break

    fig.canvas.draw()

    if occluded:
        return _policy_record(
            "WARN", f"{len(occluded)} directional marker(s) occluded by an opaque higher-zorder patch",
            detail=detail,
            suggestions=["Raise the arrow/marker zorder above the box, make the box edge-only / "
                         "translucent (alpha<0.85), or move the box so the arrowhead is visible.",
                         "An arrowhead hidden behind an opaque gene box reads as a deleted cue -- "
                         "the reader loses the strand/direction it encodes."])
    return _policy_record("PASS", "no directional marker hidden behind an opaque patch",
                          detail=[], suggestions=[])


def check_orphan_label(fig, near_px=42.0):
    """Flag a short single-token text label (e.g. 5'/3' strand-polarity cues, a
    stray single char) that is either (a) far from any Line2D / strand end, or
    (b) DUPLICATED within the same axes.

    Only hand-placed ``ax.texts`` are considered (NOT tick/axis labels or
    titles -- those are legitimately short and far from line ends). Detected
    panel letters (a, b, 'a)' ...) are exempted so the margin panel-letter
    convention does not false-positive.

    NOTE (intended over-flag, advisory): a legitimate two-strand cartoon carries
    two 5' labels in one axes; this WARNs on the duplicate. The policy is
    advisory, so this surfaces it for author review rather than blocking."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _is_short_token(s):
        s = s.strip()
        if not s or " " in s or "\n" in s:
            return False
        return len(s) <= 2

    detail = []
    flagged = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        ax_id = _ax_identifier(ax)

        ends = []
        for ln in ax.lines:
            if not ln.get_visible():
                continue
            try:
                xd = np.asarray(ln.get_xdata(), float)
                yd = np.asarray(ln.get_ydata(), float)
                pts = np.column_stack([xd, yd])
                pts = pts[np.isfinite(pts).all(axis=1)]
                if len(pts) == 0:
                    continue
                pix = _artist_px(ln, ax, pts)
                pix = pix[np.isfinite(pix).all(axis=1)]
                if len(pix):
                    ends.append(pix[0])
                    ends.append(pix[-1])
            except Exception:
                continue

        cands = []
        content_counts = {}
        for t in ax.texts:
            if not isinstance(t, Text) or not _text_is_meaningful(t):
                continue
            s = t.get_text()
            if not _is_short_token(s):
                continue
            if _PANEL_LETTER_RE.match(s.strip()):
                continue
            ext = _safe_extent(t, renderer)
            if ext is None:
                continue
            cx = 0.5 * (ext[0] + ext[2]); cy = 0.5 * (ext[1] + ext[3])
            cands.append((t, s.strip(), (cx, cy)))
            content_counts[s.strip()] = content_counts.get(s.strip(), 0) + 1

        for (t, content, (cx, cy)) in cands:
            reasons = []
            if content_counts.get(content, 0) >= 2:
                reasons.append("duplicated in axes")
            if ends:
                dmin = min(float(np.hypot(cx - e[0], cy - e[1])) for e in ends)
                if dmin > near_px:
                    reasons.append(f"far from any line end ({dmin:.0f}px > {near_px:.0f}px)")
            if reasons:
                flagged.append((ax_id, content, int(cx), int(cy), reasons))
                detail.append(f"{ax_id}: {content!r} @({int(cx)},{int(cy)}) -- " + "; ".join(reasons))

    if flagged:
        return _policy_record(
            "WARN", f"{len(flagged)} orphan/duplicate short label(s)",
            detail=detail,
            suggestions=["Place a 5'/3' polarity cue AT the strand end it labels; delete stray/"
                         "leftover single-token labels.",
                         "If two strands legitimately each need a 5' cue, this is an advisory "
                         "duplicate flag -- confirm both are intentional and anchored."])
    return _policy_record("PASS", "no orphan/duplicate short labels", detail=[], suggestions=[])


def check_canvas_containment(fig, tol_px=0.5):
    """(l) Any VISIBLE artist whose rendered extent falls outside the figure canvas.

    The blind spot this closes: ``figstyle.panel_letters`` places its letter at
    ``dy=1.06`` in AXES fractions, which on a plain 1x2 double-column figure puts
    it at ``y1/H = 1.087`` -- entirely off-canvas. ``savefig.bbox='tight'``
    silently rescues it by expanding the crop, so nothing looks wrong; the moment
    ``bbox='tight'`` is dropped for exact 89/183 mm column widths, the letter is
    simply GONE from the saved PNG. Measured: top-left dark-pixel band 1023 -> 120.

    Artists are skipped when they are invisible, degenerate, or clipped to their
    axes (a clipped artist cannot paint outside it, so a large geometric extent
    is not an overhang -- see ``_clip_ext_to_axes``).
    
    """
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception as exc:
        return _policy_record("SKIP", f"could not render figure ({type(exc).__name__})")

    W, H = float(fig.bbox.width), float(fig.bbox.height)
    if not (np.isfinite(W) and np.isfinite(H)) or W <= 0 or H <= 0:
        return _policy_record("SKIP", "figure canvas has no finite size")

    offenders = []

    def _consider(artist, label):
        try:
            if not artist.get_visible():
                return
        except Exception:
            return
        try:
            if artist.get_clip_on():
                return
        except Exception:
            pass
        ext = _safe_extent(artist, renderer)
        if ext is None:
            return
        x0, y0, x1, y1 = ext
        over = {}
        if x0 < -tol_px:
            over["left"] = (0.0 - x0) / W
        if y0 < -tol_px:
            over["bottom"] = (0.0 - y0) / H
        if x1 > W + tol_px:
            over["right"] = (x1 - W) / W
        if y1 > H + tol_px:
            over["top"] = (y1 - H) / H
        if over:
            offenders.append({
                "artist": label,
                "bbox_px": [round(v, 1) for v in ext],
                "overhang_figfrac": {k: round(v, 4) for k, v in over.items()},
            })

    for i, t in enumerate(getattr(fig, "texts", []) or []):
        _consider(t, f"fig.text:{_text_identity(t)}")
    for lg in getattr(fig, "legends", []) or []:
        _consider(lg, "fig.legend")
    for ax in fig.get_axes():
        try:
            if not ax.get_visible():
                continue
        except Exception:
            continue
        ax_id = _ax_identifier(ax)
        for t in getattr(ax, "texts", []) or []:
            _consider(t, f"{ax_id} text:{_text_identity(t)}")
        lg = ax.get_legend()
        if lg is not None:
            _consider(lg, f"{ax_id} legend")

    if not offenders:
        return _policy_record("PASS", "all visible artists lie within the figure canvas")
    names = ", ".join(o["artist"] for o in offenders[:3])
    more = "" if len(offenders) <= 3 else f" (+{len(offenders) - 3} more)"
    return _policy_record(
        "WARN",
        f"{len(offenders)} artist(s) extend beyond the canvas: {names}{more}",
        detail=offenders,
        suggestions=[
            "These are invisible in the saved PNG unless savefig.bbox='tight' expands "
            "the crop to rescue them -- do not rely on that; it also voids the exact "
            "89/183 mm column width.",
            "For panel letters, clamp the offset or use a fixed point offset from the "
            "axes corner instead of an axes-fraction dy > 1.0.",
        ],
    )


_POLICY_CHECKS = [
    ("panel_letters", "Panel letters present + in margin", check_panel_letters),
    ("canvas_containment", "Artists within figure canvas", check_canvas_containment),
    ("glyph_coverage", "Font glyph coverage / mojibake risk", check_glyph_coverage),
    ("provenance_footer", "Provenance footer present", check_provenance_footer),
    ("cross_panel_color", "Cross-panel color consistency", check_cross_panel_color),
    ("panel_spacing", "Adjacent-panel spacing", check_panel_spacing),
    ("line_width", "Line-width legibility", check_line_width),
    ("palette_compliance", "Okabe-Ito palette compliance", check_palette_compliance),
    ("colorblind", "Colorblind simulation", check_colorblind),
    ("grayscale", "Grayscale printability", check_grayscale),
    ("occluded_marker", "Directional marker occlusion", check_occluded_marker),
    ("orphan_label", "Orphan / duplicate short label", check_orphan_label),
]


def policy_check(fig, include=None, exclude=None):
    """Run the FIGURE_STYLE_POLICY.md auto-checkable subset on ``fig`` and return
    a structured, JSON-able report dict::

        {"overall": "PASS"|"WARN",
         "n_pass": int, "n_warn": int, "n_skip": int,
         "n_overlaps": int,
         "checks": {key: {"name", "status", "summary", "detail", "suggestions", ...}}}

    **Runs the overlap detector** and folds it in as an ``overlaps`` check, so this
    can never report ``PASS`` on a figure whose caption sits on the data. Before
    2026-08-03 the two passes were separate and nothing ran both: `policy_check`
    returned PASS 11/11 on a figure where `find_overlaps` reported 13 collisions,
    which was the single most misleading output in this tool.

    The **gate** is ``include_data=False`` -- text/annotation/legend collisions, the
    classes the suite positively asserts and where false positives are rare. The
    ``include_data=True`` count is still computed and reported as **advisory**
    (``detail['n_overlaps_incl_data']``), never as a gate: `Line2D` window extents
    span the whole data range, so gating on them would fail 5 of 6 real workspace
    figures. Trade-off accepted knowingly: a legend sitting on data is advisory
    here -- run ``find_overlaps(fig)`` directly when that matters.

    Never raises. Use ``include``/``exclude`` (iterables of check keys) to subset.
    """
    results = {}
    for key, name, fn in _POLICY_CHECKS:
        if include is not None and key not in include:
            continue
        if exclude is not None and key in exclude:
            continue
        rec = _safe_check(fn, fig)
        rec["name"] = name
        results[key] = rec
    n_ov = 0
    run_overlaps = (include is None or "overlaps" in include) and \
                   (exclude is None or "overlaps" not in exclude)
    if run_overlaps:
        def _overlap_check(f):
            gate = find_overlaps(f, include_data=False)
            try:
                advisory = len(find_overlaps(f, include_data=True))
            except Exception:
                advisory = None
            n = len(gate)
            det = {"n_overlaps": n, "n_overlaps_incl_data": advisory,
                   "overlaps": [o.as_dict() for o in gate[:20]]}
            if n == 0:
                return _policy_record(
                    "PASS", f"no text/legend collisions (advisory incl. data: {advisory})",
                    detail=det)
            manual = sum(1 for o in gate if getattr(o, "manual_fix_needed", False))
            return _policy_record(
                "WARN", f"{n} text/legend collision(s)"
                        + (f", {manual} need a manual fix" if manual else ""),
                detail=det,
                suggestions=[
                    "Run figqa.report(figqa.find_overlaps(fig, include_data=False)) for the pairs.",
                    "Legends: place_legend_clear(ax). Plain labels: adjust_labels(ax).",
                    "Annotations and data collisions are NOT auto-movable -- fix the script.",
                ])
        rec = _safe_check(_overlap_check, fig)
        rec["name"] = "Text/legend overlaps"
        results["overlaps"] = rec
        n_ov = (rec.get("detail") or {}).get("n_overlaps", 0) or 0

    n_pass = sum(1 for r in results.values() if r["status"] == "PASS")
    n_warn = sum(1 for r in results.values() if r["status"] == "WARN")
    n_skip = sum(1 for r in results.values() if r["status"] == "SKIP")
    overall = "WARN" if n_warn else "PASS"
    return {"overall": overall, "n_pass": n_pass, "n_warn": n_warn,
            "n_skip": n_skip, "n_overlaps": n_ov, "checks": results}


def policy_report(result) -> str:
    """Human-readable rendering of a ``policy_check`` result dict."""
    lines = []
    lines.append("=" * 72)
    lines.append("figproof.qa policy compliance (FIGURE_STYLE_POLICY.md auto-checkable subset)")
    lines.append("=" * 72)
    for key, rec in result["checks"].items():
        st = rec["status"]
        lines.append(f"[{st:4s}] {rec.get('name', key):42s} {rec.get('summary', '')}")
    lines.append("-" * 72)
    lines.append(f"overall: {result['overall']}  "
                 f"({result['n_pass']} PASS / {result['n_warn']} WARN / {result['n_skip']} SKIP)")
    for key, rec in result["checks"].items():
        if rec["status"] == "PASS":
            continue
        lines.append("")
        lines.append(f"### {rec.get('name', key)} [{rec['status']}] ###")
        for d in rec.get("detail", []):
            lines.append(f"  - {d}")
        for s in rec.get("suggestions", []):
            lines.append(f"  -> {s}")
    lines.append("")
    lines.append("Documentation-only checklist (human judgment -- see FIGURE_STYLE_POLICY.md):")
    for item in POLICY_DOC_CHECKLIST:
        lines.append(f"  [ ] {item}")
    return "\n".join(lines)


POLICY_DOC_CHECKLIST = [
    "Titles/labels declarative & caption-style (no hype/hedge words).",
    "Ruthless text economy -- each point stated once; heavy detail pushed to a supplement.",
    "Lead with the strongest evidence visually; label artifacts honestly; keep inferred-vs-proven caveats.",
    "Split overloaded multi-panels into a crisp MAIN + a SUPPLEMENT; cross-link.",
    "Anchor schematics to real coordinates; keep a coordinate-anchored overview.",
    "Schematic polarity: 5'/3' cues on every strand; name the homology/nt mediating each step.",
    "No stray/orphan artists; arrows/connectors do NOT strike through text.",
    "Consistent type sizes/weights; legible at print column width (89mm / 183mm).",
]


PROFILES = {
    "internal": {
        "panel_letter_pt": 14,
        "savefig_bbox": "tight",
        "exact_width": False,
    },
    "submission": {
        "panel_letter_pt": 8,
        "savefig_bbox": None,
        "exact_width": True,
    },
}

MIN_AXES_FRAC = 0.45


def _sidecar_path(png_path):
    """`NN_slug_fig.png` -> `NN_slug_fig.md` (K4)."""
    base, _ = os.path.splitext(png_path)
    return base + ".md"


_SIDECAR_STUB = """# {name}

> **STUB — written by `figqa.finalize()`. Fill in the TODOs.**
> The concise legend on the figure is deliberately short; this file is the
> reviewer-grade record the standard requires.

## Data source
TODO: accession(s) / protocol / strain.

## Processing
TODO: window, normalization, gate/atlas version, filters applied.

## Reference datasets
TODO: what this is compared against.

## Per-panel
TODO: (a) ... (b) ...

## Caveats
TODO: what could overturn this; inferred-vs-proven boundary.

---
figure: `{png}`
script: `{script}`
rendered: {ts}
"""


def finalize(fig, path, legend=None, provenance=None, profile="internal",
             strict=False, include_data=False, dry_run=False,
             letters="abcdefgh", pad=0.03, sidecar=True, script=None):
    """Render, QA, save, and write the `.md` sidecar in one call.

    This exists because the figure standard has six individually-forgettable steps
    and **nothing enforced them**: measured 2026-08-03, ZERO figure scripts failed
    a build on an unresolved overlap, so the QA pass was a manual step an agent
    could skip silently.

    Order is load-bearing and is NOT the obvious one::

        profile -> legend + rect reservation -> subplots_adjust -> PANEL LETTERS
        -> provenance -> draw -> checks -> save -> sidecar

    Returns a dict: ``{png, sidecar, n_overlaps, policy, warnings, saved}``.

    Parameters
    ----------
    strict : bool
        **Default False — warn and save.** A strict gate is not safe as a default
        yet: a compliant single-column figure still has real (non-phantom)
        residual collisions at every legend length swept. When True, raises
        ``RuntimeError`` on gate-class overlaps and **leaves no PNG on disk**.
    include_data : bool
        Gate class. False (default) = text/legend collisions only; the
        ``include_data=True`` count is always computed and reported as advisory.
    dry_run : bool
        Report every path that WOULD be written and write nothing.
    
    """
    import datetime as _dt
    warnings_out = []

    prof = PROFILES.get(profile)
    if prof is None:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")

    try:
        import figstyle as _figstyle
    except ImportError as exc:
        raise RuntimeError(
            "finalize() needs figstyle (it ships beside figqa in figproof/qa). "
            f"Import failed: {exc}"
        )

    from matplotlib.layout_engine import ConstrainedLayoutEngine
    eng = fig.get_layout_engine()
    if eng is not None and not isinstance(eng, ConstrainedLayoutEngine):
        raise RuntimeError(
            f"layout engine {type(eng).__name__} is not safe for the standard legend "
            f"reservation (adjust_compatible={getattr(eng, 'adjust_compatible', None)} — "
            "this flag is unreliable; TightLayoutEngine reports True and clobbers "
            "subplots_adjust at draw anyway). Build the figure with layout='none', "
            "constrained_layout=False, or layout='constrained'."
        )

    legend_top = None
    if legend:
        legend_top = _figstyle.concise_legend(fig, legend, script=script or "", pad=pad)
        if not isinstance(eng, ConstrainedLayoutEngine):
            fig.subplots_adjust(bottom=legend_top + pad)

    axes = [a for a in fig.get_axes() if a.get_visible()]
    if letters and len(axes) > 1:
        _figstyle.panel_letters(axes, letters=letters, fontsize=prof["panel_letter_pt"])

    if provenance:
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        bits = [str(provenance.get(k)) for k in ("source", "processing", "script") if provenance.get(k)]
        fig.text(0.5, 0.005, " · ".join(bits + [ts]), ha="center", fontsize=5.5, color="0.45")

    fig.canvas.draw()

    if legend and axes:
        _h = max(a.get_position().y1 for a in axes) - min(a.get_position().y0 for a in axes)
        if _h < MIN_AXES_FRAC:
            warnings_out.append(
                f"legend leaves the plot only {_h:.0%} of figure height "
                f"(< {MIN_AXES_FRAC:.0%}). Shorten the legend, widen to "
                f"nature_figsize('double'), or use profile='submission'."
            )

    gate = find_overlaps(fig, include_data=include_data)
    advisory = len(find_overlaps(fig, include_data=True))
    pol = policy_check(fig)
    for key, rec in pol["checks"].items():
        if rec["status"] == "WARN" and key != "overlaps":
            warnings_out.append(f"policy/{key}: {rec['summary']}")

    png_path = os.path.abspath(path)
    md_path = _sidecar_path(png_path) if sidecar else None

    if strict and gate:
        raise RuntimeError(
            f"finalize(strict=True): {len(gate)} gate-class overlap(s); nothing written.\n"
            + report(gate)
        )

    if dry_run:
        return {"png": png_path, "sidecar": md_path, "n_overlaps": len(gate),
                "n_overlaps_incl_data": advisory, "policy": pol,
                "warnings": warnings_out, "saved": False, "dry_run": True}

    prev_bbox = matplotlib.rcParams.get("savefig.bbox")
    try:
        matplotlib.rcParams["savefig.bbox"] = prof["savefig_bbox"]
        fig.savefig(png_path)
    finally:
        matplotlib.rcParams["savefig.bbox"] = prev_bbox

    if md_path:
        if os.path.exists(md_path):
            warnings_out.append(
                f"sidecar NOT written: {md_path} already exists (refusing to overwrite). "
                "If it is a findings doc, rename the figure to <NN>_<slug>_fig.png."
            )
        else:
            with open(md_path, "w", encoding="utf-8") as fh:
                fh.write(_SIDECAR_STUB.format(
                    name=os.path.basename(png_path).rsplit(".", 1)[0],
                    png=os.path.basename(png_path), script=script or "TODO",
                    ts=_dt.datetime.now().strftime("%Y-%m-%d %H:%M")))

    return {"png": png_path, "sidecar": md_path, "n_overlaps": len(gate),
            "n_overlaps_incl_data": advisory, "policy": pol,
            "warnings": warnings_out, "saved": True, "dry_run": False}


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="figproof.qa overlap detector (+ optional --policy compliance)")
    p.add_argument("script", help="path to a .py figure script (top-level builds a figure, or defines build())")
    p.add_argument("--no-data", action="store_true", help="do not include data artists")
    p.add_argument("--tol-px", type=float, default=2.0)
    p.add_argument("--frac-tol", type=float, default=0.05)
    p.add_argument("--policy", action="store_true",
                   help="ALSO run the FIGURE_STYLE_POLICY.md compliance checks (advisory; "
                        "does not affect exit code unless --strict)")
    p.add_argument("--strict", action="store_true",
                   help="with --policy, also exit non-zero if any policy check WARNs")
    p.add_argument("--json", action="store_true",
                   help="with --policy, print the policy result as JSON instead of text")
    args = p.parse_args(argv)

    figs = _load_figures_from_script(args.script)
    if not figs:
        print("figproof.qa: ERROR -- no Figure found in script.", file=sys.stderr)
        return 2

    total = 0
    policy_warn = 0
    for idx, fig in enumerate(figs):
        ov = find_overlaps(
            fig,
            include_data=not args.no_data,
            tol_px=args.tol_px,
            frac_tol=args.frac_tol,
        )
        if len(figs) > 1:
            print(f"--- Figure {idx} ---")
        print(report(ov))
        total += len(ov)
        if args.policy:
            pres = policy_check(fig)
            policy_warn += pres["n_warn"]
            if args.json:
                import json
                print(json.dumps(pres, indent=2, default=str))
            else:
                print()
                print(policy_report(pres))
    rc = 1 if total else 0
    if args.policy and args.strict and policy_warn:
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
