"""SIBLING of `glyphs.py` (cells and colonies) and `alignment_grammar.py` (base/codon alignment).
Imported the same way::

THE RISK OF A SIBLING IS A FORK, and that is what this whole effort exists to prevent. So the
shared convention below is a MECHANISM, not a docstring promise: `tests/track_conformance.py`
builds a real `AlignmentPanel`, reads back the column positions `draw_panel` actually produced, and
asserts a `Track` over the same bases puts every base at the same normalised position. If either
module changes its mind about where a base sits, that test fails.

==================================================================================================
 2. THE SHARED COORDINATE CONVENTION  -- read this before using `Track`
==================================================================================================
Taken from what `alignment_grammar` DOES, not from what would be tidy:

    a base occupies a COLUMN of width `bpw`.
    edge(bp)  is the column's LOW-COORDINATE BOUNDARY   -- `xs[j]` in the alignment grammar
    x(bp)     is the column's CENTRE                     -- `xs[j] + 0.5`, where letters are drawn
    span(a,b) is edge(a) .. edge(b+1), EDGE TO EDGE      -- `xs[j], xs[j+2] + 1.0`, the PAM box

🔴 A CUT IS AN EDGE, NOT A CENTRE. A blunt cut falls BETWEEN two bases, so it is drawn at
`edge(bp)`, and the alignment grammar agrees: it draws the Cas9 cut at `xs[jc[0]]`, the boundary,
while every base letter is at `+0.5`. Getting this wrong is a half-base error that looks perfect
and is the miniature of "a PAM three bases from where the sequence says it is".

`direction=-1` mirrors the whole track for a locus drawn right-to-left; `edge`/`x`/`span` all follow
it, so no caller does its own arithmetic.

  * the published msr purple #5F3D97 -- the only hue held exactly constant across EDGE 2023,
    TAGC 2024 and Sci Adv 2025 -- FAILS as a semantic token: dE2000 7.6 against `guide` under
    deuteranopia, floor 12.0. It is the same trap ARCHITECTURE section 2 already recorded for the
    purple #8C5FA8. It would also have made `editing_mechanism` a FIVE-accent set.
  * no purple anywhere within dE 45 of it clears dE 15 against the palette; the nearest that clears
    the bare 12.0 floor does so at 12.1, i.e. with no margin at all.
  * and the tint itself cannot carry much: `donor` #56B4E9 sits at contrast 2.31 on paper against a
    semantic floor of 2.0, so there is only 0.31 of headroom to lighten it. The lightest validated
    tint with any contrast margin is +3.1 L*, which is not a value step a reader will see.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy import ndimage
from matplotlib.patches import Circle, Ellipse, PathPatch, Polygon, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

from . import glyphs as G

TOK = G.TOK
INK, RULE, MUTE, PAPER = G.INK, G.RULE, G.MUTE, G.PAPER
GUIDE, DONOR, BARCODE = G.GUIDE, G.DONOR, G.BARCODE
RECRUIT, EDIT, PHOSPHO = G.RECRUIT, G.EDIT, G.PHOSPHO
DOMAIN, PAM = G.DOMAIN, G.PAM
PROTEIN_BODY = G.TOK["palette"]["structural"]["protein_body"]
TYPE, STROKE = G.TYPE, G.STROKE
inch_axes, text_in, shade, exempt = G.inch_axes, G.text_in, G.shade, G.exempt
check_type_floor, check_stroke_floor, check_floors = (
    G.check_type_floor, G.check_stroke_floor, G.check_floors)

__all__ = [
    "Track", "REGISTERS", "register_spec",
    "duplex", "helix", "helix_path", "ease_path", "surface", "load_structure", "list_structures", "STRUCT_TOKENS",
    "BP_PER_TURN", "SURFACE_LUMP_MIN_IN", "SURFACE_RENDER_MIN_IN", "SURFACE_LIGHT",
    "PROTEIN_BODY", "SURFACE_RELIEF_MIN_IN", "KEYLINE_FRAC", "keyline_lw", "surface_render", "surface_legend", "EVIDENCE_NOTE", "shade_depth",
    "strand", "region_arc", "span_bar", "polarity", "break_", "elide", "mark",
    "protein", "cas9", "guide_rna", "rloop", "pam_mark", "cut_mark",
    "base_editor", "prime_editor",
    "copied_span", "sdsa",
    "two_state", "ssdna_loop", "A_PER_NT",
    "fork", "crossover", "FORK_LW_RANGE", "UNASSIGNED_CARRIER",
    "INK", "RULE", "MUTE", "GUIDE", "DONOR", "BARCODE", "RECRUIT", "EDIT", "PHOSPHO",
    "DOMAIN", "PAM",
    "check_type_floor", "check_stroke_floor", "check_floors", "exempt",
]


REGISTERS = ("icon", "cartoon", "mechanism", "sequence")

_REGISTER = {
    "icon":     dict(protein="flat neutral body, name inside",
                     hue="the guide and at most one addition",
                     detail=0, rungs="comb", polarity="none", labels=False, accents=2),
    "cartoon":  dict(protein="flat neutral body; lobes named, not coloured",
                     hue="the nucleic acids", detail=0, rungs="comb",
                     polarity="none", labels=True, accents=4),
    "mechanism": dict(protein="flat neutral body; lobes named, not coloured",
                      hue="the nucleic acids", detail=1, rungs="ladder",
                      polarity="corners", labels=True, accents=4),
    "sequence": dict(protein="a segmented domain bar and nothing else",
                     hue="the domains, acting as a legend", detail=2, rungs="ladder",
                     polarity="corners", labels=True, accents=4),
}


def register_spec(register: str) -> dict:
    """What a register commits the caller to. Raises on an unknown name rather than guessing."""
    if register not in _REGISTER:
        raise ValueError(f"unknown register {register!r}; expected one of {REGISTERS}")
    return dict(_REGISTER[register])


class Track:
    """A base-pair coordinate axis on the page, and the inch frame every glyph on it draws in.

    A cell glyph is placed at an anchor and sized in inches. A molecular object is placed ALONG a
    track, at a coordinate in BASE PAIRS. This class is the second composition model that split
    `molecules.py` off from `glyphs.py`, and it owns the bp -> inches conversion so that no caller
    ever does that arithmetic itself.

    Parameters
    ----------
    ax, x, y : the anchor, in `coords` (default axes fraction). The track's CENTRELINE is at y.
    width    : the track's length in INCHES.
    bp0, bp1 : the first and last base pair the track shows, INCLUSIVE.
    direction: +1 draws increasing bp to the right; -1 mirrors the whole track.
    register : one of REGISTERS -- sets default detail, rung form and polarity for glyphs on it.

    Coordinates (see module docstring section 2 -- these mirror `alignment_grammar` exactly):
      ``edge(bp)``   the low-coordinate boundary of base `bp`  -- where a CUT goes
      ``x(bp)``      the centre of base `bp`                   -- where a LETTER goes
      ``span(a,b)``  ``(edge(a), edge(b+1))``, edge to edge    -- where a FEATURE BOX goes
    """

    def __init__(self, ax, x, y, *, width, bp0, bp1, direction=+1, coords="axes",
                 register="mechanism", zorder=4):
        if bp1 < bp0:
            raise ValueError(f"bp1 ({bp1}) is before bp0 ({bp0}); use direction=-1 to mirror")
        if direction not in (+1, -1):
            raise ValueError("direction must be +1 or -1")
        register_spec(register)
        self.ax, self.anchor, self.coords = ax, (x, y), coords
        self.width, self.bp0, self.bp1 = float(width), int(bp0), int(bp1)
        self.direction, self.register, self.zorder = direction, register, zorder
        self.tr = inch_axes(ax, x, y, coords)
        self.bpw = self.width / self.n
        self._x0 = 0.0 if direction > 0 else self.width

    @property
    def n(self) -> int:
        """Number of base pairs on the track (inclusive of both ends)."""
        return self.bp1 - self.bp0 + 1

    def edge(self, bp) -> float:
        """The LOW-COORDINATE BOUNDARY of base `bp`, in inches from the track anchor.

        🔴 This is where a cut, a nick and a junction go -- they fall BETWEEN two bases. It is also
        where a span starts. `alignment_grammar` draws its Cas9 cut at exactly this position
        (`xs[jc[0]]`), not at a base centre.
        """
        return self._x0 + self.direction * (bp - self.bp0) * self.bpw

    def x(self, bp) -> float:
        """The CENTRE of base `bp`, in inches from the track anchor -- where a letter is drawn.

        The alignment grammar's `xs[j] + 0.5`.
        """
        return self.edge(bp) + self.direction * self.bpw / 2.0

    def span(self, bp_a, bp_b):
        """`(lo, hi)` in inches covering bases `bp_a..bp_b` INCLUSIVE, edge to edge.

        The alignment grammar's `xs[j], xs[j+2] + 1.0` for a 3-base PAM box.
        """
        a, b = self.edge(bp_a), self.edge(bp_b + 1)
        return (a, b) if a <= b else (b, a)

    def contains(self, bp) -> bool:
        return self.bp0 <= bp <= self.bp1

    def clamp(self, bp) -> int:
        return max(self.bp0, min(self.bp1, int(bp)))

    def frame(self, bp, dy=0.0, *, at_edge=False):
        """`{"tr":…, "at":…}` to splat into any free-standing glyph, placing it at base `bp`.

        This is how a protein, a machine or a switch is hung on a track without the caller doing
        bp arithmetic::

            M.cas9(ax, 0, 0, 0.6, **t.frame(31, dy=0.30))

        `at_edge=True` places it at the base BOUNDARY instead of the base centre -- use it for
        anything that sits at a cut.
        """
        return dict(tr=self.tr, at=((self.edge(bp) if at_edge else self.x(bp)), dy))

    def axes_xy(self, dx_in, dy_in):
        """Convert a point in this track's inch frame to the axes fraction a CELL glyph wants."""
        fig = self.ax.figure
        bb = self.ax.get_position()
        fw, fh = fig.get_size_inches()
        ax_w_in, ax_h_in = bb.width * fw, bb.height * fh
        if ax_w_in <= 0 or ax_h_in <= 0:
            raise ValueError("axes has zero size; set the figure size before converting")
        return (self.anchor[0] + dx_in / ax_w_in, self.anchor[1] + dy_in / ax_h_in)

    def anchors(self) -> dict:
        """Named anchors, in this track's own inch frame."""
        lo, hi = self.span(self.bp0, self.bp1)
        return dict(start=(self.edge(self.bp0), 0.0), end=(self.edge(self.bp1 + 1), 0.0),
                    left=(lo, 0.0), right=(hi, 0.0), centre=((lo + hi) / 2.0, 0.0),
                    above=((lo + hi) / 2.0, 0.0), below=((lo + hi) / 2.0, 0.0))

    def __repr__(self):
        return (f"Track(bp {self.bp0}..{self.bp1}, n={self.n}, width={self.width:.3f} in, "
                f"bpw={self.bpw:.5f} in, direction={self.direction:+d}, "
                f"register={self.register!r})")


def _lw(tier="secondary"):
    return STROKE[tier]


def _line(ax, tr, pts, *, color=INK, lw=None, zorder=5, ls="solid", cap="round", alpha=1.0):
    lw = _lw() if lw is None else lw
    (xs, ys) = zip(*pts)
    return ax.plot(xs, ys, transform=tr, color=color, lw=lw, zorder=zorder, linestyle=ls,
                   solid_capstyle=cap, clip_on=False, alpha=alpha)[0]


def _text_w_in(s, pt):
    """Approximate rendered width of `s` at `pt`, in inches."""
    return len(str(s)) * pt * 0.60 / 72.0


def _label_in_body(ax, tr, x, y, s, body_w, *, pt=None, color=None, zorder=6, below_dy=0.0):
    """Put `s` inside the body if it FITS, otherwise immediately below it. Returns True if inside."""
    pt = TYPE["in_figure"] if pt is None else pt
    color = INK if color is None else color
    fits = _text_w_in(s, pt) <= body_w * 0.92
    ax.text(x, y if fits else y - below_dy, s, transform=tr, ha="center",
            va="center" if fits else "top", fontsize=pt, color=color, zorder=zorder,
            clip_on=False)
    return fits


def _resolve(track, detail, rungs, polarity_form):
    """Fill unset drawing options from the track's REGISTER rather than from a hidden default."""
    spec = register_spec(track.register)
    return (spec["detail"] if detail is None else detail,
            spec["rungs"] if rungs is None else rungs,
            spec["polarity"] if polarity_form is None else polarity_form)


RAIL_GAP = 0.155

LETTER_ROW_GAP = 0.20


def duplex(track, bp_a, bp_b, *, style="rails", detail=None, rungs=None, y=0.0, spans=(),
           edits=(), gap=None, color=INK, rung_color=None, tier="primary", zorder=None,
           comb_every=None, seq_top=None, seq_bot=None, polarity_form=None, label=None,
           glow=None, bp_per_turn=None, amp=None):
    """Double-stranded nucleic acid along `track`, from base `bp_a` to `bp_b` inclusive.

      detail 0  two parallel rails, no rungs                    -- workflow / overview scale
      detail 1  rungs between the rails, in ONE OF TWO forms    -- `rungs=`, see below
      detail 2  two rows of monospace letters, no rungs         -- when bases must be READ
      detail 3  beads on a ribbon                               -- when PAIRING is the claim

    🔴 `rungs=` PICKS BETWEEN TWO NON-INTERCHANGEABLE FORMS AND THE DIFFERENCE IS SILENT:

    `spans` is `[(bp0, bp1, colour), ...]`. At detail 1 with `rungs="ladder"` a span recolours the
    RAILS AND THE RUNGS over its extent, which is E9's device and costs no extra ink. `edits` is
    `[(bp, colour), ...]` and colours single rungs -- it REQUIRES the ladder and says so.

    Returns a dict of named anchors in the track's inch frame.
    
    """
    gap = RAIL_GAP if gap is None else gap
    if style == "helix":
        if edits:
            raise ValueError(
                "duplex(style='helix') cannot take edits=: a helix's rungs are a texture whose "
                "count comes from the drawn period, not from the sequence, so a single base pair is "
                "not addressable on one. Use style='rails', detail=1, rungs='ladder' for a per-base "
                "claim.")
        return helix(track, bp_a, bp_b, y=y, amp=amp, bp_per_turn=bp_per_turn, color=color,
                     tier=tier, zorder=zorder, glow=glow, label=label)
    if style != "rails":
        raise ValueError(f"duplex(style=…) must be 'rails' or 'helix', not {style!r}")
    detail, rungs, polarity_form = _resolve(track, detail, rungs, polarity_form)
    ax, tr = track.ax, track.tr
    zorder = track.zorder if zorder is None else zorder
    if rungs not in ("comb", "ladder"):
        raise ValueError(f"rungs must be 'comb' or 'ladder', not {rungs!r}")
    if edits and rungs != "ladder":
        raise ValueError(
            "edits= colours individual base pairs, which only a LADDER can address. A comb tick is "
            "decorative density, not a base pair.")
    if detail >= 2:
        gap = max(gap, LETTER_ROW_GAP)
    lo, hi = track.span(bp_a, bp_b)
    yt, yb = y + gap / 2.0, y - gap / 2.0
    lw = _lw(tier)

    def _span_colour(bp):
        for s0, s1, c in spans:
            if min(s0, s1) <= bp <= max(s0, s1):
                return c
        return None

    if detail >= 2:
        pass
    elif spans and detail <= 1:
        for bp in range(bp_a, bp_b + 1):
            c = _span_colour(bp) or color
            e0, e1 = track.span(bp, bp)
            for yy in (yt, yb):
                _line(ax, tr, [(e0, yy), (e1, yy)], color=c, lw=lw, zorder=zorder, cap="butt")
    else:
        for yy in (yt, yb):
            _line(ax, tr, [(lo, yy), (hi, yy)], color=color, lw=lw, zorder=zorder)

    if detail == 1:
        rc = rung_color or color
        if rungs == "ladder":
            for bp in range(bp_a, bp_b + 1):
                c = dict(edits).get(bp) or _span_colour(bp) or rc
                xx = track.x(bp)
                _line(ax, tr, [(xx, yb), (xx, yt)], color=c,
                      lw=_lw("hairline") if c == rc else _lw("secondary"),
                      zorder=zorder + 0.1, cap="butt")
        else:
            step = comb_every or max(1, int(round(0.030 / max(track.bpw, 1e-9))))
            t = gap * 0.34
            for bp in range(bp_a, bp_b + 1, step):
                xx = track.x(bp)
                _line(ax, tr, [(xx, yt), (xx, yt - t)], color=rc, lw=_lw("hairline"),
                      zorder=zorder + 0.1, cap="butt")
                _line(ax, tr, [(xx, yb), (xx, yb + t)], color=rc, lw=_lw("hairline"),
                      zorder=zorder + 0.1, cap="butt")

    if detail >= 2:
        _bases(track, bp_a, bp_b, seq_top, seq_bot, y, gap, detail, spans, edits, zorder)

    if polarity_form and polarity_form != "none":
        polarity(track, bp_a, bp_b, form=polarity_form, y=y, gap=gap)
    if label:
        text_in(ax, *track.anchor, (lo + hi) / 2.0, y + gap / 2.0 + 0.10, label,
                coords=track.coords, size=TYPE["in_figure"], color=RULE)

    return dict(left=(lo, y), right=(hi, y), centre=((lo + hi) / 2.0, y),
                top=((lo + hi) / 2.0, yt), bottom=((lo + hi) / 2.0, yb),
                top_left=(lo, yt), top_right=(hi, yt),
                bottom_left=(lo, yb), bottom_right=(hi, yb),
                rail_top=yt, rail_bottom=yb, gap=gap)


LETTER_MIN_IN = 0.0737
BEAD_MIN_IN = 0.1475


def _bases(track, bp_a, bp_b, seq_top, seq_bot, y, gap, detail, spans, edits, zorder):
    """Letters (detail 2) or beads on a ribbon (detail 3), with the auto-degrade the floor forces."""
    ax, tr = track.ax, track.tr
    need = BEAD_MIN_IN if detail >= 3 else LETTER_MIN_IN
    if track.bpw < need:
        duplex(track, bp_a, bp_b, detail=1, rungs="ladder", y=y, gap=gap, spans=spans,
               edits=edits, zorder=zorder, polarity_form="none")
        return
    size = G.pt_for_cap_mm()
    for bp in range(bp_a, bp_b + 1):
        i = bp - bp_a
        top = (seq_top or "")[i:i + 1] or "N"
        bot = (seq_bot or "")[i:i + 1] or _complement(top)
        xx = track.x(bp)
        if detail >= 3:
            for yy, ch in ((y + gap / 2.0, top), (y - gap / 2.0, bot)):
                ax.add_patch(Circle((xx, yy), track.bpw * 0.38, transform=tr, clip_on=False,
                                    fc=INK, ec=INK, lw=_lw("hairline"), zorder=zorder + 0.2))
                ax.text(xx, yy, ch, transform=tr, ha="center", va="center", fontsize=size,
                        color=PAPER, zorder=zorder + 0.3, clip_on=False, family="monospace")
            _line(ax, tr, [(xx, y - gap / 2.0), (xx, y + gap / 2.0)], color=RULE,
                  lw=_lw("hairline"), zorder=zorder)
        else:
            for yy, ch in ((y + gap / 2.0, top), (y - gap / 2.0, bot)):
                ax.text(xx, yy, ch, transform=tr, ha="center", va="center", fontsize=size,
                        color=INK, zorder=zorder + 0.2, clip_on=False, family="monospace")


def _complement(b):
    return {"A": "T", "T": "A", "G": "C", "C": "G", "U": "A", "N": "N"}.get(b.upper(), "N")


def strand(track, bp_a, bp_b, *, kind="ssdna", y=0.0, detail=None, color=None, fold=None,
           tier=None, zorder=None, ticks=True, label=None):
    """A SINGLE polymer -- ssDNA, RNA, or one half of a hybrid.

      "ssdna"  ONE comb -- a rail with ticks on one side and no partner; the stroke steps DOWN one
               tier and the value lightens. At detail 0 it is one bare rail with nothing opposite.
      "rna"    one comb whose ticks are SHORTER and MORE WIDELY SPACED, drawn as a FOLDED path,
               stroke one tier lighter than DNA. At base level the letters use `U`, never `T`.
      "rnadna" two combs of different tick form paired together.

    `color` is the molecule's IDENTITY, which is orthogonal: `donor` #56B4E9 is the same blue whether
    the donor is ssDNA or a duplex. Only the comb count and the stroke tier change. This is also how
    msr is drawn -- see module docstring section 4.
    
    """
    ax, tr = track.ax, track.tr
    zorder = track.zorder if zorder is None else zorder
    detail = register_spec(track.register)["detail"] if detail is None else detail
    lo, hi = track.span(bp_a, bp_b)
    if kind not in ("ssdna", "rna", "rnadna", "dsdna"):
        raise ValueError(f"unknown strand kind {kind!r}")
    tier = tier or ("secondary" if kind in ("ssdna", "rna") else "primary")
    color = color if color is not None else RULE
    lw = _lw(tier)

    amp = 0.0 if fold is None else fold
    if kind == "rna" and fold is None:
        amp = 0.030
    n = max(24, int((hi - lo) / 0.02))
    xs = np.linspace(lo, hi, n)
    if amp:
        ys = y + amp * np.sin(np.linspace(0, 3.0 * math.pi, n))
    else:
        ys = np.full(n, y)
    _line(ax, tr, list(zip(xs, ys)), color=color, lw=lw, zorder=zorder)

    if ticks and detail >= 1:
        t = 0.020 if kind != "rna" else 0.013
        step_in = 0.030 if kind != "rna" else 0.048
        k = max(1, int(round(step_in / max(track.bpw, 1e-9))))
        for bp in range(bp_a, bp_b + 1, k):
            xx = track.x(bp)
            yy = float(np.interp(xx, xs, ys)) if amp else y
            _line(ax, tr, [(xx, yy), (xx, yy - t)], color=color, lw=_lw("hairline"),
                  zorder=zorder + 0.1, cap="butt")
    if label:
        text_in(ax, *track.anchor, (lo + hi) / 2.0, y + 0.09, label, coords=track.coords,
                size=TYPE["in_figure"], color=INK)
    return dict(left=(lo, y), right=(hi, y), centre=((lo + hi) / 2.0, y),
                five_p=(lo, y), three_p=(hi, y))


def region_arc(track, bp_a, bp_b, color, *, y=0.0, side="top", offset=0.016, tier="emphasis",
               zorder=None, label=None):
    """    A thick arc in the semantic token laid just INSIDE the strand's own path over the span. The
    backbone underneath stays structural. Two arcs of different tokens can sit on one backbone,
    which a recolour cannot do -- that is the whole point of the device.
    
    """
    ax, tr = track.ax, track.tr
    zorder = (track.zorder if zorder is None else zorder) + 0.4
    lo, hi = track.span(bp_a, bp_b)
    yy = y + (offset if side == "top" else -offset)
    _line(ax, tr, [(lo, yy), (hi, yy)], color=color, lw=_lw(tier), zorder=zorder, cap="butt")
    if label:
        text_in(ax, *track.anchor, (lo + hi) / 2.0, yy + (0.055 if side == "top" else -0.055),
                label, coords=track.coords, size=TYPE["in_figure"], color=INK,
                va="bottom" if side == "top" else "top")
    return dict(left=(lo, yy), right=(hi, yy), centre=((lo + hi) / 2.0, yy))


def span_bar(track, bp_a, bp_b, label, color, *, y=0.0, side="above", offset=0.11,
             channel="bar", height=None, zorder=None):
    """    `channel="bar"` is a flat filled rounded rectangle spanning EXACTLY the bases it names, its
    label set inside in paper white. `channel="underline"` is a short heavy rule beneath the run.
    Both channels exist because features overlap and one channel cannot show two.
    """
    ax, tr = track.ax, track.tr
    zorder = (track.zorder if zorder is None else zorder) + 0.5
    lo, hi = track.span(bp_a, bp_b)
    fs = TYPE["in_figure"]
    if channel != "underline" and height is None:
        height = max(0.075, 1.40 * fs / 72.0)
        offset = offset - 0.0375 + height / 2.0
    yy = y + (offset if side == "above" else -offset)
    if channel == "underline":
        _line(ax, tr, [(lo, yy), (hi, yy)], color=color, lw=_lw("emphasis"), zorder=zorder,
              cap="butt")
        text_in(ax, *track.anchor, (lo + hi) / 2.0, yy + (0.05 if side == "above" else -0.05),
                label, coords=track.coords, size=TYPE["in_figure"], color=INK,
                va="bottom" if side == "above" else "top")
    else:
        from matplotlib.patches import FancyBboxPatch
        p = FancyBboxPatch((lo, yy - height / 2.0), max(hi - lo, 1e-4), height,
                           boxstyle="round,pad=0,rounding_size=0.018", transform=tr,
                           fc=color, ec="none", zorder=zorder, clip_on=False)
        ax.add_patch(p)
        exempt(p)
        if 0.60 * fs / 72.0 * len(label) <= (hi - lo) - 0.04:
            ax.text((lo + hi) / 2.0, yy, label, transform=tr, ha="center", va="center",
                    fontsize=fs, color=PAPER, zorder=zorder + 0.1, clip_on=False)
        else:
            up = side == "above"
            ax.text((lo + hi) / 2.0, yy + (height / 2.0 + 0.03) * (1 if up else -1), label, transform=tr,
                    ha="center", va="bottom" if up else "top", fontsize=fs, color=INK,
                    zorder=zorder + 0.1, clip_on=False)
    return dict(left=(lo, yy), right=(hi, yy), centre=((lo + hi) / 2.0, yy))


def polarity(track, bp_a, bp_b, *, form="stacked", y=0.0, gap=None, zorder=None,
             top_5p_left=True):
    ""
    gap = RAIL_GAP if gap is None else gap
    if form not in ("none", "corners", "stacked", "inline"):
        raise ValueError(f"unknown polarity form {form!r}")
    if form == "none":
        return {}
    ax, tr = track.ax, track.tr
    zorder = (track.zorder if zorder is None else zorder) + 0.6
    lo, hi = track.span(bp_a, bp_b)
    half = max(gap / 2.0, 0.060)
    yt, yb = y + half, y - half
    d = 0.035
    s = TYPE["in_figure"]
    out = {}
    if form == "inline":
        ax.text(lo - d, y, "5'-", transform=tr, ha="right", va="center", fontsize=s, color=INK,
                clip_on=False, zorder=zorder)
        ax.text(hi + d, y, "-3'", transform=tr, ha="left", va="center", fontsize=s, color=INK,
                clip_on=False, zorder=zorder)
        return dict(five_p=(lo - d, y), three_p=(hi + d, y))
    pairs = [(lo - d, yt, "5'" if top_5p_left else "3'", "right"),
             (hi + d, yt, "3'" if top_5p_left else "5'", "left"),
             (lo - d, yb, "3'" if top_5p_left else "5'", "right"),
             (hi + d, yb, "5'" if top_5p_left else "3'", "left")]
    for xx, yy, txt, ha in pairs:
        ax.text(xx, yy, txt, transform=tr, ha=ha, va="center", fontsize=s, color=INK,
                clip_on=False, zorder=zorder)
        out[f"{txt}_{'l' if ha == 'right' else 'r'}_{'t' if yy > y else 'b'}"] = (xx, yy)
    return out


def break_(track, bp, *, kind="nick", y=0.0, gap=None, side="top", agent=None, zorder=None,
           label=None, stagger=0):
    """    🔴 A BREAK IS AT AN EDGE. `bp` names the base the break falls BEFORE, so the glyph is drawn at
    ``track.edge(bp)``; that is the same place `alignment_grammar` puts its Cas9 cut.

      "nick"     a small gap in ONE rail PLUS a coloured ARROW pointing at it from outside the
                 duplex -- above for the top strand, below for the bottom. 🔴 The gap alone is
                 invisible at figure size, which is why the arrow is not optional. Two independent
                 derivations (Anzalone 2019 and Bai 2026 draw the identical glyph).
      "blunt"    both rails stop at the same x, with a vertical gap between the halves.
      "resect"   the two rails stop at DIFFERENT x. `stagger` is the offset in base pairs. Identical
                 drawing to "stagger"; the LABEL is what distinguishes them.
      "flap"     a short LADDER SEGMENT lifting off at ~45 degrees whose rungs have NO OPPOSING
                 RAIL -- unpaired ticks are the ssDNA cue. 3' and 5' flaps are drawn identically and
                 told apart by the label, never by the shape.
      "junction" 🔴 NO GLYPH AT ALL -- a COLOUR CHANGE inside the strand. Returns the anchor so the
                 caller can switch colour there.
    """
    gap = RAIL_GAP if gap is None else gap
    ax, tr = track.ax, track.tr
    zorder = (track.zorder if zorder is None else zorder) + 0.7
    xe = track.edge(bp)
    yt, yb = y + gap / 2.0, y - gap / 2.0
    col = agent or EDIT
    out = dict(at=(xe, y), top=(xe, yt), bottom=(xe, yb))

    if kind == "cleave":
        raise NotImplementedError(
            "break_(kind='cleave') is deferred: it needs the tokens `cleave_full` and "
            "`cleave_partial`, which have not been searched or CVD-validated. Borrowing `recruit` "
            "or `barcode` would give one hue two molecular identities -- the exact defect the "
            "`pam` token was created to fix. See 1293 section 2.6.")

    if kind == "nick":
        yy = yt if side == "top" else yb
        w = min(track.bpw * 0.9, 0.020)
        ax.add_patch(Rectangle((xe - w / 2.0, yy - 0.004), w, 0.008, transform=tr, fc=PAPER,
                               ec="none", zorder=zorder, clip_on=False))
        d = 0.075
        y0 = yy + (d if side == "top" else -d)
        from matplotlib.patches import FancyArrowPatch
        a = FancyArrowPatch((xe, y0), (xe, yy + (0.014 if side == "top" else -0.014)),
                            arrowstyle="-|>", mutation_scale=7.0, lw=_lw("secondary"),
                            color=col, transform=tr, clip_on=False, zorder=zorder + 0.1,
                            shrinkA=0, shrinkB=0)
        ax.add_patch(a)
        out["arrow"] = (xe, y0)
    elif kind == "blunt":
        w = min(track.bpw * 1.2, 0.026)
        ax.add_patch(Rectangle((xe - w / 2.0, yb - 0.006), w, gap + 0.012, transform=tr,
                               fc=PAPER, ec="none", zorder=zorder, clip_on=False))
    elif kind in ("resect", "stagger"):
        off = abs(stagger) * track.bpw
        x0 = min(xe, xe + track.direction * off)
        gone_top = (side == "top")
        pad = _lw("primary") / 72.0 / 2.0 + 0.002
        y_lo, h = (yb + pad, (yt - yb)) if gone_top else (yb - pad, (yt - yb))
        ax.add_patch(Rectangle((x0, y_lo), off, h, transform=tr, fc=PAPER, ec="none",
                               zorder=zorder, clip_on=False))
        keep = yb if gone_top else yt
        k = max(1, int(round(0.030 / max(track.bpw, 1e-9))))
        for bp_i in range(bp, bp + abs(stagger), k):
            xx = track.x(bp_i)
            if min(x0, x0 + off) <= xx <= max(x0, x0 + off):
                _line(ax, tr, [(xx, keep), (xx, keep + (0.026 if gone_top else -0.026))],
                      color=RULE, lw=_lw("hairline"), zorder=zorder + 0.1, cap="butt")
        out["overhang"] = (x0 + off / 2.0, keep)
        out["bottom"] = (xe + track.direction * off, yb)
    elif kind == "flap":
        yy = yt if side == "top" else yb
        sgn = 1.0 if side == "top" else -1.0
        L = 0.16
        dx, dy = L * math.cos(math.radians(45)), sgn * L * math.sin(math.radians(45))
        _line(ax, tr, [(xe, yy), (xe + dx, yy + dy)], color=INK, lw=_lw("primary"),
              zorder=zorder)
        for f in (0.30, 0.50, 0.70, 0.90):
            bx, by = xe + dx * f, yy + dy * f
            nx, ny = -dy / L * 0.020, dx / L * 0.020
            _line(ax, tr, [(bx, by), (bx + nx, by + ny)], color=RULE, lw=_lw("hairline"),
                  zorder=zorder, cap="butt")
        out["tip"] = (xe + dx, yy + dy)
    elif kind == "junction":
        pass
    else:
        raise ValueError(f"unknown break kind {kind!r}")

    if label:
        lx, ly = out.get("tip") or out.get("arrow") or (xe, yt if side == "top" else yb)
        text_in(ax, *track.anchor, lx, ly + (0.045 if side == "top" else -0.045), label,
                coords=track.coords, size=TYPE["in_figure"], color=INK,
                va="bottom" if side == "top" else "top")
    return out


def elide(track, bp_a, bp_b, *, y=0.0, gap=None, form="dots", per_strand=True, zorder=None):
    """    🔴 It must be visually distinct from a rung: DOTS RUN ALONG the strand, RUNGS RUN ACROSS it.
    Structural -- it never takes a semantic token.
    
    """
    gap = RAIL_GAP if gap is None else gap
    ax, tr = track.ax, track.tr
    zorder = track.zorder if zorder is None else zorder
    lo, hi = track.span(bp_a, bp_b)
    ys = (y + gap / 2.0, y - gap / 2.0) if per_strand else (y,)
    if form == "ellipsis":
        ax.text((lo + hi) / 2.0, y, "•••", transform=tr, ha="center", va="center",
                fontsize=TYPE["in_figure"], color=RULE, zorder=zorder, clip_on=False)
    else:
        for yy in ys:
            _line(ax, tr, [(lo, yy), (hi, yy)], color=RULE, lw=_lw("secondary"), zorder=zorder,
                  ls=(0, (1.2, 1.8)))
    return dict(left=(lo, y), right=(hi, y), centre=((lo + hi) / 2.0, y))


def mark(track, bp, letter, color, *, y=0.0, side="top", offset=0.09, d=0.055, zorder=None):
    ""
    ax, tr = track.ax, track.tr
    zorder = (track.zorder if zorder is None else zorder) + 0.8
    xx = track.x(bp)
    yy = y + (offset if side == "top" else -offset)
    ax.add_patch(Circle((xx, yy), d / 2.0, transform=tr, fc=PAPER, ec=color,
                        lw=_lw("secondary"), zorder=zorder, clip_on=False))
    ax.text(xx, yy, letter, transform=tr, ha="center", va="center",
            fontsize=TYPE["in_figure"], color=INK, zorder=zorder + 0.1, clip_on=False)
    _line(ax, tr, [(xx, y + (0.5 if side == "top" else -0.5) * RAIL_GAP),
                   (xx, yy - (d / 2.0 if side == "top" else -d / 2.0))],
          color=color, lw=_lw("hairline"), zorder=zorder - 0.1)
    return dict(at=(xx, yy), stem=(xx, y))


DOME_FACE_CUT = 0.35

FACE_LANDING = 0.45


def body_overlap(seg, *outlines, samples=240):
    """Fraction of a straight connection `seg` that falls INSIDE any body outline. 0.0 when clear.

    `seg` is ``((x0, y0), (x1, y1))`` and each outline a sequence of points, all in ONE frame (the
    `at` frame the glyph was drawn in). Endpoints are excluded: a connection is SUPPOSED to touch the
    outline it lands on.
    
    """
    (x0, y0), (x1, y1) = seg
    hits = 0
    for i in range(1, samples):
        t = i / float(samples)
        px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        if any(_point_in(o, px, py) for o in outlines):
            hits += 1
    return hits / float(samples)


def _point_in(poly, x, y):
    """Even-odd point-in-polygon. Plain arithmetic: it runs inside tests and gates, not on a canvas."""
    inside = False
    n = len(poly)
    for i in range(n):
        ax0, ay0 = poly[i]
        ax1, ay1 = poly[(i + 1) % n]
        if (ay0 > y) != (ay1 > y) and x < (ax1 - ax0) * (y - ay0) / ((ay1 - ay0) or 1e-18) + ax0:
            inside = not inside
    return inside


def protein(ax, x, y, d=0.42, *, feature="dome", label=None, fill=None, lw=None, coords="axes",
            zorder=6, tr=None, at=(0.0, 0.0), state="a", angle=0.0, shadow=0.030, sublabel=None):
    """A protein body: a NAMEABLE GEOMETRIC PRIMITIVE CARRYING ONE FUNCTIONAL FEATURE.

    Protein is STRUCTURE unless the protein is the subject: `fill` defaults to the structural
    neutral and a caller must pass a token deliberately.
    
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    fill = MUTE if fill is None else fill
    lw = _lw("secondary") if lw is None else lw
    ax0, ay0 = at
    r = d / 2.0
    th = np.linspace(0, 2 * math.pi, 240)
    amp = 0.0

    if feature == "dome":
        pts = [(r * math.cos(t), r * math.sin(t)) for t in th if math.sin(t) > -DOME_FACE_CUT]
        flat = -DOME_FACE_CUT * r
        pts = [(px, max(py, flat)) for px, py in pts]
        if state == "b":
            pts = [(px, py + (0.16 * r if (py <= flat + 1e-9 and abs(px) < 0.42 * r) else 0.0))
                   for px, py in pts]
    elif feature == "notch":
        cut = math.radians(34 if state == "a" else 60)
        keep = [t for t in th if not (-cut / 2 < ((t + math.pi) % (2 * math.pi) - math.pi) < cut / 2)]
        pts = [(r * math.cos(t), r * math.sin(t)) for t in keep] + [(0.18 * r, 0.0)]
    elif feature == "slab":
        w, h = r * 1.7, r * 0.92
        pts = [(-w, -h), (w, -h), (w, h), (-w, h)]
        if state == "b":
            pts = [(-w, -h), (w, -h), (w, h * 0.42), (w * 0.45, h), (-w, h)]
    elif feature == "ovoid":
        amp = 0.10 if state == "a" else 0.17
        pts = []
        for t in th:
            dt = (t + math.pi) % (2 * math.pi) - math.pi
            k = 1.0 + amp * math.exp(-(dt / 0.22) ** 2)
            pts.append((k * r * math.cos(t), 0.82 * k * r * math.sin(t)))
    else:
        raise ValueError(f"unknown protein feature {feature!r}; expected dome, notch, slab or ovoid")

    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    pts = [(ax0 + px * ca - py * sa, ay0 + px * sa + py * ca) for px, py in pts]
    xs, ys = zip(*pts)
    G._body(ax, lambda: Polygon(pts, closed=True), tr, fill=fill, zorder=zorder, lw=lw,
            shadow=shadow, gloss=(ax0 - 0.28 * r, ay0 + 0.30 * r, 0.72 * r, 0.44 * r, -26),
            extent=(min(xs), max(xs), min(ys), max(ys)))
    if label:
        _label_in_body(ax, tr, ax0, ay0 + (0.06 * r if sublabel else 0.0), label,
                       (max(xs) - min(xs)), zorder=zorder + 0.4, below_dy=r * 1.12)
    if sublabel:
        ax.text(ax0, ay0 - 0.30 * r, sublabel, transform=tr, ha="center", va="center",
                fontsize=TYPE["annotation"], color=RULE, zorder=zorder + 0.4, clip_on=False)
    if feature == "ovoid":
        fx, fy, half, nrm = (1.0 + amp) * r, 0.0, 0.0, (1.0, 0.0)
    elif feature == "dome":
        fx, fy, half, nrm = 0.0, -DOME_FACE_CUT * r, math.sqrt(1.0 - DOME_FACE_CUT ** 2) * r, (0.0, -1.0)
    else:
        fx, fy, half, nrm = 0.0, -DOME_FACE_CUT * r, 0.0, (0.0, -1.0)
    feat = (ax0 + fx * ca - fy * sa, ay0 + fx * sa + fy * ca)
    fdir = (nrm[0] * ca - nrm[1] * sa, nrm[0] * sa + nrm[1] * ca)
    return dict(centre=(ax0, ay0), top=(ax0, ay0 + r), bottom=(ax0, ay0 - r),
                left=(ax0 - r, ay0), right=(ax0 + r, ay0),
                face=feat, feature=feat, r=r,
                face_half=half, face_dir=fdir, outline=tuple(pts))


def _clamp_outline(r, *, direction, half, depth, ovality=0.90, lobe=0.30, n=220):
    """A bilobed CLAMP with an open jaw — the Cas9 silhouette, and NOT a budded cell.

    It was never taste. `cas9()` called `glyphs._union_outline(r1, 0.66*r1, 0.34, 0.96)` and
    `glyphs._body()` — the SAME primitive and the SAME five-step fill stack as `glyphs.yeast()`. At
    the size a mechanism panel uses (0.30 in, `label=None`, `lobes=False`) the two silhouettes were
    identical, and hue cannot separate them: the colour contract says structure is never coloured,
    so both are neutral on the page.

    ⚠️ **The first fix attempt failed, and how it failed is the lesson.** Pulling the radius in with
    a Gaussian — the negative of `_union_outline`'s `bumps` — deepened the notch the union ALREADY
    has between mother and bud. A deeper neck reads as a cell budding harder. **Any edit that keeps
    the two-disc outline makes the glyph MORE cell-like, not less.** The outline itself is the bug.

    `direction` points the jaw (degrees), `half` is its half-angle, `depth` is the fraction of the
    radius left at the apex, and `lobe` is a gentle swell on the far side that keeps the bilobed
    read the REC and NUC labels depend on.
    
    """
    d0 = math.radians(direction)
    hw = math.radians(half)

    swells = ((d0 + hw + 0.80, 0.30, 0.72),
              (d0 - hw - 0.55, 0.15, 0.55),
              (d0 + math.pi, lobe, 1.05))

    def rim(t):
        rr = 1.0
        for centre, amp, width in swells:
            dth = (t - centre + math.pi) % (2 * math.pi) - math.pi
            rr += amp * math.exp(-(dth / width) ** 2)
        return r * rr

    ts = np.linspace(d0 + hw, d0 + 2 * math.pi - hw, n)
    pts = [(rim(t) * math.cos(t), rim(t) * math.sin(t) * ovality) for t in ts]
    apex = (depth * r * math.cos(d0), depth * r * math.sin(d0) * ovality)
    pts.append(apex)

    def at_angle(deg_off, frac):
        """A point inside the body, `frac` of the local rim radius along `d0 + deg_off`."""
        t = d0 + math.radians(deg_off)
        return (frac * rim(t) * math.cos(t), frac * rim(t) * math.sin(t) * ovality)

    return pts, apex, dict(rec=at_angle(math.degrees(hw) + 46.0, 0.56),
                           nuc=at_angle(-math.degrees(hw) - 32.0, 0.52),
                           body=at_angle(180.0, 0.42))


def cas9(ax, x, y, d=0.62, *, state="nuclease", label="Cas9", lobes=False, flip=False,
         fill=None, lw=None, coords="axes", zorder=6, tr=None, at=(0.0, 0.0), shadow=0.030,
         addition=None, cleft=0.20, cleft_half=44.0, gloss=False, dna="above"):
    """    This is the same construction `glyphs.yeast()` uses for mother and bud, so the primitive is
    imported rather than re-derived -- `glyphs._union_outline`. Two overlapping circles drawn as two
    circles leave an internal arc and read as a figure-eight of bubbles.

    🔴 `flip` IS NOT OPTIONAL. Two independent exemplars draw the antisense complex as a 180-degree
    mirror, and an antisense guide is the normal case in half of our designs. A hardcoded
    orientation mis-draws them while producing a picture that looks completely plausible -- the same
    family as `protospacer-seed-inverts-on-antisense`.
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    fill = PROTEIN_BODY if fill is None else fill
    lw = _lw("secondary") if lw is None else lw
    ax0, ay0 = at
    sgn = -1.0 if flip else 1.0
    r1 = d / 2.0
    r2 = r1 * 0.66
    if dna not in ("above", "below"):
        raise ValueError("cas9(dna=): expected 'above' or 'below', got %r. The jaw must face the "
                         "substrate and this glyph cannot infer which side it is on." % (dna,))
    _dir = (90.0 - sgn * 14.0) if dna == "above" else (-90.0 + sgn * 14.0)
    pv, _apex, _lobes = _clamp_outline(r1, direction=_dir, half=cleft_half, depth=cleft)
    verts = [(ax0 + px, ay0 + py) for px, py in pv]
    xs, ys = zip(*verts)
    G._body(ax, lambda: Polygon(verts, closed=True), tr, fill=fill, zorder=zorder, lw=lw,
            shadow=shadow, edge=INK,
            gloss=((ax0 - 0.30 * r1, ay0 + 0.34 * r1, 0.80 * r1, 0.48 * r1, -28)
                   if gloss else None),
            extent=(min(xs), max(xs), min(ys), max(ys)))
    nuc = (ax0 + _lobes["nuc"][0], ay0 + _lobes["nuc"][1])
    rec = (ax0 + _lobes["rec"][0], ay0 + _lobes["rec"][1])
    if lobes:
        for xy, nm in ((rec, "REC"), (nuc, "NUC")):
            ax.text(xy[0], xy[1], nm, transform=tr, ha="center", va="center",
                    fontsize=TYPE["annotation"], color=RULE, zorder=zorder + 0.4, clip_on=False)
        if label:
            ax.text(ax0, ay0 + r1 * 1.14, label, transform=tr, ha="center", va="bottom",
                    fontsize=TYPE["in_figure"], color=INK, zorder=zorder + 0.4, clip_on=False)
    elif label:
        ax.text(rec[0], rec[1], label, transform=tr, ha="center", va="center",
                fontsize=TYPE["in_figure"], color=INK, zorder=zorder + 0.4, clip_on=False)
    if state in ("dead", "nickase_hnh", "nickase_ruvc"):
        ax.add_patch(Circle((nuc[0], nuc[1] - 0.22 * r1), 0.055 * d, transform=tr, fc=RULE,
                            ec=INK, lw=_lw("hairline"), zorder=zorder + 0.5, clip_on=False))
    out = dict(centre=(ax0, ay0), notch=(ax0 + sgn * 0.30 * r1, ay0 - 0.52 * r1),
               dna_in=(ax0 - sgn * r1 * 1.05, ay0 - 0.62 * r1),
               dna_out=(ax0 + sgn * r1 * 1.35, ay0 - 0.62 * r1),
               rec=rec, nuc=nuc, top=(ax0, ay0 + r1), bottom=(ax0, ay0 - r1),
               effector_n=(ax0 - sgn * 0.86 * r1, ay0 + 0.52 * r1),
               effector_c=(ax0 + sgn * 0.92 * r1, ay0 + 0.10 * r1), r=r1,
               jaw=dict(depth=cleft, half_angle=cleft_half, direction=_dir, dna=dna,
                        apex=(ax0 + _apex[0], ay0 + _apex[1])), gloss=bool(gloss))
    if addition is not None:
        name, token = addition
        axy = out["effector_c"]
        pos = (axy[0] + sgn * 0.86 * r1, axy[1] + 0.86 * r1)
        _line(ax, tr, [axy, pos], color=RULE, lw=_lw("secondary"), zorder=zorder - 0.1)
        protein(ax, 0, 0, d * 0.46, feature="dome", fill=token, tr=tr, at=pos,
                zorder=zorder + 0.2, shadow=0.0)
        ax.text(pos[0] + sgn * (d * 0.23 + 0.045), pos[1], name, transform=tr,
                ha="left" if sgn > 0 else "right", va="center", fontsize=TYPE["annotation"],
                color=INK, zorder=zorder + 0.5, clip_on=False)
        out["addition"] = pos
    return out


def base_editor(ax, x, y, d=0.62, **kw):
    ""
    kw.setdefault("label", "nCas9")
    kw.setdefault("state", "nickase_hnh")
    return cas9(ax, x, y, d, addition=("deaminase", DOMAIN), **kw)


def prime_editor(ax, x, y, d=0.62, **kw):
    """nCas9 + EXACTLY ONE addition: the reverse transcriptase. Same parent, same rule."""
    kw.setdefault("label", "nCas9")
    kw.setdefault("state", "nickase_hnh")
    return cas9(ax, x, y, d, addition=("RT", DOMAIN), **kw)


def guide_rna(ax, x, y, w=0.70, *, loops=3, style="serpentine", spacer_frac=0.34,
              spacer_color=None, scaffold_color=None, modified=(), coords="axes", zorder=6,
              tr=None, at=(0.0, 0.0), label=None, lw=None):
    """    Parts are COLOUR SEGMENTS ALONG THAT ONE LINE: the spacer is the 5'-most run in `guide`; the
    scaffold is the folded remainder in `rule`. A stem-loop is two paired antiparallel runs closed
    by a 180-degree bend.

    `modified` is `[(loop_index, token), ...]` and recolours an ENGINEERED stem-loop -- which is how
    the MAGESTIC MS2 aptamer is drawn, in `recruit`, on the same backbone as everything else.
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    lw = _lw("secondary") if lw is None else lw
    ax0, ay0 = at
    spacer_color = GUIDE if spacer_color is None else spacer_color
    scaffold_color = RULE if scaffold_color is None else scaffold_color
    mod = dict(modified)
    sx = w * spacer_frac
    _line(ax, tr, [(ax0, ay0), (ax0 + sx, ay0)], color=spacer_color, lw=_lw("primary"),
          zorder=zorder)
    out = dict(five_p=(ax0, ay0), spacer_end=(ax0 + sx, ay0), loop_tops=[])
    seg = (w - sx) / max(loops, 1)
    px = ax0 + sx
    for i in range(loops):
        c = mod.get(i, scaffold_color)
        h = 0.17 + 0.025 * (i % 2)
        if style == "lollipop":
            _line(ax, tr, [(px + seg * 0.5, ay0), (px + seg * 0.5, ay0 + h)], color=c, lw=lw,
                  zorder=zorder)
            ax.add_patch(Circle((px + seg * 0.5, ay0 + h + 0.030), 0.030, transform=tr,
                                fc=PAPER, ec=c, lw=lw, zorder=zorder + 0.1, clip_on=False))
            out["loop_tops"].append((px + seg * 0.5, ay0 + h + 0.060))
        else:
            a, b = px + seg * 0.34, px + seg * 0.52
            pts = [(px, ay0), (a, ay0), (a, ay0 + h)]
            pts += [(a + (b - a) / 2 + (b - a) / 2 * math.cos(t),
                     ay0 + h + (b - a) / 2 * math.sin(t))
                    for t in np.linspace(math.pi, 0, 26)]
            pts += [(b, ay0 + h), (b, ay0), (px + seg, ay0)]
            _line(ax, tr, [p for p in pts], color=c, lw=lw, zorder=zorder)
            for f in (0.35, 0.65):
                yy = ay0 + h * f
                _line(ax, tr, [(a, yy), (b, yy)], color=c, lw=_lw("hairline"),
                      zorder=zorder - 0.1, cap="butt")
            out["loop_tops"].append((a + (b - a) / 2, ay0 + h + (b - a) / 2))
        px += seg
    out["three_p"] = (px, ay0)
    if label:
        ax.text(ax0 + sx / 2.0, ay0 - 0.065, label, transform=tr, ha="center", va="top",
                fontsize=TYPE["in_figure"], color=INK, zorder=zorder, clip_on=False)
    return out


def pam_mark(track, bp_a, bp_b=None, *, y=0.0, gap=None, style="block", zorder=None,
             label="PAM"):
    """The protospacer-adjacent motif, in the `pam` token.

    ⚠️ `pam` is `#2400B2`, a searched token added 2026-09-04. It was `#E69F00` -- the same hex as
    `barcode` -- so one hue carried two molecular identities, and `palette_check` could not see it
    because the PAM was a hex in a prose table rather than a token.
    
    """
    gap = RAIL_GAP if gap is None else gap
    bp_b = bp_a + 2 if bp_b is None else bp_b
    ax, tr = track.ax, track.tr
    zorder = (track.zorder if zorder is None else zorder) + 0.5
    lo, hi = track.span(bp_a, bp_b)
    yt = y + gap / 2.0
    if style == "block":
        p = Rectangle((lo, y - gap / 2.0), hi - lo, gap, transform=tr, fc=PAM, alpha=0.85,
                      ec="none", zorder=zorder - 0.2, clip_on=False)
        ax.add_patch(p)
        exempt(p)
    else:
        _line(ax, tr, [(lo, yt + 0.02), (hi, yt + 0.02)], color=PAM, lw=_lw("emphasis"),
              zorder=zorder, cap="butt")
    if label:
        ax.text((lo + hi) / 2.0, yt + 0.055, label, transform=tr, ha="center", va="bottom",
                fontsize=TYPE["in_figure"], color=INK, zorder=zorder, clip_on=False)
    return dict(left=(lo, y), right=(hi, y), centre=((lo + hi) / 2.0, y), top=((lo + hi) / 2.0, yt))


def cut_mark(track, bp, *, y=0.0, gap=None, label="Cas9 cut", zorder=None, color=None):
    """The blunt Cas9 cut. 🔴 Drawn at ``track.edge(bp)`` -- a cut falls BETWEEN two bases."""
    gap = RAIL_GAP if gap is None else gap
    ax, tr = track.ax, track.tr
    zorder = (track.zorder if zorder is None else zorder) + 0.9
    color = INK if color is None else color
    xe = track.edge(bp)
    _line(ax, tr, [(xe, y - gap / 2.0 - 0.035), (xe, y + gap / 2.0 + 0.035)], color=color,
          lw=_lw("primary"), zorder=zorder)
    if label:
        ax.text(xe, y + gap / 2.0 + 0.048, label, transform=tr, ha="center", va="bottom",
                fontsize=TYPE["in_figure"], color=INK, zorder=zorder, clip_on=False)
    return dict(at=(xe, y), top=(xe, y + gap / 2.0 + 0.035))


def rloop(track, bp_a, bp_b, *, guide_side, y=0.0, gap=None, bubble="hex", depth=0.10,
          zorder=None, guide_color=None, label=True):
    """    🔴 `guide_side` HAS NO DEFAULT AND MUST BE PASSED. Two exemplars put the displaced strand on
    top and a third puts it on the bottom; which is correct depends on the guide's orientation. A
    hardcoded value mis-draws every antisense design while producing a picture that looks completely
    plausible. Take it from the design record, not from the drawing code.

    Geometry is the flat-topped hexagon: each strand leaves the duplex on a straight diagonal, runs
    horizontally, and rejoins on a mirrored diagonal. The displaced strand carries base-ticks
    pointing OUTWARD into empty space, which is the 4-of-4 attested ssDNA cue.
    
    """
    gap = RAIL_GAP if gap is None else gap
    if guide_side not in ("top", "bottom"):
        raise ValueError("rloop(guide_side=...) must be 'top' or 'bottom' and must be taken from "
                         "the design, not defaulted -- see 1292 G2.")
    ax, tr = track.ax, track.tr
    zorder = track.zorder if zorder is None else zorder
    guide_color = GUIDE if guide_color is None else guide_color
    lo, hi = track.span(bp_a, bp_b)
    ramp = min(0.14, (hi - lo) * 0.22)
    sgn_g = 1.0 if guide_side == "top" else -1.0
    for which in ("guide", "displaced"):
        sgn = sgn_g if which == "guide" else -sgn_g
        y0 = y + sgn * gap / 2.0
        y1 = y + sgn * (gap / 2.0 + depth)
        if bubble == "lens":
            xs = np.linspace(lo, hi, 80)
            ys = y0 + (y1 - y0) * np.sin(np.linspace(0, math.pi, 80))
            pts = list(zip(xs, ys))
        else:
            pts = [(lo, y0), (lo + ramp, y1), (hi - ramp, y1), (hi, y0)]
        col = guide_color if which == "guide" else RULE
        _line(ax, tr, pts, color=col, lw=_lw("primary" if which == "guide" else "secondary"),
              zorder=zorder + 0.2)
        if which == "displaced":
            for f in np.linspace(0.18, 0.82, 7):
                bx = lo + (hi - lo) * f
                _line(ax, tr, [(bx, y1), (bx, y1 + sgn * 0.026)], color=RULE,
                      lw=_lw("hairline"), zorder=zorder + 0.2, cap="butt")
        else:
            for f in np.linspace(0.18, 0.82, 7):
                bx = lo + (hi - lo) * f
                _line(ax, tr, [(bx, y1), (bx, y - sgn * gap / 2.0)], color=col,
                      lw=_lw("hairline"), zorder=zorder + 0.1, cap="butt")
    if label:
        ax.text((lo + hi) / 2.0, y + sgn_g * (gap / 2.0 + depth + 0.05), "R-loop", transform=tr,
                ha="center", va="bottom" if sgn_g > 0 else "top", fontsize=TYPE["in_figure"],
                color=RULE, zorder=zorder + 0.4, clip_on=False)
    return dict(left=(lo, y), right=(hi, y),
                guide_apex=((lo + hi) / 2.0, y + sgn_g * (gap / 2.0 + depth)),
                displaced_apex=((lo + hi) / 2.0, y - sgn_g * (gap / 2.0 + depth)))


def copied_span(track, bp0, bp1, *, y=0.0, gap=None, color=None, arrow_at="right",
                payload=None, soften=0.025, zorder=None, label=None):
    """    A bounded segment OF THE RECIPIENT'S RAIL in `donor`, beginning at the invading 3' end and
    ending where synthesis stopped, with a small arrowhead at its 3' terminus. Everything either
    side stays `ink`.

    🔴 `bp0, bp1` ARE THE BOUNDARIES OF WHAT ACTUALLY INTEGRATES, TAKEN FROM THE DESIGN, NEVER FROM
    THE DONOR'S DRAWN EXTENT. That single parameterisation is what makes the glyph honest: the drawn
    donor can extend past `bp1` without implying anything. Only the edit-proximal donor sequence
    reaches the genome; the boundaries are annealing anchors, and a product built by splicing a
    whole donor arm into the CDS imports sequence that never integrates. It fails silently, as a
    plausible number -- that is how a fabricated truncation offset once reached a rendered PDF.

    🔴 `donor`, NOT `edit`. `edit` already means SNV / indel / variant / CUT SITE, so a copied
    stretch in `edit` collides with the cut site immediately upstream of it. `donor` is also the
    published rule: a copy carries the colour of its information SOURCE.

    🔴 THE ENDS SOFTEN INTO `ink` rather than butt-joining. The exact crossover point inside the
    homology is not determined by the design, and a hard edge asserts a base-precise boundary we
    cannot claim.
    
    """
    gap = RAIL_GAP if gap is None else gap
    ax, tr = track.ax, track.tr
    zorder = (track.zorder if zorder is None else zorder) + 0.35
    color = DONOR if color is None else color
    lo, hi = track.span(bp0, bp1)
    span_w = hi - lo
    soft = max(span_w * soften, 1e-4)
    yt, yb = y + gap / 2.0, y - gap / 2.0
    n = 60
    for yy in (yt, yb):
        for i in range(n):
            f0, f1 = i / n, (i + 1) / n
            x0, x1 = lo + span_w * f0, lo + span_w * f1
            a = min(1.0, (x0 - lo) / soft, (hi - x1) / soft)
            if a <= 0:
                continue
            _line(ax, tr, [(x0, yy), (x1, yy)], color=color, lw=_lw("primary"),
                  zorder=zorder, cap="butt", alpha=float(min(1.0, max(0.0, a))))
    if arrow_at:
        from matplotlib.patches import FancyArrowPatch
        tip = hi if arrow_at == "right" else lo
        back = tip - track.direction * 0.045 if arrow_at == "right" else tip + track.direction * 0.045
        a = FancyArrowPatch((back, yt), (tip, yt), arrowstyle="-|>", mutation_scale=8.0,
                            lw=_lw("secondary"), color=color, transform=tr, clip_on=False,
                            zorder=zorder + 0.2, shrinkA=0, shrinkB=0)
        ax.add_patch(a)
    if payload is not None:
        p0, p1, tok, name = payload
        plo, phi = track.span(p0, p1)
        pt = Rectangle((plo, yb), phi - plo, gap, transform=tr, fc=tok, ec=INK,
                       lw=_lw("hairline"), zorder=zorder + 0.3, clip_on=False, hatch="///")
        ax.add_patch(pt)
        exempt(pt)
        if _text_w_in(name, TYPE["annotation"]) <= (phi - plo) * 0.90:
            ax.text((plo + phi) / 2.0, y, name, transform=tr, ha="center", va="center",
                    fontsize=TYPE["annotation"], color=PAPER, zorder=zorder + 0.4,
                    clip_on=False)
        else:
            ax.text((plo + phi) / 2.0, yt + 0.030, name, transform=tr, ha="center",
                    va="bottom", fontsize=TYPE["annotation"], color=INK,
                    zorder=zorder + 0.4, clip_on=False)
    if label:
        ax.text((lo + hi) / 2.0, yb - 0.055, label, transform=tr, ha="center", va="top",
                fontsize=TYPE["in_figure"], color=INK, zorder=zorder, clip_on=False)
    return dict(left=(lo, y), right=(hi, y), centre=((lo + hi) / 2.0, y),
                three_p=(hi if arrow_at == "right" else lo, yt))


def sdsa(track, bp_break, *, y=0.0, gap=None, donor_y=0.36, resect=10, invade=18,
         copied=None, payload=None, zorder=None, labels=True):
    """Synthesis-dependent strand annealing -- 🔴 THE ONLY REPAIR TOPOLOGY THIS MODULE DRAWS.

    Draws: the resected break, the donor held ALONGSIDE (never touching), the invading 3' end, and
    `copied_span` written into the LOCUS rail at the design's own boundaries.
    
    """
    gap = RAIL_GAP if gap is None else gap
    ax, tr = track.ax, track.tr
    zorder = track.zorder if zorder is None else zorder
    out = {}
    out["break"] = break_(track, bp_break, kind="resect", stagger=resect, side="top",
                          y=y, gap=gap, label=None)
    if labels:
        rx = track.edge(bp_break) - track.direction * 0.10
        ax.text(rx, y + gap / 2.0 + 0.105, "resected", transform=tr,
                ha="right" if track.direction > 0 else "left", va="bottom",
                fontsize=TYPE["in_figure"], color=INK, zorder=zorder + 1.2, clip_on=False)
    dlo, dhi = track.span(max(track.bp0, bp_break - invade), min(track.bp1, bp_break + invade))
    for yy in (donor_y + gap / 2.0, donor_y - gap / 2.0):
        _line(ax, tr, [(dlo, yy), (dhi, yy)], color=DONOR, lw=_lw("primary"), zorder=zorder)
    for f in (0.12, 0.5, 0.88):
        xx = dlo + (dhi - dlo) * f
        _line(ax, tr, [(xx, donor_y - gap / 2.0 - 0.012), (xx, y + gap / 2.0 + 0.012)],
              color=RULE, lw=_lw("hairline"), zorder=zorder - 0.2, ls=(0, (1.6, 2.0)))
    if labels:
        ax.text((dlo + dhi) / 2.0, donor_y + gap / 2.0 + 0.05, "donor, held near the break",
                transform=tr, ha="center", va="bottom", fontsize=TYPE["in_figure"], color=INK,
                zorder=zorder, clip_on=False)
    out["donor"] = dict(left=(dlo, donor_y), right=(dhi, donor_y),
                        centre=((dlo + dhi) / 2.0, donor_y))
    if copied is not None:
        out["copied"] = copied_span(track, copied[0], copied[1], y=y, gap=gap, payload=payload,
                                    zorder=track.zorder + 1.0,
                                    label="copied into the genome" if labels else None)
    return out


def two_state(ax, x, y, *, draw, states=("dark", "light"), d=0.52, sep=0.95, coords="axes",
              zorder=6, labels=True, reversible=True, outcome=None, coupling="outcome"):
    """A general TWO-STATE affordance: two registered instances of ONE glyph, differing in one part.

    🔴 P5: reversibility is TWO STRAIGHT SOLID ARROWS SIDE BY SIDE, one each way, between vertically
    stacked states, GROUND STATE ON TOP.
    
    """
    tr = inch_axes(ax, x, y, coords)
    out = {}
    for i, st in enumerate(states):
        dy = -i * sep
        out[st] = draw(ax, 0, 0, d, tr=tr, at=(0.0, dy), state=st, zorder=zorder)
        if labels:
            ax.text(-d * 1.95, dy, st, transform=tr, ha="right", va="center",
                    fontsize=TYPE["in_figure"], color=INK, zorder=zorder + 0.5, clip_on=False)
        if outcome is not None:
            from matplotlib.patches import FancyArrowPatch
            ok = outcome[i]
            a = FancyArrowPatch((d * 1.30, dy), (d * 1.86, dy), arrowstyle="-|>",
                                mutation_scale=9.0, lw=_lw("secondary"),
                                color=INK if ok else RULE, transform=tr, clip_on=False,
                                zorder=zorder, shrinkA=0, shrinkB=0)
            ax.add_patch(a)
            if ok:
                ax.text(d * 2.00, dy, "product", transform=tr, ha="left", va="center",
                        fontsize=TYPE["in_figure"], color=INK, zorder=zorder + 0.5,
                        clip_on=False)
            else:
                cx, r = d * 2.14, 0.035
                for s0, s1 in (((-r, -r), (r, r)), ((-r, r), (r, -r))):
                    _line(ax, tr, [(cx + s0[0], dy + s0[1]), (cx + s1[0], dy + s1[1])],
                          color=EDIT, lw=_lw("secondary"), zorder=zorder + 0.5)
    if reversible and len(states) == 2:
        from matplotlib.patches import FancyArrowPatch
        for k, (y0, y1) in enumerate(((-0.30, -sep + 0.30), (-sep + 0.30, -0.30))):
            dx = -0.055 if k == 0 else 0.055
            a = FancyArrowPatch((dx, y0), (dx, y1), arrowstyle="-|>", mutation_scale=8.0,
                                lw=_lw("secondary"), color=RULE, transform=tr, clip_on=False,
                                zorder=zorder, shrinkA=0, shrinkB=0)
            ax.add_patch(a)
    return out


UNASSIGNED_CARRIER = "unassigned"


SCHEMATIC_REF_NT = 15.0

A_PER_NT = 6.3


def ssdna_loop(ax, x, y, *, anchor, dir_deg, contour_in, tr=None, coords="axes", color=None,
               aspect=0.62, ticks=True, tick_n=26, zorder=5, lw=None, label=None,
               label_side="out"):
    """A loop of single-stranded DNA whose drawn CONTOUR LENGTH is the real one.

    Leaves `anchor` along `dir_deg`, balloons out and returns to the same point — the shape an
    engineered insert makes when it is carried by a molecule that stays put. `contour_in` is what
    makes it honest::

        contour_in = n_nt * molecules.A_PER_NT * s      # s = inches per Angstrom of whatever it joins

    🔴 THE POINT OF DRAWING IT TO SCALE IS THAT IT IS BIG. A 100 nt donor is 630 A of contour, which
    closes into a loop 201 A across against an Ec86 RT that is 113 A wide — **1.8x the enzyme that
    made it**. A donor drawn as a tidy little bump beside the protein is not a simplification, it is
    a different claim about what the molecule looks like.

    🔴 SINGLE-STRANDED IS CARRIED BY THE TICKS, NEVER BY HUE (module docstring section 4): one comb
    with no opposing rail. The loop takes the `donor` token like every other donor species.

    ⚠️ A loop drawn this way is a MODEL. No structure holds it, so a panel using it must say so —
    the same contract `surface_render` enforces for a `placed` group.
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    col = DONOR if color is None else color
    lw = _lw("secondary") if lw is None else lw
    if contour_in <= 0:
        raise ValueError("ssdna_loop(contour_in=) must be positive; it is a real length, in inches")
    t = np.linspace(0, 2 * math.pi, 720)
    ux, uy = (1 - np.cos(t)), aspect * np.sin(t)
    per = float(np.sum(np.hypot(np.diff(ux), np.diff(uy))))
    a = contour_in / per
    lx, ly = a * ux, a * uy
    th = math.radians(dir_deg)
    ca, sa = math.cos(th), math.sin(th)
    px = anchor[0] + lx * ca - ly * sa
    py = anchor[1] + lx * sa + ly * ca
    ax.plot(px, py, transform=tr, color=col, lw=lw, zorder=zorder, clip_on=False,
            solid_capstyle="round")
    if ticks:
        step = max(1, len(t) // tick_n)
        for i in range(0, len(t) - 1, step):
            dx, dy = px[i + 1] - px[i], py[i + 1] - py[i]
            m = math.hypot(dx, dy) or 1e-9
            nxp, nyp = -dy / m, dx / m
            cxp, cyp = px.mean(), py.mean()
            if (px[i] - cxp) * nxp + (py[i] - cyp) * nyp < 0:
                nxp, nyp = -nxp, -nyp
            ln, = ax.plot([px[i], px[i] + nxp * 0.017], [py[i], py[i] + nyp * 0.017],
                          transform=tr, color=col, lw=_lw("hairline"), zorder=zorder,
                          clip_on=False, solid_capstyle="butt")
            exempt(ln)
    far = (anchor[0] + 2 * a * ca, anchor[1] + 2 * a * sa)
    if label:
        off = 0.055
        ax.text(far[0] + off * ca, far[1] + off * sa, label, transform=tr, ha="center",
                va="center", fontsize=TYPE["annotation"], color=INK, zorder=zorder + 0.4,
                clip_on=False)
    return dict(anchor=tuple(anchor), far=far, a=a, b=a * aspect,
                width_in=2 * a, height_in=2 * a * aspect, contour_in=contour_in)


RNA_RISE_A = 2.8
DOME_ASPECT = 0.34
DUPLEX_DIAM_A = 23.0


def _arc_len(xs, ys):
    return float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))


def _solve_bow(d, contour, n=480, lobe_ratio=1.7):
    """Slack meander of horizontal span `d` whose ARC LENGTH is `contour`. Returns (xs, ys).

    🔴 THE SLACK IS SPENT IN SEVERAL SMALL LOBES, NOT ONE TALL ARCH. A 20 nt spacer is 126 A of
    contour across a 28 A end-to-end distance; put into a single half-sine that is an arch taller
    than the hairpins it joins, which draws the eye to the linker instead of the molecule. The lobe
    count comes from the excess contour (`contour / (lobe_ratio * d)`), so a slacker spacer gets MORE
    lobes rather than a taller one, which is also what a flexible chain looks like.

    🔴 THE SLACK IS THE POINT. A single-stranded spacer has a real contour length (`A_PER_NT` per
    nt) and no structure holding it straight, so drawing it as a taut rod between two elements
    asserts a rigidity it does not have -- and reads as a long linker. Drawn slack, the same
    nucleotides occupy an honest end-to-end distance and look like what they are.
    """
    t = np.linspace(0.0, 1.0, n)
    xs = t * d
    if d <= 0 or contour <= d * 1.001:
        return xs, np.zeros_like(xs)
    m = max(1, int(round(contour / (lobe_ratio * d))))
    shape = np.sin(m * math.pi * t)
    lo, hi = 0.0, max(contour, d)
    for _ in range(60):
        a = 0.5 * (lo + hi)
        if _arc_len(xs, a * shape) < contour:
            lo = a
        else:
            hi = a
    return xs, 0.5 * (lo + hi) * shape


def _solve_cap(w, contour, n=180):
    """Half-ellipse of base `w` whose arc length is `contour` -- a hairpin loop at its real length."""
    th = np.linspace(math.pi, 0.0, n)
    cx = (w / 2.0) * np.cos(th)
    lo, hi = 0.0, max(contour, w)
    for _ in range(60):
        b = 0.5 * (lo + hi)
        if _arc_len(cx, b * np.sin(th)) < contour:
            lo = b
        else:
            hi = b
    b = 0.5 * (lo + hi)
    return cx, b * np.sin(th), b


def rna_fold(ax, x, y, elements, *, s, coords="axes", zorder=6, tr=None, at=(0.0, 0.0),
             color=None, lw=None, ticks=True, labels=True, kind="rna", ends=True, scale="contour"):
    """A LINE-AND-STICK nucleic acid: one backbone whose every element costs its own nucleotides.

    🔴 ONE SCALE, `s` INCHES PER ANGSTROM, AND EVERY LENGTH IS DERIVED FROM IT. Coil contour is
    `nt * A_PER_NT * s`; a stem's height is `bp * RNA_RISE_A * s`; a loop closes at its own contour;
    a hairpin's footprint is `DUPLEX_DIAM_A * s`. Pass the same `s` a `surface_render` on the panel
    uses (`w / extent`) and the cartoon and the structure are at one scale, which is what lets them
    sit in one figure without either lying about the other's size.

    `elements` is a list of dicts, left to right in 5'->3' order::

        {"kind": "coil",    "nt": 20}
        {"kind": "hairpin", "stem_bp": 7, "loop_nt": 4, "label": "MS2", "color": M.RECRUIT}
        {"kind": "domain",  "nt": 72, "stems": [6, 6, 3, 2, 4], "label": "HDV ribozyme"}

    A `domain` is a folded region drawn as its OWN stems side by side, with its unpaired nucleotides
    spent on the loops and the joins between them. ⚠️ It is a topology-free depiction: it states how
    much helix the domain has, not how those helices pack. A pseudoknotted fold (HDV) must not be
    drawn as nested hairpins, which would assert a topology it does not have -- say so in the legend.

    Class cues follow the module contract (docstring section 4): unpaired runs carry ONE comb,
    stems carry a rung LADDER (one rung per base pair, real here), and neither is a hue.
    
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    base = DONOR if color is None else color
    lw = _lw("secondary") if lw is None else lw
    ax0, ay0 = at
    if scale not in ("contour", "schematic"):
        raise ValueError(f"rna_fold(scale=…) is 'contour' or 'schematic', not {scale!r}")
    if scale == "schematic":
        elements = [dict(el, **({"nt": math.sqrt(float(el["nt"]) * SCHEMATIC_REF_NT)}
                                if el.get("nt") is not None else {})) for el in elements]
    cx, out = ax0, {"items": [], "contour_in": 0.0, "scale": scale}
    per_nt = A_PER_NT * s
    foot = DUPLEX_DIAM_A * s

    def comb(xs, ys, n_nt, col, up=True):
        if not ticks or n_nt <= 0:
            return
        step = max(1, len(xs) // max(int(n_nt), 1))
        for i in range(0, len(xs) - 1, step):
            dx, dy = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
            m = math.hypot(dx, dy) or 1e-9
            nx, ny = (-dy / m, dx / m) if up else (dy / m, -dx / m)
            ln, = ax.plot([xs[i], xs[i] + nx * 0.013], [ys[i], ys[i] + ny * 0.013], transform=tr,
                          color=col, lw=_lw("hairline"), zorder=zorder, clip_on=False,
                          solid_capstyle="butt")
            exempt(ln)

    def hairpin(x0, stem_bp, loop_nt, col, label=None):
        h = max(stem_bp, 1) * RNA_RISE_A * s
        gap = min(foot, max(RAIL_GAP, 0.30 * foot))
        xl = x0 + (foot - gap) / 2.0
        xr = xl + gap
        lc = max(loop_nt, 3) * per_nt
        lx, ly, b = _solve_cap(gap, lc)
        _line(ax, tr, [(xl, ay0), (xl, ay0 + h)], color=col, lw=lw, zorder=zorder)
        _line(ax, tr, [(xr, ay0 + h), (xr, ay0)], color=col, lw=lw, zorder=zorder)
        _line(ax, tr, list(zip(xl + gap / 2.0 + lx, ay0 + h + ly)), color=col, lw=lw,
              zorder=zorder)
        _line(ax, tr, [(x0, ay0), (xl, ay0)], color=col, lw=lw, zorder=zorder)
        _line(ax, tr, [(xr, ay0), (x0 + foot, ay0)], color=col, lw=lw, zorder=zorder)
        for k in range(stem_bp):
            yy = ay0 + h * (k + 0.5) / max(stem_bp, 1)
            _line(ax, tr, [(xl, yy), (xr, yy)], color=col, lw=_lw("hairline"),
                  zorder=zorder - 0.1, cap="butt")
        top = ay0 + h + b
        if label and labels:
            ax.text(xl + gap / 2.0, top + 0.050, label, transform=tr, ha="center", va="bottom",
                    fontsize=TYPE["in_figure"], color=INK, zorder=zorder + 0.4, clip_on=False)
        return dict(x0=x0, x1=x0 + foot, top=top, centre=(x0 + foot / 2.0, ay0 + h / 2.0),
                    height_in=h + b, contour_in=2 * stem_bp * RNA_RISE_A * s + lc)

    def lollipop(x0, stem_bp, loop_nt, col, label=None, cargo=None):
        """🔴 THE FIELD'S STEM-LOOP DEVICE: a thin two-line stem with an OPEN CIRCLE tip."""
        h = max(stem_bp, 1) * RNA_RISE_A * s
        gap = min(foot * 0.42, max(0.030, 0.16 * foot))
        xl = x0 + (foot - gap) / 2.0
        xr = xl + gap
        r = max(loop_nt, 3) * A_PER_NT * s / (2.0 * math.pi)
        r = max(r, gap * 0.62)
        _line(ax, tr, [(x0, ay0), (xl, ay0)], color=col, lw=lw, zorder=zorder)
        _line(ax, tr, [(xr, ay0), (x0 + foot, ay0)], color=col, lw=lw, zorder=zorder)
        _line(ax, tr, [(xl, ay0), (xl, ay0 + h)], color=col, lw=lw, zorder=zorder)
        _line(ax, tr, [(xr, ay0), (xr, ay0 + h)], color=col, lw=lw, zorder=zorder)
        cy = ay0 + h + r * 0.72
        ring = Circle((xl + gap / 2.0, cy), r, transform=tr, fc=PAPER, ec=col, lw=lw,
                      zorder=zorder + 0.2, clip_on=False)
        ax.add_patch(ring)
        top = cy + r
        if cargo:
            from matplotlib.patches import FancyBboxPatch
            pad = max(r * 0.55, 0.035)
            bx = FancyBboxPatch((xl - pad, ay0 + h * 0.10), gap + 2 * pad, (top - ay0) * 0.95,
                                boxstyle="round,pad=0.004,rounding_size=0.02", transform=tr,
                                fc="none", ec=RULE, lw=_lw("hairline"),
                                linestyle=(0, (2.4, 1.8)), zorder=zorder + 0.3, clip_on=False)
            ax.add_patch(bx)
            if labels:
                ax.text(xl + gap / 2.0, ay0 - 0.115, cargo, transform=tr, ha="center",
                        va="top", fontsize=TYPE["annotation"], color=RULE, zorder=zorder + 0.4,
                        clip_on=False)
        if label and labels:
            ax.text(xl + gap / 2.0, top + 0.045, label, transform=tr, ha="center", va="bottom",
                    fontsize=TYPE["in_figure"], color=INK, zorder=zorder + 0.4, clip_on=False)
        return dict(x0=x0, x1=x0 + foot, top=top, tip=(xl + gap / 2.0, cy), r=r,
                    centre=(xl + gap / 2.0, ay0 + h / 2.0),
                    contour_in=2 * stem_bp * RNA_RISE_A * s + max(loop_nt, 3) * A_PER_NT * s)

    def dome(x0, nt, col, label=None, form="dome"):
        """The large OPEN CIRCLE the backbone runs around -- the a1/a2 inverted-repeat region.

        Buffington Fig 1a's central device. Its circumference is the region's own contour, so the
        circle's size is a nucleotide count rather than a drawing whim.
        """
        circ = max(nt, 8) * A_PER_NT * s
        if form == "circle":
            r = circ / (2.0 * math.pi)
            cxx, cyy = x0 + r, ay0 + r * 0.92
            ax.add_patch(Circle((cxx, cyy), r, transform=tr, fc="none", ec=col, lw=lw,
                                zorder=zorder + 0.1, clip_on=False))
            _line(ax, tr, [(x0, ay0), (cxx - r * 0.72, ay0 + r * 0.22)], color=col, lw=lw,
                  zorder=zorder)
            _line(ax, tr, [(cxx + r * 0.72, ay0 + r * 0.22), (x0 + 2 * r, ay0)], color=col, lw=lw,
                  zorder=zorder)
            return dict(x0=x0, x1=x0 + 2 * r, top=cyy + r, centre=(cxx, cyy), r=r, contour_in=circ)
        th = np.linspace(math.pi, 0.0, 200)
        lo, hi = 0.0, circ
        for _ in range(60):
            wd = 0.5 * (lo + hi)
            ux, uy = (wd / 2.0) * np.cos(th), (wd * DOME_ASPECT) * np.sin(th)
            if _arc_len(ux, uy) < circ:
                lo = wd
            else:
                hi = wd
        wd = 0.5 * (lo + hi)
        b = wd * DOME_ASPECT
        ux, uy = (wd / 2.0) * np.cos(th), np.sin(th) * b
        cxx = x0 + wd / 2.0
        _line(ax, tr, list(zip(cxx + ux, ay0 + uy)), color=col, lw=lw, zorder=zorder + 0.1)
        cyy = ay0 + b * 0.60
        r = b
        return dict(x0=x0, x1=x0 + wd, top=ay0 + b, centre=(cxx, cyy), r=r, contour_in=circ)
        raise AssertionError("unreachable")

    def branch_g(x0, col):
        ""
        r = 0.052
        ring = Circle((x0 + r, ay0), r, transform=tr, fc=PAPER, ec=col, lw=_lw("secondary"),
                      zorder=zorder + 0.5, clip_on=False)
        ax.add_patch(ring)
        t_ = ax.text(x0 + r, ay0, "G", transform=tr, ha="center", va="center",
                     fontsize=TYPE["annotation"], color=col, zorder=zorder + 0.6, clip_on=False)
        exempt(t_)
        return dict(x0=x0, x1=x0 + 2 * r, top=ay0 + r, centre=(x0 + r, ay0), contour_in=A_PER_NT * s)

    for el in elements:
        k = el.get("kind")
        col = el.get("color", base)
        if k == "coil":
            nt = float(el["nt"])
            d = A_PER_NT * math.sqrt(nt) * s
            up = el.get("side", "up") == "up"
            if el.get("slack", True):
                xs, ys = _solve_bow(d, nt * per_nt)
            else:
                xs, ys = np.linspace(0.0, d, 24), np.zeros(24)
            ys = ys if up else -ys
            _line(ax, tr, list(zip(cx + xs, ay0 + ys)), color=col, lw=lw, zorder=zorder)
            comb(cx + xs, ay0 + ys, nt, col, up=up)
            if el.get("label") and labels:
                ax.text(cx + d / 2.0, ay0 + (ys.max() if up else ys.min())
                        + (0.040 if up else -0.040), el["label"], transform=tr, ha="center",
                        va="bottom" if up else "top", fontsize=TYPE["annotation"], color=INK,
                        zorder=zorder + 0.4, clip_on=False)
            out["items"].append(dict(kind="coil", nt=nt, x0=cx, x1=cx + d,
                                     contour_in=nt * per_nt, label=el.get("label")))
            out["contour_in"] += nt * per_nt
            cx += d
        elif k == "hairpin":
            r = hairpin(cx, int(el["stem_bp"]), int(el.get("loop_nt", 4)), col, el.get("label"))
            r.update(kind="hairpin", label=el.get("label"))
            out["items"].append(r)
            out["contour_in"] += r["contour_in"]
            cx = r["x1"]
        elif k == "domain":
            stems = list(el.get("stems") or [])
            nt = float(el["nt"])
            unpaired = max(nt - 2 * sum(stems), 3 * len(stems))
            per_loop = unpaired / max(len(stems) + max(len(stems) - 1, 0), 1)
            tops, x_start = [], cx
            for i, bp in enumerate(stems):
                if i:
                    jn = per_loop
                    d = A_PER_NT * math.sqrt(jn) * s
                    xs, ys = _solve_bow(d, jn * per_nt)
                    _line(ax, tr, list(zip(cx + xs, ay0 - ys)), color=col, lw=lw, zorder=zorder)
                    comb(cx + xs, ay0 - ys, jn, col, up=False)
                    out["contour_in"] += jn * per_nt
                    cx += d
                r = hairpin(cx, int(bp), max(int(round(per_loop)), 3), col)
                tops.append(r["top"])
                out["contour_in"] += r["contour_in"]
                cx = r["x1"]
            if el.get("label") and labels:
                ax.text((x_start + cx) / 2.0, max(tops) + 0.050, el["label"], transform=tr,
                        ha="center", va="bottom", fontsize=TYPE["in_figure"], color=INK,
                        zorder=zorder + 0.4, clip_on=False)
            out["items"].append(dict(kind="domain", nt=nt, stems=stems, x0=x_start, x1=cx,
                                     top=max(tops), label=el.get("label")))
        elif k == "lollipop":
            r_ = lollipop(cx, int(el["stem_bp"]), int(el.get("loop_nt", 4)), col, el.get("label"),
                          el.get("cargo"))
            r_.update(kind="lollipop", label=el.get("label"))
            out["items"].append(r_)
            out["contour_in"] += r_["contour_in"]
            cx = r_["x1"]
        elif k == "dome":
            r_ = dome(cx, float(el["nt"]), col, None, el.get("form", "dome"))
            if el.get("label") and labels:
                ax.text(r_["centre"][0], r_["top"] + 0.045, el["label"], transform=tr, ha="center",
                        va="bottom", fontsize=TYPE["in_figure"], color=INK, zorder=zorder + 0.4,
                        clip_on=False)
            r_.update(kind="dome", label=el.get("label"))
            out["items"].append(r_)
            out["contour_in"] += r_["contour_in"]
            cx = r_["x1"]
        elif k == "branch_g":
            r_ = branch_g(cx, col)
            r_.update(kind="branch_g", label="G")
            out["items"].append(r_)
            out["contour_in"] += r_["contour_in"]
            cx = r_["x1"]
        else:
            raise ValueError(f"rna_fold: unknown element kind {k!r}; expected 'coil', 'hairpin', "
                             f"'lollipop', 'dome' or 'branch_g'")
    if ends and labels:
        for xx, s_lab, ha in ((ax0, "5'", "right"), (cx, "3'", "left")):
            ax.text(xx + (-0.030 if ha == "right" else 0.030), ay0, s_lab, transform=tr, ha=ha,
                    va="center", fontsize=TYPE["annotation"], color=INK, zorder=zorder + 0.4,
                    clip_on=False)
    out.update(five_p=(ax0, ay0), three_p=(cx, ay0), width_in=cx - ax0, s=s)
    return out


FORK_LW_RANGE = (STROKE["hairline"], STROKE["emphasis"])


def fork(ax, x, y, *, arms, basis, coords="axes", zorder=5, tr=None, at=(0.0, 0.0),
         length=0.46, spread=112.0, heading=-90.0, label=None, gap=0.035, label_pad=0.045):
    """A weighted outcome fan: one input, several competing fates, ARM WIDTH = RELATIVE PROBABILITY.

    `arms` is a list of dicts, drawn in order across the fan::

        dict(label="cell death", weight=0.55)                       # solid, the canonical path
        dict(label="NHEJ indel", weight=0.30, blocked="nej1$\\Delta$")   # the arm is closed
        dict(label="donor edit", weight=0.15, style="dashed",
             condition="+ donor")                                   # a condition is MANDATORY here

    🔴 `blocked` IS DRAWN BY THE FORK, NOT BY THE CALLER, and that is deliberate. The first build
    left the caller to place an `inhibit()` bar themselves and the smoke test put it at the fork's
    APEX, square across all three arms -- a picture saying every outcome was blocked, rendered from
    code that raised nothing and passed both floor checks. The fork is the only object that knows an
    arm's angle and length, so it owns the placement. `blocked_at` is the position in units of arm
    length (default just past the arm's own label).

    🔴 `basis` IS A REQUIRED KEYWORD, and it is the honest half of this glyph. Encoding a
    probability in stroke width invites the reader to compare widths, which is a QUANTITATIVE claim
    -- so the caller has to say, on the record, where the numbers came from: ``basis="illustrative"``
    for a schematic, or a string naming the figure or table they were measured in. A picture that
    ranks four repair outcomes by eye and looks measured is the same failure as reporting a count
    without its denominator.

    Returns `{"tips": {label: (x, y)}, "origin": …, "angles": {label: deg},
    "blocked": {label: …}}` so a caller can hang a second fork or a cell glyph off any arm.
    
    """
    if not basis:
        raise ValueError(
            "fork(basis=...) is REQUIRED. Arm width encodes relative probability, which is a "
            "quantitative claim; say where it came from -- basis='illustrative' for a schematic, or "
            "a string naming the panel or table the weights were measured in.")
    arms = [dict(a) for a in arms]
    if len(arms) < 2:
        raise ValueError("fork(arms=...) needs at least two outcomes; one outcome is not a fork.")
    for a in arms:
        if not a.get("label"):
            raise ValueError("every fork arm needs a label -- an unlabelled outcome is not an outcome.")
        if float(a.get("weight", 0.0)) <= 0.0:
            raise ValueError(f"fork arm {a['label']!r} needs weight > 0.")
        if a.get("style", "solid") == "dashed" and not a.get("condition"):
            raise ValueError(
                f"fork arm {a['label']!r} is dashed but names no condition. `the schematic module` rule 8 "
                "as amended 2026-09-04: a dashed branch is labelled with the CONDITION that opens "
                "it, not just with its outcome, because the dash already carries five meanings "
                "across the exemplar set. Pass condition='...', or make the arm solid.")
    from matplotlib.patches import FancyArrowPatch
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    ax0, ay0 = at
    n = len(arms)
    wmax = max(float(a["weight"]) for a in arms)
    lo, hi = FORK_LW_RANGE
    out = {"origin": (ax0, ay0), "tips": {}, "angles": {}, "basis": basis}
    for i, a in enumerate(arms):
        deg = heading + spread * ((i / (n - 1.0)) - 0.5)
        th = math.radians(deg)
        w = float(a["weight"]) / wmax
        lw = max(G.STROKE_MIN_PT, lo + (hi - lo) * w)
        col = a.get("color", INK if a.get("style", "solid") == "solid" else RULE)
        L = float(a.get("length", length))
        p0 = (ax0 + gap * math.cos(th), ay0 + gap * math.sin(th))
        p1 = (ax0 + L * math.cos(th), ay0 + L * math.sin(th))
        style = a.get("style", "solid")
        arr = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6.0 + 3.6 * lw,
                              lw=lw, color=col, transform=tr, clip_on=False,
                              zorder=zorder + 0.2, shrinkA=0, shrinkB=0,
                              linestyle="solid" if style == "solid" else (0, (2.4, 1.7)))
        ax.add_patch(arr)
        tx = ax0 + (L + label_pad) * math.cos(th)
        ty = ay0 + (L + label_pad) * math.sin(th)
        ha = "center" if abs(math.cos(th)) < 0.30 else ("left" if math.cos(th) > 0 else "right")
        va = "center" if abs(math.sin(th)) < 0.30 else ("bottom" if math.sin(th) > 0 else "top")
        txt = a["label"] if not a.get("condition") else f"{a['label']}\n({a['condition']})"
        ax.text(tx, ty, txt, transform=tr, ha=ha, va=va, fontsize=TYPE["in_figure"],
                color=INK, zorder=zorder + 0.4, clip_on=False,
                linespacing=1.20)
        out["tips"][a["label"]] = p1
        out["angles"][a["label"]] = deg
        if a.get("blocked"):
            n_lines = 2 if a.get("condition") else 1
            tw = _text_w_in(max(txt.split("\n"), key=len), TYPE["in_figure"])
            thh = G.cap_mm(TYPE["in_figure"]) / 25.4 * 1.45 * n_lines
            cx_lab = tx + (-tw / 2.0 if ha == "right" else (tw / 2.0 if ha == "left" else 0.0))
            lo_lab = ty - (thh if va == "top" else (thh / 2.0 if va == "center" else 0.0))
            hi_lab = ty + (thh if va == "bottom" else (thh / 2.0 if va == "center" else 0.0))
            down = math.sin(th) <= 0.0
            bx = cx_lab
            by = (lo_lab - 0.030) if down else (hi_lab + 0.030)
            fb = a.get("blocked_at")
            if fb is not None:
                d_bar = float(fb) * L
                bx, by = ax0 + d_bar * math.cos(th), ay0 + d_bar * math.sin(th)
            out.setdefault("blocked", {})[a["label"]] = inhibit(
                ax, x, y, genotype=a["blocked"], tr=tr, at=(bx, by),
                angle=a.get("blocked_angle", 0.0 if down else 180.0),
                zorder=zorder + 1, color=a.get("blocked_color", INK),
                w=float(a.get("blocked_w", 0.115)),
                stem=float(a.get("blocked_stem", 0.075)))
    if label:
        ax.text(ax0, ay0 + 0.055, label, transform=tr, ha="center", va="bottom",
                fontsize=TYPE["in_figure"], color=INK, zorder=zorder + 0.4, clip_on=False)
    return out


def crossover(ax, x, y, *, top, bottom, w=0.070, coords="axes", zorder=4, tr=None, color=None,
              label=None, style="dashed", lw=None):
    """Homologous pairing between two DNA elements, drawn as the crossed dashed pair.

    The two segments cross, which is the whole point: it says the left of the upper element
    continues into the right of the lower one and vice versa. The bioRxiv strategy 2 (Guo/Chavez
    in-vivo plasmid assembly) puts two of these between the linearized plasmid's arms and the
    genomic locus, and they are what makes it gap repair rather than two molecules drawn near
    each other.

    🔴 STRUCTURE, SO NO HUE. A crossover is a relationship between two molecules, not a molecule,
    so it is `rule` grey by default. Colouring it would spend an accent on something that has no
    molecular identity of its own -- and in a panel where both partners are already coloured, a
    third hue between them reads as a third species.
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    col = RULE if color is None else color
    ls = "solid" if style == "solid" else (0, (1.9, 1.5))
    lw = _lw("secondary") if lw is None else lw
    tx, ty = top
    bx, by = bottom
    _line(ax, tr, [(tx - w / 2.0, ty), (bx + w / 2.0, by)], color=col, lw=lw, zorder=zorder, ls=ls)
    _line(ax, tr, [(tx + w / 2.0, ty), (bx - w / 2.0, by)], color=col, lw=lw, zorder=zorder, ls=ls)
    mid = ((tx + bx) / 2.0, (ty + by) / 2.0)
    if label:
        ax.text(mid[0] + w * 0.9, mid[1], label, transform=tr, ha="left", va="center",
                fontsize=TYPE["annotation"], color=INK, zorder=zorder + 0.3, clip_on=False)
    return dict(mid=mid, top=top, bottom=bottom)


SURFACE_LUMP_MIN_IN = 0.55

_STRUCT_DIR = os.environ.get("HOUSEFIG_STRUCTURES") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "structures")
_STRUCT_CACHE = {}

STRUCT_TOKENS = {
    "protein": PROTEIN_BODY, "cas9": PROTEIN_BODY, "rt": PROTEIN_BODY, "fusion": PROTEIN_BODY,
    "guide_rna": GUIDE, "target_dna": RULE, "dna": RULE,
    "ms2_loop": GUIDE,
}


KEYLINE_FRAC = 0.008


def keyline_lw(width_in):
    """The ink keyline width, in points, for a glyph drawn `width_in` inches wide.

    Clamped by the default stroke floor at one end -- below it the line drops out in print -- and by
    the `primary` tier at the other, past which it reads as a sticker outline round a photograph.
    """
    return min(max(width_in * 72.0 * KEYLINE_FRAC, G.STROKE_MIN_PT), STROKE["primary"])


def load_structure(slug):
    """Load a cached 2D contour set. 🔴 No Biopython, no network -- the CACHE is the deliverable."""
    if slug in _STRUCT_CACHE:
        return _STRUCT_CACHE[slug]
    p = os.path.join(_STRUCT_DIR, f"{slug}.json")
    if not os.path.exists(p):
        have = sorted(f[:-5] for f in os.listdir(_STRUCT_DIR)) if os.path.isdir(_STRUCT_DIR) else []
        raise FileNotFoundError(
            f"no structure cache {slug!r} in {_STRUCT_DIR}. Available: {have}. "
            f"Structure caches are built ahead of time, "
            f"never fetched at figure time.")
    import json as _json
    with open(p) as fh:
        doc = _json.load(fh)
    _STRUCT_CACHE[slug] = doc
    return doc


def surface(ax, x, y, w=1.10, *, slug, coords="axes", zorder=6, tr=None, at=(0.0, 0.0),
            tokens=None, groups=None, order=None, lw=None, alpha=1.0, lumps=True,
            underlay=False, underlay_alpha=0.28, label=None, flip=False, shadow=0.030):
    """A molecular surface contour, sized so its widest extent is `w` INCHES.

    `flip` mirrors the whole complex, for the same reason `cas9(flip=)` exists: an antisense design
    is the normal case in half of our library and a hardcoded handedness mis-draws it plausibly.
    
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    lw = _lw("secondary") if lw is None else lw
    doc = load_structure(slug)
    if w < SURFACE_LUMP_MIN_IN:
        lumps, underlay = False, False
    ax0, ay0 = at
    e = doc["extent"]
    span = max(e[2] - e[0], e[3] - e[1])
    s = w / span
    cx, cy = (e[0] + e[2]) / 2.0, (e[1] + e[3]) / 2.0
    mir = -1.0 if flip else 1.0

    def T(poly):
        a = np.asarray(poly, float)
        return np.column_stack([ax0 + mir * (a[:, 0] - cx) * s, ay0 + (a[:, 1] - cy) * s])

    hint = doc.get("draw_this_way") or {}
    tok = dict(STRUCT_TOKENS)
    tok.update(hint.get("tokens") or {})
    tok.update(tokens or {})
    order = order or hint.get("order")
    names = order or [g for g in ("protein", "cas9", "rt",
                                  "target_dna", "dna", "guide_rna")
                      if g in doc["chains"]] or list(doc["chains"])
    names += [g for g in doc["chains"] if g not in names]
    if groups:
        names = [g for g in names if g in groups]

    out = {"groups": {}}
    for i, g in enumerate(names):
        blk = doc["chains"].get(g)
        if not blk:
            continue
        c = tok.get(g, PROTEIN_BODY)
        z = zorder + i * 0.6
        for poly in blk.get("outer", []):
            pts = T(poly)
            G._body(ax, lambda p=pts: Polygon(p, closed=True), tr, fill=c, zorder=z, lw=lw,
                    edge=INK, alpha=alpha, shadow=shadow if i == 0 else 0.0,
                    gloss=None,
                    extent=(pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()))
        if lumps:
            for j, inn in enumerate(blk.get("inner", [])):
                p = Polygon(T(inn), closed=True, transform=tr, clip_on=False,
                            fc=shade(c, -0.055 - 0.045 * j), ec="none", zorder=z + 0.15 + 0.01 * j,
                            alpha=alpha)
                ax.add_patch(p)
                exempt(p)
        if underlay and blk.get("trace"):
            raw = blk["trace"]
            segs = raw if (raw and isinstance(raw[0][0], (list, tuple))) else [raw]
            for seg in segs:
                t = T(seg)
                ax.plot(t[:, 0], t[:, 1], transform=tr, color=INK, lw=_lw("hairline"),
                        alpha=underlay_alpha, zorder=z + 0.3, clip_on=False,
                        solid_capstyle="round")
        pts = T(blk["outer"][0]) if blk.get("outer") else None
        if pts is not None:
            out["groups"][g] = dict(token=c, evidence=blk.get("evidence"),
                                    centre=(float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                                    left=(float(pts[:, 0].min()), ay0),
                                    right=(float(pts[:, 0].max()), ay0),
                                    top=(ax0, float(pts[:, 1].max())),
                                    bottom=(ax0, float(pts[:, 1].min())))
    if label:
        ax.text(ax0, ay0 - (e[3] - e[1]) * s / 2.0 - 0.055, label, transform=tr, ha="center",
                va="top", fontsize=TYPE["in_figure"], color=INK, zorder=zorder + 6, clip_on=False)
    out["centre"] = (ax0, ay0)
    out["width_in"] = w
    out["provenance"] = doc.get("provenance", {})
    return out


EVIDENCE_NOTE = {
    "solved": "",
    "composite": "  \u2014 arrangement modelled",
    "connector": "  \u2014 modelled link",
    "placed": "  \u2014 modelled",
}


def surface_legend(ax, x, y, result, *, coords="axes", size=None, row=0.155, swatch=0.105,
                   names=None, title=None, zorder=20, columns=1):
    """A legend built from WHAT WAS ACTUALLY DRAWN, never from a hand-written list.

    Pass the dict `surface()` / `surface_render()` returned. It carries, per group, the token that
    was resolved at draw time and the `evidence` state the cache declared -- so the swatch cannot
    disagree with the fill, and a modelled arm is labelled as modelled by construction rather than by
    the author remembering to say so.

    🔴 This exists because of `rebuilt-figure-orphans-its-captions`: a legend typed into a figure
    script keeps saying what was true when it was typed. One that reads the render cannot.
    """
    tr = inch_axes(ax, x, y, coords)
    size = TYPE["in_figure"] if size is None else size
    pretty = dict(fusion="fusion, whole", linker="linker (GGGGS)n",
                  ms2_loop="MS2 stem-loop RNA", 
                  protein="protein",
                  guide_rna="guide RNA", target_dna="target DNA", rt="reverse transcriptase", cas9="Cas9")
    pretty.update(names or {})
    items = list(result.get("groups", {}).items())
    dy = 0.0
    if title:
        ax.text(0.0, dy, title, transform=tr, ha="left", va="center", fontsize=size,
                color=INK, fontweight="bold", zorder=zorder, clip_on=False)
        dy -= row
    per = max(1, int(math.ceil(len(items) / float(max(columns, 1)))))
    widest = max((_text_w_in(pretty.get(g, g) + EVIDENCE_NOTE.get(info.get("evidence"), ""), size)
                  for g, info in items), default=1.0)
    colw = widest + swatch + 0.22
    for i, (g, info) in enumerate(items):
        col, k = divmod(i, per)
        yy = dy - k * row
        xx = col * colw
        p = Rectangle((xx, yy - swatch / 2.0), swatch, swatch, transform=tr,
                      fc=info.get("token", PROTEIN_BODY), ec=INK, lw=_lw("hairline"),
                      zorder=zorder, clip_on=False)
        ax.add_patch(p)
        exempt(p)
        note = EVIDENCE_NOTE.get(info.get("evidence"), "")
        ax.text(xx + swatch + 0.055, yy, pretty.get(g, g) + note, transform=tr, ha="left",
                va="center", fontsize=size, color=INK, zorder=zorder, clip_on=False)
    return dict(rows=per, bottom=dy - (per - 1) * row - row / 2.0)


def list_structures():
    """What is in the structure cache, with where each came from."""
    if not os.path.isdir(_STRUCT_DIR):
        return {}
    import json as _json
    out = {}
    for f in sorted(os.listdir(_STRUCT_DIR)):
        if f.endswith(".json"):
            with open(os.path.join(_STRUCT_DIR, f)) as fh:
                d = _json.load(fh)
            out[f[:-5]] = dict(name=d.get("name"), groups=list(d.get("chains", {})),
                               source=d.get("provenance", {}).get("source"))
    return out


BP_PER_TURN = 10.5


def helix_path(ax, pts, *, tr=None, coords="axes", amp=0.050, per_turn=0.30, color=None,
               groove=0.62, rungs=True, zorder=6, lw=None, phase=0.0, rung_color=None):
    """A double helix following an ARBITRARY 2D polyline, in inches. Sibling of `helix()`.

    `helix()` draws along a straight `Track` in base-pair coordinates and is the right call for a
    locus. This one takes a path, and exists for the one thing a track cannot do: **leave a
    molecular surface at the angle that surface's own DNA actually exits, and ease to horizontal.**

    🔴 THE PHASE IS DRIVEN BY ARC LENGTH and the two backbones are offset along the LOCAL NORMAL.
    A sine wave in x keeps its period only while the path is horizontal; through a bend it stretches
    on the outside of the curve, and the major groove stops being the major groove — which is the
    one feature that makes the glyph read as B-DNA rather than as a twisted ribbon.

    🔴 SCALE IT TO WHATEVER IT JOINS, don't pick it. Against a cached structure drawn `w` inches
    wide over an extent of `span` Angstrom, ``s = w / span`` and then::

        per_turn = 34.0 * s          # B-form pitch
        amp      = 20.0 * s / 2.0    # B-form diameter / 2

    Measured on `cas9_sgrna_dna` at 1.05 in: 0.290 in and 0.085 in. The default default `per_turn` of
    0.30 in lands within 3.3% of B-form there, which is why the join reads as one molecule instead
    of as two conventions meeting.
    """
    tr = inch_axes(ax, 0.0, 0.0, coords) if tr is None else tr
    P = np.asarray(pts, float)
    if P.ndim != 2 or P.shape[0] < 2:
        raise ValueError("helix_path(pts=) needs at least two (x, y) points, in inches")
    s_ = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    L = s_[-1]
    if L <= 0:
        raise ValueError("helix_path(pts=) has zero length")
    n = max(90, int(L / 0.004))
    ss = np.linspace(0, L, n)
    cx, cy = np.interp(ss, s_, P[:, 0]), np.interp(ss, s_, P[:, 1])
    dx, dy = np.gradient(cx), np.gradient(cy)
    mg = np.hypot(dx, dy)
    mg[mg == 0] = 1e-9
    nx, ny = -dy / mg, dx / mg
    ph = phase + ss / max(per_turn, 1e-9) * 2 * math.pi
    o1, o2 = amp * np.sin(ph), amp * np.sin(ph + groove * math.pi)
    x1, y1, x2, y2 = cx + nx * o1, cy + ny * o1, cx + nx * o2, cy + ny * o2
    col = INK if color is None else color
    lw = _lw("primary") if lw is None else lw
    if rungs:
        k = max(1, int(round(n / max(L / max(per_turn, 1e-9) * 10.5, 1))))
        for i in range(0, n, k):
            ln, = ax.plot([x1[i], x2[i]], [y1[i], y2[i]], transform=tr,
                          color=rung_color or RULE, lw=_lw("hairline"), zorder=zorder - 0.2,
                          clip_on=False, solid_capstyle="butt")
            exempt(ln)
    for xs, ys, dz in ((x1, y1, 0.0), (x2, y2, 0.1)):
        ax.plot(xs, ys, transform=tr, color=col, lw=lw, zorder=zorder + dz, clip_on=False,
                solid_capstyle="round")
    return dict(start=(cx[0], cy[0]), end=(cx[-1], cy[-1]), length=L, per_turn=per_turn, amp=amp)


def ease_path(p0, dir_deg, p1, dir1_deg=0.0, *, n=44, k=0.42):
    """Cubic Bezier leaving `p0` along `dir_deg` and arriving at `p1` along `dir1_deg`, in inches.

    The companion to `helix_path`: it is what turns a measured exit angle into a path that settles
    to the panel's own horizontal without a visible corner.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d0 = np.array([math.cos(math.radians(dir_deg)), math.sin(math.radians(dir_deg))])
    d1 = np.array([math.cos(math.radians(dir1_deg)), math.sin(math.radians(dir1_deg))])
    dist = float(np.linalg.norm(p1 - p0)) * k
    c0, c1 = p0 + d0 * dist, p1 - d1 * dist
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * c0 + \
           3 * (1 - t) * (t ** 2) * c1 + (t ** 3) * p1


def helix(track, bp_a, bp_b, *, y=0.0, amp=None, bp_per_turn=None, color=None, tier="primary",
          rungs=True, zorder=None, glow=None, groove=0.62, label=None, min_turn_in=0.085):
    """Duplex DNA as a flat-vector double helix -- two phase-shifted backbones with rungs.

    Reached through `duplex(style="helix")`; exposed directly for callers that want the curve.

    Two details separate this from a pair of sine waves:
      * **the grooves are asymmetric.** The two backbones are offset by `groove` x pi, not pi, which
        is what produces the alternating wide (major) and narrow (minor) groove that makes B-DNA
        recognisable. At exactly pi the strands are symmetric and it reads as a ribbon twist.
      * **the strand in front is drawn over the one behind**, switching every half period. Drawn as
        two continuous curves with a fixed z-order, the crossings read as a flat woven braid.
    """
    ax, tr = track.ax, track.tr
    zorder = track.zorder if zorder is None else zorder
    color = INK if color is None else color
    bpt = bp_per_turn or BP_PER_TURN
    lo, hi = track.span(bp_a, bp_b)
    turn_in = bpt * track.bpw
    if turn_in < min_turn_in:
        return duplex(track, bp_a, bp_b, detail=0, y=y, color=color, tier=tier, zorder=zorder,
                      polarity_form="none", label=label)
    amp = 0.055 if amp is None else amp
    n = max(60, int((hi - lo) / 0.006))
    xs = np.linspace(lo, hi, n)
    ph = (xs - lo) / max(turn_in, 1e-9) * 2 * math.pi
    y1 = y + amp * np.sin(ph)
    y2 = y + amp * np.sin(ph + groove * math.pi)

    if glow:
        g0, g1, gc = glow
        glo, ghi = track.span(g0, g1)
        m = (xs >= min(glo, ghi)) & (xs <= max(glo, ghi))
        if m.any():
            for k, a in ((3.2, 0.10), (2.1, 0.13), (1.2, 0.20)):
                ax.fill_between(xs[m], np.minimum(y1, y2)[m] - amp * 0.30 * k,
                                np.maximum(y1, y2)[m] + amp * 0.30 * k, transform=tr,
                                color=gc, alpha=a, lw=0, zorder=zorder - 0.5, clip_on=False)

    if rungs:
        step = max(1, int(round(0.020 / max(track.bpw, 1e-9))))
        for i in range(0, n, max(1, n // max(1, int((hi - lo) / (track.bpw * step))))):
            ax.plot([xs[i], xs[i]], [y1[i], y2[i]], transform=tr, color=shade(color, 0.42),
                    lw=_lw("hairline"), zorder=zorder + 0.05, clip_on=False,
                    solid_capstyle="butt")

    half = max(2, int(n * (turn_in / 2.0) / max(hi - lo, 1e-9)))
    for start in range(0, n - 1, half):
        sl = slice(start, min(n, start + half + 1))
        front_first = ((start // half) % 2 == 0)
        pair = ((y1, y2) if front_first else (y2, y1))
        for k, yy in enumerate(pair):
            ax.plot(xs[sl], yy[sl], transform=tr, color=color,
                    lw=_lw(tier) * (1.0 if k else 0.86),
                    zorder=zorder + (0.3 if k else 0.1), clip_on=False,
                    solid_capstyle="round")
    if label:
        text_in(ax, *track.anchor, (lo + hi) / 2.0, y + amp + 0.075, label, coords=track.coords,
                size=TYPE["in_figure"], color=RULE, va="bottom")
    return dict(left=(lo, y), right=(hi, y), centre=((lo + hi) / 2.0, y),
                top=((lo + hi) / 2.0, y + amp), bottom=((lo + hi) / 2.0, y - amp), amp=amp)


SURFACE_LIGHT = (-0.48, 0.60, 0.64)

SURFACE_RENDER_MIN_IN = 0.62
SURFACE_RELIEF_MIN_IN = 0.34


def _load_depth(slug):
    """The `.npz` beside the JSON, or None. 🔴 A missing sidecar is NOT an error -- the smooth tier
    still draws, so a JSON copied on its own keeps working."""
    p = os.path.join(_STRUCT_DIR, f"{slug}.npz")
    if not os.path.exists(p):
        return None
    key = f"_npz::{slug}"
    if key not in _STRUCT_CACHE:
        _STRUCT_CACHE[key] = dict(np.load(p))
    return _STRUCT_CACHE[key]


def shade_depth(Q, mask, z_range, base, *, grid_A=0.5, light=SURFACE_LIGHT, ambient=0.50,
                diffuse=0.62, spec=0.34, shininess=22.0, ao=0.55, ao_radius=3.0, ao_depth=4.0,
                rim=0.45, rim_px=2.5, smooth_A=0.0):
    """Turn a depth buffer into an RGBA image, lit from the default direction, in `base` colour.

    Lambert plus a Blinn-Phong specular over the surface normals, which are the gradient of the
    height field. The defaults were picked from a rendered sweep, not guessed: ambient 0.50 /
    diffuse 0.62 gives a luminance spread of 0.58 across the body, which reads as relief without the
    deep shadows of a 0.42/0.72 setting that made the surface look charcoal-rubbed at figure size. The atomic bumps ARE the relief -- nothing here is procedural texture, so the
    detail a reader sees is the real solvent-excluded surface of the real structure.
    """
    z0, z1 = z_range
    Z = z0 + (Q.astype(np.float32) / 65535.0) * (z1 - z0)
    if mask.any():
        idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        Zf = Z[tuple(idx)]
    else:
        Zf = np.zeros_like(Z)
    if smooth_A:
        Zf = ndimage.gaussian_filter(Zf, max(smooth_A / max(grid_A, 1e-6), 0.0))
    gy, gx = np.gradient(Zf, grid_A)
    nx, ny, nz = -gx, -gy, np.ones_like(Zf)
    n = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / n, ny / n, nz / n
    lx, ly, lz = light
    ln = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / ln, ly / ln, lz / ln
    lam = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    if ao or rim:
        blur = ndimage.gaussian_filter(Zf, ao_radius)
        occ = np.clip((blur - Zf) / max(ao_depth, 1e-6), 0.0, 1.0) if ao else 0.0
        if rim:
            din = ndimage.distance_transform_edt(mask)
            edge = np.clip(din / max(rim_px, 1e-6), 0.0, 1.0)
        else:
            edge = 1.0
        lam = lam * (1.0 - ao * occ) * (1.0 - rim * (1.0 - edge))
    hx, hy, hz = lx, ly, lz + 1.0
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    sp = np.clip(nx * hx / hn + ny * hy / hn + nz * hz / hn, 0.0, 1.0) ** shininess
    br, bg, bb = G._rgb(base)
    shade_f = ambient + diffuse * lam
    rgb = np.dstack([np.clip(br * shade_f + spec * sp, 0, 1),
                     np.clip(bg * shade_f + spec * sp, 0, 1),
                     np.clip(bb * shade_f + spec * sp, 0, 1)])
    return np.dstack([rgb, mask.astype(np.float32)])


CARTOON_RELIEF = 10.0

CARTOON_SIGMA_LADDER_A = (4.0, 3.0, 2.25, 1.5, 1.0, 0.5)
CARTOON_MAX_AREA_LOSS = 0.10
CARTOON_ISLAND_FRAC = 0.01


def _cartoon_silhouette(slug, g, Q, mask, extent, grid):
    """One smoothed silhouette for a group: the mask, the depth field and the keyline rings together.

    Returns a dict with the PADDED depth `q` and `mask`, the padded `extent` (Angstrom), the keyline
    `rings` in Angstrom, the chosen `sigma_A`, and the guard numbers `area_change` and `islands`.
    Cached per (slug, group), so a sheet that draws a structure several times pays once.
    """
    key = f"_cartoon::{slug}::{g}"
    if key in _STRUCT_CACHE:
        return _STRUCT_CACHE[key]
    import contourpy
    m0 = np.asarray(mask, bool)
    a0 = max(int(m0.sum()), 1)
    h0, w0 = m0.shape
    x0, y0, x1, y1 = extent
    pw, ph = (x1 - x0) / w0, (y1 - y0) / h0

    def n_sig(m):
        lab, n = ndimage.label(m)
        if not n:
            return 0
        return int((np.bincount(lab.ravel())[1:] >= CARTOON_ISLAND_FRAC * a0).sum())

    n0 = n_sig(m0)
    pad = int(math.ceil(3.0 * max(CARTOON_SIGMA_LADDER_A) / grid)) + 2
    mp = np.pad(m0, pad)
    pick, field = 0.0, mp.astype(np.float32)
    for s_a in CARTOON_SIGMA_LADDER_A:
        f = ndimage.gaussian_filter(mp.astype(np.float32), s_a / grid)
        ms = f > 0.5
        if n_sig(ms) == n0 and 1.0 - ms.sum() / a0 <= CARTOON_MAX_AREA_LOSS:
            pick, field = s_a, f
            break
    ms = field > 0.5
    lab, n = ndimage.label(ms)
    if n:
        keep = np.bincount(lab.ravel()) >= CARTOON_ISLAND_FRAC * a0
        keep[0] = False
        ms = keep[lab]
    ms = ndimage.binary_fill_holes(ms)
    fc = field.copy()
    fc[ms & (fc <= 0.5)] = 1.0
    fc[~ms & (fc > 0.5)] = 0.0
    qp = np.pad(np.asarray(Q, np.float32), pad)
    idx = ndimage.distance_transform_edt(~mp, return_distances=False, return_indices=True)
    qf = qp[tuple(idx)]
    if pick:
        qf = ndimage.gaussian_filter(qf, pick / grid)
    X0, Y0 = x0 - pad * pw, y0 - pad * ph
    rings = []
    for ln in contourpy.contour_generator(z=fc, line_type="Separate").lines(0.5):
        if len(ln) < 4:
            continue
        rings.append(np.column_stack([X0 + (ln[:, 0] + 0.5) * pw, Y0 + (ln[:, 1] + 0.5) * ph]))
    out = dict(q=qf, mask=ms, extent=(X0, Y0, x1 + pad * pw, y1 + pad * ph), rings=rings,
               sigma_A=pick, area_change=float(ms.sum() / a0 - 1.0), islands=(n0, n_sig(ms)))
    _STRUCT_CACHE[key] = out
    return out


def surface_render(ax, x, y, w=1.30, *, slug, coords="axes", zorder=6, tr=None, at=(0.0, 0.0),
                   tokens=None, order=None, flip=False, label=None, outline=None, lw=None,
                   shadow=0.022, shade_kw=None, allow_placed=False, relief=0.0,
                   style=None):
    """A shaded ATOMIC-DETAIL molecular surface. Falls back to `surface()` when it cannot be drawn.

    Three fallbacks, all deliberate:
      * `glyphs.STYLE == "flat"` -- the module-wide pure-vector mode. A shaded surface is a raster,
        and `STYLE="flat"` exists so a caller can demand vector output.
      * no `.npz` sidecar for this structure.
      * drawn width below `SURFACE_RENDER_MIN_IN`, where the atomic relief becomes noise.

    🔴 `relief` IS THE THIRD TIER, and it is the one to reach for at manuscript size. Passing a
    smoothing length in Angstrom (`relief=3.0` is the default default look) low-passes the height
    field, so the silhouette keeps its shape and gains lighting without the atomic granularity —
    a lit silhouette rather than a photograph of a surface. Two consequences follow from the
    smoothing rather than from taste:
    """
    cartoon = style == "cartoon"
    if cartoon:
        if not relief:
            relief = CARTOON_RELIEF
    elif style is not None:
        raise ValueError(f"unknown surface style {style!r}; expected 'cartoon' or None")
    doc = load_structure(slug)
    dep = _load_depth(slug)
    floor = SURFACE_RELIEF_MIN_IN if relief else SURFACE_RENDER_MIN_IN
    if outline is None:
        outline = bool(relief)
    if dep is None or G.STYLE == "flat" or w < floor:
        return surface(ax, x, y, w, slug=slug, coords=coords, zorder=zorder, tr=tr, at=at,
                       tokens=tokens, order=order, flip=flip, label=label, lw=lw)
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    lw = keyline_lw(w) if lw is None else lw
    ax0, ay0 = at
    e = doc["extent"]
    span = max(e[2] - e[0], e[3] - e[1])
    s = w / span
    cx, cy = (e[0] + e[2]) / 2.0, (e[1] + e[3]) / 2.0
    mir = -1.0 if flip else 1.0
    hint = doc.get("draw_this_way") or {}
    tok = dict(STRUCT_TOKENS)
    tok.update(hint.get("tokens") or {})
    tok.update(tokens or {})
    names = order or hint.get("order") or list(doc["chains"])
    placed_groups = [g for g in names
                     if (doc["chains"].get(g) or {}).get("evidence") == "placed"]
    if placed_groups and not (label or allow_placed):
        raise ValueError(
            f"{slug}: group(s) {placed_groups} are HAND-PLACED, not solved. Draw them only with a "
            f"`label=` naming the arrangement as modelled, or pass allow_placed=True having said so "
            f"in the figure some other way. A placed interface rendered like a solved one asserts a "
            f"structure that does not exist.")
    out = {"groups": {}, "placed": placed_groups}
    st = G._keep_limits(ax)
    for i, g in enumerate(names):
        blk = doc["chains"].get(g)
        if not blk or "depth" not in blk or f"{g}__q" not in dep:
            continue
        d = blk["depth"]
        Q, mask = dep[f"{g}__q"], dep[f"{g}__m"].astype(bool)
        sil = None
        if cartoon:
            sil = _cartoon_silhouette(slug, g, Q, mask, d["extent"], d["grid"])
            Q, mask = sil["q"], sil["mask"]
        _sk = dict(shade_kw or {})
        if relief:
            _sk.setdefault("smooth_A", float(relief))
            _sk.setdefault("spec", 0.18)
            _sk.setdefault("shininess", 14.0)
        rgba = shade_depth(Q, mask, d["z_range"], tok.get(g, PROTEIN_BODY), grid_A=d["grid"], **_sk)
        x0, y0, x1, y1 = sil["extent"] if sil else d["extent"]
        ex = [ax0 + mir * (x0 - cx) * s, ax0 + mir * (x1 - cx) * s,
              ay0 + (y0 - cy) * s, ay0 + (y1 - cy) * s]
        if mir < 0:
            ex[0], ex[1] = ex[1], ex[0]
            rgba = rgba[:, ::-1]
            mask = mask[:, ::-1]
        if shadow and i == 0 and G.EFFECTS:
            sh = ndimage.gaussian_filter(mask.astype(np.float32), 3.0)
            sm = np.zeros(mask.shape + (4,), np.float32)
            sm[..., :3] = np.array(G._rgb(INK))
            sm[..., 3] = np.clip(sh, 0, 1) * 0.30
            d_in = shadow
            sx = ax.imshow(sm, extent=[ex[0] + d_in, ex[1] + d_in, ex[2] - d_in, ex[3] - d_in],
                           origin="lower", interpolation="bilinear", aspect="auto",
                           transform=tr, zorder=zorder - 0.4, clip_on=False)
            sx.set_rasterized(True)
            exempt(sx)
        im = ax.imshow(rgba, extent=ex, origin="lower", interpolation="bilinear",
                       aspect="auto", transform=tr, zorder=zorder + i * 0.6, clip_on=False)
        im.set_rasterized(True)
        exempt(im)
        if outline:
            for poly in (sil["rings"] if sil else blk.get("outer", [])):
                a = np.asarray(poly, float)
                pts = np.column_stack([ax0 + mir * (a[:, 0] - cx) * s, ay0 + (a[:, 1] - cy) * s])
                p = Polygon(pts, closed=True, transform=tr, clip_on=False, fc="none",
                            ec=(outline if isinstance(outline, str)
                                else G.shade(tok.get(g, PROTEIN_BODY), -0.45)),
                            lw=lw, zorder=zorder + i * 0.6 + 0.3)
                ax.add_patch(p)
        gc = (ax0, ay0)
        if blk.get("outer"):
            a = np.asarray(blk["outer"][0], float)
            px = ax0 + mir * (a[:, 0] - cx) * s
            py = ay0 + (a[:, 1] - cy) * s
            gc = (float(px.mean()), float(py.mean()))
        out["groups"][g] = dict(centre=gc, top=(gc[0], gc[1]),
                                token=tok.get(g, PROTEIN_BODY), evidence=blk.get("evidence"),
                                cartoon_sigma_A=(sil["sigma_A"] if sil else None))
    G._restore_limits(ax, st)
    if label:
        ax.text(ax0, ay0 - (e[3] - e[1]) * s / 2.0 - 0.055, label, transform=tr, ha="center",
                va="top", fontsize=TYPE["in_figure"], color=INK, zorder=zorder + 6, clip_on=False)
    out["centre"] = (ax0, ay0)
    out["width_in"] = w
    return out
