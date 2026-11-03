"""figproof.schematics — small "what is measured" pictures that ride beside a slide's row or column labels.

Each schematic is a callable `fn(ax, width_in)` that draws into a reserved axes whose data range is 0..1 in both directions and whose width
on the page is `width_in` inches. `figproof.slides.grid` creates that axes and calls it; nothing here places itself.
"""
from __future__ import annotations

from matplotlib.patches import Polygon

from . import tokens as T

TRI_W, TRI_H, TRI_TIP = 0.090, 0.075, 0.080
R_CAP = 0.042
DASH = (0, (2.2, 1.6))


def _mods():
    from . import glyphs as G, molecules as M
    return G, M


def _track(ax, width_in, y=0.5):
    """A molecules.Track across the schematic axes, anchored in axes fractions so the caller's box sets the scale."""
    _, M = _mods()
    return M.Track(ax, 0.02, y, width=width_in * 0.96, bp0=0, bp1=100, coords="axes")


def _tri(t, bp, dy=0.0, filled=True, end="5p"):
    """An end mark: filled is the OBSERVED end, open is the TRUE end. `end="5p"` sits above the row line and `end="3p"` below it, each
    tip pointing at the line, so the side of the line says which end is meant."""
    G, _ = _mods()
    ink, paper = T.colour("neutral.ink"), T.colour("neutral.paper")
    sgn = 1.0 if end == "5p" else -1.0
    x, yt = t.x(bp), dy + sgn * TRI_TIP
    pts = [(x, yt), (x - TRI_W / 2, yt + sgn * TRI_H), (x + TRI_W / 2, yt + sgn * TRI_H)]
    t.ax.add_patch(Polygon(pts, closed=True, transform=t.tr, facecolor=ink if filled else paper,
                           edgecolor="none" if filled else ink, lw=0 if filled else G.STROKE["secondary"],
                           joinstyle="miter", zorder=8, clip_on=False))


def _cap(t, bp, dy=0.0):
    from matplotlib.patches import Circle
    t.ax.add_patch(Circle((t.x(bp), dy), R_CAP, transform=t.tr, facecolor=T.colour("neutral.ink"), edgecolor="none",
                          zorder=8, clip_on=False))


def _rna(t, a, b, *, dashed=False):
    _, M = _mods()
    n0 = len(t.ax.lines)
    M.strand(t, a, b, kind="rna", y=0.0, detail=0, color=T.colour("neutral.subtle"))
    if dashed:
        for ln in t.ax.lines[n0:]:
            ln.set_linestyle(DASH)


def _polya(t, a, b):
    _, M = _mods()
    M.region_arc(t, a, b, T.colour("mark.polya"), y=0.0, side="top", label=None)


def full_length(ax, width_in):
    """A capped full-length molecule: the observed 5′ end IS the cap, and the tail is intact."""
    t = _track(ax, width_in)
    _rna(t, 4, 78)
    _polya(t, 78, 96)
    _cap(t, 4)
    _tri(t, 4)


def broken_5p(ax, width_in):
    """A molecule broken in its 5′ part: what is lost is DASHED, and the observed 5′ end is the break, not a cap."""
    t = _track(ax, width_in)
    _rna(t, 4, 34, dashed=True)
    _rna(t, 34, 78)
    _polya(t, 78, 96)
    _tri(t, 34)


def read_stop(ax, width_in):
    """The read stopped short of the molecule's real end: the observed end (filled) is not the true end (open), and the gap is dashed."""
    t = _track(ax, width_in)
    _rna(t, 4, 26, dashed=True)
    _rna(t, 26, 78)
    _polya(t, 78, 96)
    _cap(t, 4)
    _tri(t, 4, filled=False)
    _tri(t, 26)


def decay_intermediate(ax, width_in):
    """A decay intermediate: broken at the 5′ side and already deadenylated, so there is no tail to draw."""
    t = _track(ax, width_in)
    _rna(t, 4, 40, dashed=True)
    _rna(t, 40, 92)
    _tri(t, 40)


RNA_END_CLASSES = {"full_length": full_length, "broken_5p": broken_5p, "read_stop": read_stop,
                   "decay_intermediate": decay_intermediate}


EXON_H = 0.055
TICK_H = 0.150
SS_TRI = 0.080


def _exon(t, a, b, dy=0.0, color=None):
    from matplotlib.patches import Rectangle
    lo, hi = t.span(a, b)
    t.ax.add_patch(Rectangle((lo, dy - EXON_H / 2.0), hi - lo, EXON_H, transform=t.tr,
                             facecolor=color or T.colour("neutral.ink"), edgecolor="none", zorder=6, clip_on=False))


def _intron(t, a, b, dy=0.0, color=None):
    G, _ = _mods()
    lo, hi = t.span(a, b)
    t.ax.plot([lo, hi], [dy, dy], transform=t.tr, color=color or T.colour("neutral.subtle"),
              lw=G.STROKE["hairline"], zorder=5, clip_on=False, solid_capstyle="butt")


def _splice_site(t, bp, dy=0.0):
    """The 5′ splice site: a tick in the `splice` identity with a small triangle pointing INTO the intron."""
    from matplotlib.patches import Polygon
    G, _ = _mods()
    col = T.colour("mark.splice")
    x, top = t.x(bp), dy + EXON_H / 2.0 + TICK_H
    t.ax.plot([x, x], [dy + EXON_H / 2.0, top], transform=t.tr, color=col, lw=G.STROKE["secondary"], zorder=8,
              clip_on=False, solid_capstyle="butt")
    t.ax.add_patch(Polygon([(x, top), (x + SS_TRI, top), (x, top - SS_TRI)], closed=True, transform=t.tr,
                           facecolor=col, edgecolor="none", zorder=8, clip_on=False))


def free_exon1(ax, width_in):
    """The first-step splicing product: exon 1 released at the 5′ splice site, its observed 3′ end tailed.

    Two rows. Above, the pre-mRNA as a gene model (exon 1, intron, exon 2) with the 5′ splice site marked in the `splice` identity. Below,
    free exon 1 on its own, ending exactly AT that splice site, with the observed 3′ end as a filled triangle BELOW the line (the side of
    the line is what separates it from the 5′ marks) and a short poly(A) over it (ePAP tailing, or a natural tail). The lariat-exon 2 intermediate is deliberately not drawn: at this width it crowds the row, and
    `lariat` draws it on its own when a slide needs it.
    """
    top = _track(ax, width_in, y=0.68)
    _exon(top, 4, 30)
    _intron(top, 30, 62)
    _exon(top, 62, 96)
    _splice_site(top, 30)
    bot = _track(ax, width_in, y=0.30)
    _exon(bot, 4, 30)
    _polya(bot, 30, 44)
    _tri(bot, 30, end="3p")


def lariat(ax, width_in):
    """The excised intron lariat: a loop closed at the branch point, with its 3′ tail (the dbr1-delta and lariat slides).

    The branch point is a dot in the `splice` identity at the closure, which is what makes the shape a lariat rather than a circle with a
    line beside it.
    """
    from matplotlib.patches import Circle, Ellipse
    G, _ = _mods()
    t = _track(ax, width_in, y=0.5)
    col, ink = T.colour("mark.splice"), T.colour("neutral.subtle")
    lo, hi = t.span(6, 40)
    cx, w = (lo + hi) / 2.0, (hi - lo)
    t.ax.add_patch(Ellipse((cx, 0.0), w, w * 0.78, transform=t.tr, facecolor="none", edgecolor=ink,
                           lw=G.STROKE["secondary"], zorder=6, clip_on=False))
    bx = cx + w / 2.0
    tx = t.span(40, 96)[1]
    t.ax.plot([bx, tx], [0.0, 0.0], transform=t.tr, color=ink, lw=G.STROKE["secondary"], zorder=6,
              clip_on=False, solid_capstyle="round")
    t.ax.add_patch(Circle((bx, 0.0), R_CAP * 0.9, transform=t.tr, facecolor=col, edgecolor="none", zorder=8, clip_on=False))


RNA_END_CLASSES["free_exon1"] = free_exon1
RNA_END_CLASSES["lariat"] = lariat
