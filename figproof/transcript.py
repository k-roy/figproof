"""figproof.transcript — the default ILLUSTRATED register for RNA and the proteins that act on it.

SIZE: `size_for(slug)` draws a surface at k * sqrt(its longest extent in A), floored above `SURFACE_RELIEF_MIN_IN`. A stated compression
(the default `scale="schematic"` rule for cartoons): larger machines stay larger, but a 200 A RNP does not dwarf a 60 A enzyme. A figure that
claims a common physical scale must not use it (Fig 5 does not; it uses one inches-per-Angstrom constant).

Coordinates are the caller's data units; every length argument is in INCHES, converted with the axes' own units-per-inch, so the same call
draws the same glyph on a 0-100 panel and on a slide canvas in inches.
"""
from __future__ import annotations

import math

import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Ellipse, Polygon

from . import glyphs as G
from . import molecules as M
from . import tokens as TK

INK, SUBTLE, HAIR = TK.colour("neutral.ink"), TK.colour("neutral.subtle"), TK.colour("neutral.hairline")
MUTE, PAPER = TK.colour("neutral.mute"), TK.colour("neutral.paper")
POLYA, SPLICE = TK.colour("mark.polya"), TK.colour("mark.splice")
ST = G.STROKE

SURFACE_STYLE = dict(relief=3.0, shadow=0.0, shade_kw=dict(ambient=0.76, diffuse=0.28, spec=0.0, ao=0.18, rim=0.12))
WOBBLE_IN = 0.030
TICK_IN, TICK_EVERY_IN = 0.030, 0.19
BEAD_D_IN = 0.125
BEAD_PITCH_IN = 0.150
SIZE_K, SIZE_FLOOR_IN = 0.046, 0.36
RNA_IN_MACHINE = "#A6A6A6"


def upi(ax):
    """Data units per inch, (x, y), from the axes' current limits and its size on the page."""
    bb = ax.get_position(); fw, fh = ax.figure.get_size_inches()
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    return (x1 - x0) / (bb.width * fw), (y1 - y0) / (bb.height * fh)


def wobble(ax, x0, x1, y, seed=0, *, amp_in=WOBBLE_IN, taper_in=0.18, n=600, periods=(0.62, 0.37, 0.21), weights=(1.0, 0.55, 0.30)):
    ux, uy = upi(ax)
    xs = np.linspace(x0, x1, n)
    r = np.random.default_rng(seed)
    yy = np.zeros_like(xs)
    for per, wt in zip(periods, weights):
        yy += wt * np.sin(2 * np.pi * ((xs - x0) / ux) / per + r.uniform(0, 2 * np.pi))
    yy /= max(np.abs(yy).max(), 1e-9)
    t = np.clip(np.minimum(xs - x0, x1 - xs) / (taper_in * ux), 0, 1)
    return xs, y + amp_in * uy * yy * t


class Strand:
    """One RNA along x. `parts` = [(x_a, x_b, kind)], kind in rna | exon | focus | lost; unlisted spans draw as `rna`."""

    STYLE = {"rna": (HAIR, "secondary", "solid", 2.0), "exon": (INK, "primary", "solid", 3.0),
             "focus": (INK, "emphasis", "solid", 3.2), "lost": (MUTE, "secondary", (0, (2.4, 1.8)), 2.0)}

    def __init__(self, ax, x0, x1, y, *, seed=0, parts=(), ticks=True, amp_in=WOBBLE_IN, base="rna", z=2.0):
        self.ax, self.x0, self.x1, self.y0 = ax, x0, x1, y
        self.ux, self.uy = upi(ax)
        self.xs, self.ys = wobble(ax, x0, x1, y, seed, amp_in=amp_in)
        segs = [(x0, x1, base)] + list(parts)
        for a, b, kind in segs:
            col, tier, ls, zz = self.STYLE[kind]
            m = (self.xs >= a - 1e-9) & (self.xs <= b + 1e-9)
            ax.plot(self.xs[m], self.ys[m], color=col, lw=ST[tier], ls=ls, zorder=z + zz / 10, solid_capstyle="round")
        if ticks:
            g = np.gradient(self.ys, self.xs)
            for xx in np.arange(x0 + 0.5 * TICK_EVERY_IN * self.ux, x1 - 0.2 * TICK_EVERY_IN * self.ux, TICK_EVERY_IN * self.ux):
                k = int(np.clip(np.searchsorted(self.xs, xx), 0, len(self.xs) - 1)); yy = self.y(xx)
                nx, ny = -g[k] / self.ux * self.uy, 1.0
                nn = math.hypot(nx, ny)
                ax.plot([xx, xx - TICK_IN * self.ux * nx / nn], [yy, yy - TICK_IN * self.uy * ny / nn], color=HAIR,
                        lw=ST["hairline"], zorder=z + 0.05, solid_capstyle="butt")

    def y(self, x):
        return float(np.interp(x, self.xs, self.ys))

    def slope_deg(self, x):
        g = np.gradient(self.ys, self.xs); k = int(np.clip(np.searchsorted(self.xs, x), 0, len(self.xs) - 1))
        return math.degrees(math.atan2(g[k] / self.uy, 1.0 / self.ux))


def bead(ax, s, x, letter, *, ring=None, bold=False, size=None, y=None, z=5.0):
    """A base as a white disc ON the strand, lettered; `ring` (a layer-B identity, e.g. SPLICE for the branch A) marks a special base."""
    ux, uy = upi(ax); yy = s.y(x) if y is None else y
    d = BEAD_D_IN
    ax.add_patch(Ellipse((x, yy), d * ux, d * uy, facecolor=PAPER, edgecolor=ring or HAIR,
                         lw=ST["primary"] if ring else ST["hairline"], zorder=z))
    ax.text(x, yy - 0.004 * uy, letter, ha="center", va="center", fontsize=size or 9.5, color=INK,
            fontweight="bold" if bold or ring else "normal", zorder=z + 0.1)


def beads(ax, s, x_start, seq, *, pitch_in=BEAD_PITCH_IN, rings=None, size=None):
    ux, _ = upi(ax); rings = rings or {}
    xs = [x_start + i * pitch_in * ux for i in range(len(seq))]
    for i, (x, ch) in enumerate(zip(xs, seq)):
        bead(ax, s, x, ch, ring=rings.get(i), size=size)
    return xs


def cap(ax, s, x, *, r_in=0.045):
    ux, uy = upi(ax)
    ax.add_patch(Ellipse((x, s.y(x)), 2 * r_in * ux, 2 * r_in * uy, facecolor=INK, edgecolor="none", zorder=6))


def end5(ax, s, x, chem="p", *, size=None):
    """The 5′ end a panel is about: the default end triangle ABOVE the line, tip down, with its chemistry to the left."""
    ux, uy = upi(ax); yy = s.y(x)
    tip, w, h = 0.07, 0.085, 0.07
    ax.add_patch(Polygon([(x, yy + tip * uy), (x - w / 2 * ux, yy + (tip + h) * uy), (x + w / 2 * ux, yy + (tip + h) * uy)],
                         closed=True, facecolor=INK, edgecolor="none", zorder=7))
    if chem:
        ax.text(x - 0.045 * ux, yy + 0.005 * uy, chem, ha="right", va="center", fontsize=size or 11, color=INK, zorder=7)


def polya(ax, s, x, *, n=4, native=True, size=None):
    """A lettered poly(A) tail starting at x on the strand's line: native = solid `polya` letters; ePAP-added = the same letters HOLLOW,
    outlined in `polya`, so the tail's origin is read from form. Returns the x where the letters end."""
    ux, uy = upi(ax); fs = size or 11
    t = ax.text(x + 0.02 * ux, s.y(x), "A" * n, ha="left", va="center", fontsize=fs, fontweight="bold", zorder=8,
                color=POLYA if native else PAPER)
    if native:
        t.set_path_effects([pe.withStroke(linewidth=2.6, foreground=PAPER, alpha=0.9)])
    else:
        t.set_path_effects([pe.withStroke(linewidth=1.3, foreground=POLYA), pe.Normal()])
    return x + 0.02 * ux + n * fs / 72 * 0.72 * ux


def splice_tick(ax, s, x, *, up=True, h_in=0.13):
    """A splice site as Fig 5 draws it: a short `splice` bar across the strand."""
    ux, uy = upi(ax); yy = s.y(x)
    y1 = yy + (h_in if up else -h_in) * uy
    ax.plot([x, x], [yy - (0.03 if up else -0.03) * uy, y1], color=SPLICE, lw=ST["emphasis"], zorder=6, solid_capstyle="butt")


def cpa(ax, s, x, *, h_in=0.16, open_=False):
    ux, uy = upi(ax); yy = s.y(x)
    ax.plot([x, x], [yy, yy + h_in * uy], color=POLYA, lw=ST["primary"], zorder=6, solid_capstyle="butt")
    ax.add_patch(Ellipse((x, yy + h_in * uy), 0.075 * ux, 0.075 * uy, facecolor=PAPER if open_ else POLYA, edgecolor=POLYA,
                         lw=ST["primary"], zorder=6.5))


def cut(ax, s, x, *, h_in=0.075):
    """Where the enzyme cuts: a short ink tick across the strand, square to it."""
    ux, uy = upi(ax); yy = s.y(x); a = math.radians(s.slope_deg(x) + 90)
    dx, dy = math.cos(a) * h_in * ux, math.sin(a) * h_in * uy
    ax.plot([x - dx, x + dx], [yy - dy, yy + dy], color=INK, lw=ST["emphasis"], zorder=7, solid_capstyle="butt")


def lariat(ax, x_bp, y, *, loop_in=0.40, tail_to=None, seed=0, focus=False, parts=(), bp_letter="A"):
    """An intron lariat: a gently wobbling loop to the LEFT of the branch A, closed there, and its tail to `tail_to`.
    Returns (tail Strand or None, loop centre)."""
    ux, uy = upi(ax)
    col, tier = (INK, "emphasis") if focus else (HAIR, "secondary")
    r = loop_in / 2
    cx, cy = x_bp - r * 0.92 * ux, y + r * 0.55 * uy
    th = np.linspace(-0.35, 2 * np.pi - 0.35, 300)
    rng = np.random.default_rng(seed)
    w = 1 + 0.045 * np.sin(3 * th + rng.uniform(0, 6.28)) + 0.03 * np.sin(5 * th + rng.uniform(0, 6.28))
    xs, ys = cx + r * w * np.cos(th) * ux, cy + r * 0.80 * w * np.sin(th) * uy
    xs[0] = xs[-1] = x_bp; ys[0] = ys[-1] = y
    ax.plot(xs, ys, color=col, lw=ST[tier], zorder=2.3, solid_capstyle="round")
    tail = None
    if tail_to is not None:
        tail = Strand(ax, x_bp, tail_to, y, seed=seed + 1, parts=parts, base="focus" if focus else "rna", ticks=True)
    bead(ax, tail if tail else _Flat(y), x_bp, bp_letter, ring=SPLICE)
    return tail, (cx, cy)


class _Flat:
    def __init__(self, y):
        self._y = y

    def y(self, x):
        return self._y


def hairpin(ax, s, x, *, h_in=0.30, gap_in=0.075, loop_in=0.11, rungs=5, focus=False):
    """A stem-loop standing on the strand at x: two rails with base-pair rungs, closed by a loop."""
    ux, uy = upi(ax); yy = s.y(x)
    col, tier = (INK, "emphasis") if focus else (HAIR, "secondary")
    xl, xr = x - gap_in / 2 * ux, x + gap_in / 2 * ux
    for xx in (xl, xr):
        ax.plot([xx, xx], [yy, yy + h_in * uy], color=col, lw=ST[tier], zorder=2.4)
    for k in range(1, rungs + 1):
        yk = yy + h_in * uy * k / (rungs + 1)
        ax.plot([xl, xr], [yk, yk], color=col, lw=ST["hairline"], zorder=2.4)
    ax.add_patch(Ellipse((x, yy + (h_in + loop_in * 0.42) * uy), loop_in * ux, loop_in * uy, facecolor="none", edgecolor=col,
                         lw=ST[tier], zorder=2.4))
    return xl, xr, yy + h_in * uy


def size_for(slug, *, k=SIZE_K, floor_in=SIZE_FLOOR_IN):
    e = M.load_structure(slug)["extent"]
    return max(floor_in, k * math.sqrt(max(e[2] - e[0], e[3] - e[1])))


def dock(ax, slug, x, y_strand, w=None, *, gap_in=0.05, behind=True, ghost=False, label=None, label_size=11, order=None,
         tokens=None, allow_placed=False, label_dy_in=0.04):
    ""
    ux, uy = upi(ax); w = w or size_for(slug)
    doc = M.load_structure(slug); e = doc["extent"]
    sc = w / max(e[2] - e[0], e[3] - e[1])
    ax_, ay_ = (doc.get("anchors") or {}).get("rna_site") or ((e[0] + e[2]) / 2.0, e[1])
    dx, dy = (ax_ - (e[0] + e[2]) / 2.0) * sc, (ay_ - (e[1] + e[3]) / 2.0) * sc
    cx, cy = x - dx * ux, y_strand - (dy - gap_in) * uy
    n_im, n_pa, n_li = len(ax.images), len(ax.patches), len(ax.lines)
    kw = dict(SURFACE_STYLE)
    prot = [g for g, v in doc["chains"].items() if "trace" in v]
    if order is None:
        order = [g for g in (doc.get("draw_this_way") or {}).get("order", list(doc["chains"]))
                 if g in prot or not any(k in g for k in ("product", "substrate", "pre_trna", "target"))]
    if tokens is None:
        tokens = {g: (M.PROTEIN_BODY if g in prot else RNA_IN_MACHINE) for g in doc["chains"]}
    if ghost:
        tokens = {g: "#E4E4E4" for g in doc["chains"]}
    out = M.surface_render(ax, cx, cy, w, slug=slug, coords="data", zorder=1.2 if behind else 7, order=order, tokens=tokens,
                           allow_placed=allow_placed, **kw)
    if ghost:
        for art in ax.images[n_im:]:
            art.set_alpha(0.40)
        for art in list(ax.patches[n_pa:]) + list(ax.lines[n_li:]):
            art.set_alpha(0.55); art.set_linestyle((0, (3, 2)))
    h = (e[3] - e[1]) * sc
    top = cy + h / 2 * uy
    if label:
        ax.text(cx, top + label_dy_in * uy, label, ha="center", va="bottom", fontsize=label_size,
                color=SUBTLE if ghost else INK, zorder=8)
    out.update(top=top, centre=(cx, cy), width_in=w, height_in=h)
    return out
