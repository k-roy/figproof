"""glyphs.py -- the schematic glyph vocabulary: organisms, cultures, conditions, hardware.

Sibling of `alignment_grammar.py` and imported the same way::

--------------------------------------------------------------------------------------------------
THE THREE THINGS THIS MODULE EXISTS TO GET RIGHT
--------------------------------------------------------------------------------------------------

2. TYPE AND STROKE FLOORS.  ``CAP_MIN_MM`` and ``STROKE_MIN_PT`` are read from ``tokens.json`` and
   enforced by ``check_type_floor`` / ``check_stroke_floor``, which walk the rendered figure.  A
   constant nobody reads is not a standard.  Smaller type requires an explicit ``exempt()``.

3. VOLUME, NOT DECORATION.  BioRender's molecular art reads as three-dimensional because every
   object carries a base fill, a soft interior gradient, a slightly darker rim, one specular
   highlight and a low contact shadow -- in that order.  ``_body()`` applies exactly that stack to
   any path.  The colour contract is untouched: a gradient only varies the LIGHTNESS of the token
   colour, never its hue, so `barcode` amber is still `barcode` amber and the CVD separations in
   `tokens.json` still hold.  Set ``STYLE = "flat"`` for pure-vector output with no raster gradient.

--------------------------------------------------------------------------------------------------
THE CONTRACT THIS MODULE DOES NOT TOUCH
--------------------------------------------------------------------------------------------------
* One colour = one molecular identity, forever (`tokens.json` -> `meaning`).
* Structure is never coloured.  A cell wall, a flask, a plate, a plasmid backbone is `ink`/`rule`/
  `mute`.  Hue is spent only on molecular identity -- which is why ``yeast()`` takes the payload
  colour as ``fill`` and defaults to no fill at all.
* Two neutrals + at most FOUR semantic accents per panel.
* Never `#000000`.  Shadows are `ink` at low alpha.

--------------------------------------------------------------------------------------------------
GLYPHS
--------------------------------------------------------------------------------------------------
    yeast(ax, x, y, d)          budded S. cerevisiae -- TRUE UNION outline with a real bud neck,
                                visible nucleus, optional vacuole.  State by fill (rule 5).
    flask(ax, x, y, h)          Erlenmeyer culture holding budded cells at density.
    sun / moon / condition      the light and dark condition marks for the whole project.
    plate(ax, x, y, d)          petri dish + colonies, for the clone-sampling steps.
    plate96(ax, x, y, w)        12 x 8 microplate, for the 96-clone sampling steps.
    plasmid(ax, x, y, d)        ink backbone, hue only on the features riding on it.
    colony / disc / droplet     shared primitives.
"""
from __future__ import annotations

import colorsys
import json
import math
import os

import numpy as np
from matplotlib.patches import Circle, Ellipse, PathPatch, Polygon, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Affine2D, ScaledTranslation

__all__ = [
    "TOK", "INK", "RULE", "MUTE", "PAPER", "GUIDE", "DONOR", "BARCODE", "RECRUIT", "EDIT",
    "PHOSPHO", "DOMAIN", "PAM", "TYPE", "SLIDE", "STROKE", "CAP_MIN_MM", "STROKE_MIN_PT", "DETAIL_MIN_PT", "CAP_RATIO", "STYLE",
    "inch_axes", "text_in", "shade", "cap_mm", "pt_for_cap_mm", "standard_rc", "exempt",
    "check_type_floor", "check_stroke_floor", "check_floors",
    "CELL", "POOL_FILL", "AGAR", "COLONY", "NUCLEOPLASM", "SUN_FILL", "SUN_RIM", "MOON_FILL",
    "yeast", "tetrad", "flask", "tube", "vial", "spot_dilution", "sun", "moon", "condition", "condition_mark", "plate", "array_plate", "plate96", "plasmid", "colony",
    "disc", "droplet", "arrow",
    "pombe", "mouse", "human", "species", "species_row", "SPECIES",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "_views", "magestic_tokens.json")) as _fh:
    TOK = json.load(_fh)

_SEM, _STR = TOK["palette"]["semantic"], TOK["palette"]["structural"]
GUIDE, DONOR, BARCODE = _SEM["guide"], _SEM["donor"], _SEM["barcode"]
RECRUIT, EDIT, PHOSPHO = _SEM["recruit"], _SEM["edit"], _SEM["phospho"]
PAM = _SEM["pam"]
DOMAIN = _SEM["domain"]
INK, RULE, MUTE, PAPER = _STR["ink"], _STR["rule"], _STR["mute"], _STR["paper"]
CELL, POOL_FILL, AGAR, COLONY = _STR["cell"], _STR["pool"], _STR["agar"], _STR["colony"]
NUCLEOPLASM = _STR["nucleoplasm"]
SUN_FILL, SUN_RIM, MOON_FILL = _STR["sun"], _STR["sun_rim"], _STR["moon"]

TYPE = dict(TOK["typography"]["manuscript_pt"])
SLIDE = dict(TOK["typography"]["slide_pt"])
STROKE = dict(TOK["geometry"]["stroke_pt"])
CAP_MIN_MM = TOK["typography"]["cap_min_mm"]
STROKE_MIN_PT = TOK["geometry"]["stroke_min_pt"]
CAP_RATIO = TOK["typography"]["cap_height_ratio"]

STYLE = "volume"
EFFECTS = True

_MM_PER_PT = 25.4 / 72.0

_DETAIL_MIN_IN = 0.34

DETAIL_MIN_PT = 0.55


def cap_mm(pt: float) -> float:
    """On-page cap height in mm for a font size in points."""
    return pt * CAP_RATIO * _MM_PER_PT


def pt_for_cap_mm(mm: float = None) -> float:
    """Font size in points whose cap height is `mm` (default: the default floor)."""
    return (CAP_MIN_MM if mm is None else mm) / (CAP_RATIO * _MM_PER_PT)


def standard_rc(matplotlib_module, *, slide: bool = False) -> dict:
    """Apply the default font family and the floored default sizes to rcParams.

    Does NOT set figure size, dpi or savefig behaviour -- those belong to the calling figure.
    """
    T = SLIDE if slide else TYPE
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": TOK["typography"]["family"],
        "font.size": T["in_figure"],
        "axes.labelsize": T["axis_label"],
        "xtick.labelsize": T["tick_label"],
        "ytick.labelsize": T["tick_label"],
        "legend.fontsize": T["annotation"],
        "axes.linewidth": STROKE["secondary"],
        "xtick.major.width": STROKE["secondary"],
        "ytick.major.width": STROKE["secondary"],
        "lines.solid_capstyle": "round",
        "mathtext.fontset": "custom",
        "mathtext.rm": TOK["typography"]["family"][0],
        "mathtext.it": TOK["typography"]["family"][0] + ":italic",
        "mathtext.bf": TOK["typography"]["family"][0] + ":bold",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
    matplotlib_module.rcParams.update(rc)
    return rc


_EXEMPT_GID = "magestic-floor-exempt"


def exempt(artist):
    """Mark an artist as deliberately below a floor, so the checkers skip it.

    The floors bound what a READER HAS TO FOLLOW -- a label, a rule, the outline of a primary
    object.  Three things are therefore exempt, and only these three:

      * the provenance footer, the stamp and the concise legend (`fig.texts`), whose sizes
        `the figure standard` fixes at 5.5-9 pt -- auto-exempt, no call needed;
      * a POOL MEMBER drawn as multiplicity (`the schematic module` rule 3) -- ``flask`` marks its own;
      * ORGANELLE DETAIL inside a cell -- a nucleus rim on a 2 mm cell is detail, and floored to
        0.6 pt it would swallow the organelle it outlines.

    Anything else must say so out loud by calling this.
    """
    artist.set_gid(_EXEMPT_GID)
    return artist


def _is_exempt(artist) -> bool:
    return artist.get_gid() == _EXEMPT_GID


def check_type_floor(fig, *, floor_mm: float = None, verbose: bool = True):
    """Every Text in `fig` whose cap height is below the floor.  Returns a list of (where, pt, mm).

    `fig.texts` (the provenance footer, the stamp, the concise legend) is auto-exempt.
    """
    floor = CAP_MIN_MM if floor_mm is None else floor_mm
    tol = 1e-3
    bad, auto = [], set(id(t) for t in fig.texts)
    for ax in fig.axes:
        for t in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            if t is None or not t.get_text().strip() or _is_exempt(t) or id(t) in auto:
                continue
            if not t.get_visible():
                continue
            pt = t.get_fontsize()
            if cap_mm(pt) < floor - tol:
                bad.append((t.get_text().replace("\n", " ")[:40], round(pt, 2),
                            round(cap_mm(pt), 2)))
        if not ax.axison:
            continue
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            if not t.get_text().strip() or _is_exempt(t) or not t.get_visible():
                continue
            if cap_mm(t.get_fontsize()) < floor - tol:
                bad.append(("tick:" + t.get_text()[:20], round(t.get_fontsize(), 2),
                            round(cap_mm(t.get_fontsize()), 2)))
    if verbose:
        print("glyphs: type floor %.2f mm cap (= %.1f pt) -- %s"
              % (floor, pt_for_cap_mm(floor), "OK" if not bad else "%d BELOW" % len(bad)))
        for w, pt, mm in bad[:12]:
            print("   %-42s %.1f pt  cap %.2f mm" % (w, pt, mm))
    return bad


def check_stroke_floor(fig, *, floor_pt: float = None, verbose: bool = True):
    """Every stroked artist below the stroke floor.  Pool members marked `exempt` are skipped."""
    floor = STROKE_MIN_PT if floor_pt is None else floor_pt
    bad = []
    for ax in fig.axes:
        for a in list(ax.patches) + list(ax.lines) + list(ax.collections):
            if _is_exempt(a):
                continue
            lw = a.get_linewidth()
            lw = float(np.atleast_1d(lw)[0]) if np.ndim(lw) else float(lw)
            ec = a.get_edgecolor() if hasattr(a, "get_edgecolor") else None
            if hasattr(a, "get_facecolor") and ec is not None and np.ndim(ec) <= 1:
                if len(np.atleast_1d(ec)) >= 4 and np.atleast_1d(ec)[-1] == 0:
                    continue
            if 0 < lw < floor - 1e-9:
                bad.append((type(a).__name__, round(lw, 3)))
    if verbose:
        print("glyphs: stroke floor %.2f pt -- %s"
              % (floor, "OK" if not bad else "%d BELOW" % len(bad)))
        for k, lw in bad[:12]:
            print("   %-42s %.2f pt" % (k, lw))
    return bad


def check_floors(fig, **kw):
    """Both floors.  Returns True when the figure is clean."""
    return not (check_type_floor(fig, **kw) or check_stroke_floor(fig, **kw))


def _tag_identity(ax):
    ""
    if getattr(ax, "_hf_layer", None) is None and not ax.axison:
        ax._hf_layer = "identity"


def inch_axes(ax, x: float, y: float, coords: str = "axes"):
    """A transform whose units are INCHES, with its origin at (x, y) in `coords`.

    `coords` is "axes" (fraction of the axes), "data", or "figure".  The transform is lazy: it is
    re-evaluated at draw time, so a later ``subplots_adjust`` moves the glyph with its axes and a
    circle stays round whatever the axes aspect.  Equivalent to
    ``matplotlib.transforms.offset_copy(..., units="inches")`` but anchored rather than offset.
    """
    _tag_identity(ax)
    fig = ax.figure
    base = {"axes": ax.transAxes, "data": ax.transData,
            "figure": fig.transFigure}[coords]
    return fig.dpi_scale_trans + ScaledTranslation(x, y, base)


def text_in(ax, x, y, dx, dy, s, *, coords="axes", size=None, color=None, **kw):
    """Text placed `dx`, `dy` INCHES from the glyph anchor -- the label companion to a glyph."""
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    return ax.text(dx, dy, s, transform=inch_axes(ax, x, y, coords),
                   fontsize=TYPE["in_figure"] if size is None else size,
                   color=INK if color is None else color, clip_on=False, **kw)


def _rgb(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))


def shade(c: str, amount: float) -> str:
    """Lighten (amount > 0) or darken (amount < 0) a colour, HUE UNCHANGED.

    The semantic contract is a contract about hue; volume is made from lightness.  This is what lets
    a glyph look three-dimensional without spending a second accent on it.
    """
    h, l, s = colorsys.rgb_to_hls(*_rgb(c))
    l = min(1.0, l + (1.0 - l) * amount) if amount >= 0 else max(0.0, l * (1.0 + amount))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))


def _mix(a, b, t):
    """Blend two colours in linear-ish sRGB.  `t` is how far toward `b`."""
    ra, rb = _rgb(a), _rgb(b)
    m = tuple(ra[i] + (rb[i] - ra[i]) * t for i in range(3))
    return "#%02X%02X%02X" % tuple(round(v * 255) for v in m)


def _rim(c):
    """The rim colour for a filled body: the fill darkened, never `ink` outlining everything."""
    return shade(c, -0.42)


def _keep_limits(ax):
    return ax.get_xlim(), ax.get_ylim(), ax.get_aspect(), ax.get_autoscale_on()


def _restore_limits(ax, st):
    xl, yl, asp, auto = st
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_aspect(asp)
    ax.set_autoscale_on(auto)


def _gradient(ax, patch, c_lo, c_hi, extent, tr, zorder, *, radial=False,
              center=(0.35, 0.72), falloff=1.15):
    """A soft interior gradient clipped to `patch`.  Lightness only -- see `shade`."""
    n = 220
    if radial:
        yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
        d = np.hypot(xx - center[0], yy - center[1]) / 1.05
        f = np.clip(1.0 - d, 0.0, 1.0) ** falloff
    else:
        f = np.linspace(0.0, 1.0, n)[:, None].repeat(n, 1)
    lo, hi = np.array(_rgb(c_lo)), np.array(_rgb(c_hi))
    arr = lo[None, None, :] + (hi - lo)[None, None, :] * f[:, :, None]
    st = _keep_limits(ax)
    im = ax.imshow(np.clip(arr, 0, 1), extent=extent, transform=tr, origin="lower",
                   interpolation="bicubic", aspect="auto", zorder=zorder)
    im.set_clip_path(patch)
    im.set_clip_box(None)
    exempt(im)
    _restore_limits(ax, st)
    return im


def _body(ax, path_or_patch, tr, *, fill, zorder, lw, edge=None, shadow=0.0, gloss=None,
          radial=True, extent=None, alpha=1.0, grad=(-0.16, 0.34), grad_center=(0.35, 0.72),
          grad_falloff=1.15):
    """base fill -> interior gradient -> rim -> specular highlight -> contact shadow.

    This five-step stack is the whole of the BioRender look, and every glyph in this module goes
    through it so that a cell, a colony and a flask read as members of one drawn world.
    """
    _tag_identity(ax)

    def _mk(**kw):
        if isinstance(path_or_patch, Path):
            return PathPatch(path_or_patch, transform=tr, clip_on=False, **kw)
        p = path_or_patch()
        p.set_transform(tr)
        p.set_clip_on(False)
        for k, v in kw.items():
            getattr(p, "set_" + k)(v)
        return p

    if shadow and EFFECTS:
        for _k, _al in ((1.00, 0.085), (1.035, 0.050), (1.070, 0.028)):
            sh = _mk(facecolor=INK, edgecolor="none", alpha=_al, linewidth=0, zorder=zorder - 0.3)
            sh.set_transform(Affine2D().scale(_k).translate(0.25 * shadow, -0.90 * shadow) + tr)
            exempt(sh)
            ax.add_patch(sh)

    base = _mk(facecolor=fill, edgecolor="none", linewidth=0, zorder=zorder, alpha=alpha)
    ax.add_patch(base)
    exempt(base)

    if STYLE == "volume" and extent is not None:
        lo, hi = grad
        _gradient(ax, base, shade(fill, lo), shade(fill, hi), extent, tr,
                  zorder + 0.1, radial=radial, center=grad_center, falloff=grad_falloff)

    if gloss and EFFECTS:
        gx, gy, gw, gh, ga = gloss[:5]
        g_peak = gloss[5] if len(gloss) > 5 else 0.55
        n = 128
        yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
        dd = np.hypot(xx - 0.5, yy - 0.5) * 2.0
        al = g_peak * np.clip(1.0 - dd, 0.0, 1.0) ** 2.2
        rgba = np.dstack([np.ones((n, n, 3)), al])
        st = _keep_limits(ax)
        hl = ax.imshow(rgba, extent=(-gw / 2, gw / 2, -gh / 2, gh / 2), origin="lower",
                       interpolation="bicubic", aspect="auto", zorder=zorder + 0.2,
                       transform=Affine2D().rotate_deg(ga).translate(gx, gy) + tr)
        hl.set_clip_path(base)
        hl.set_clip_box(None)
        exempt(hl)
        _restore_limits(ax, st)

    rim = _mk(facecolor="none", edgecolor=edge or _rim(fill), linewidth=lw,
              zorder=zorder + 0.3, joinstyle="round", capstyle="round")
    ax.add_patch(rim)
    return rim


def disc(ax, x, y, d, *, fill=PAPER, edge=None, lw=None, zorder=5, coords="axes", gloss=True,
         shadow=0.0, at=(0.0, 0.0), tr=None, gloss_scale=1.0):
    """A circle of diameter `d` INCHES -- round whatever the axes aspect.

    `at` offsets the circle within the anchor's own inch frame, and `tr` reuses an existing frame.
    Composite glyphs (a plate's colonies) MUST place their parts this way: a returned patch cannot
    simply be re-transformed afterwards, because `_body` builds five artists and only one of them
    comes back.
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    r = d / 2.0
    ax0, ay0 = at
    lw = STROKE["secondary"] if lw is None else lw
    g = ((ax0 - 0.30 * r, ay0 + 0.36 * r, 0.80 * r * gloss_scale, 0.46 * r * gloss_scale, -28)
         if gloss else None)
    return _body(ax, lambda: Circle((ax0, ay0), r), tr, fill=fill, zorder=zorder, lw=lw, edge=edge,
                 shadow=shadow, gloss=g, extent=(ax0 - r, ax0 + r, ay0 - r, ay0 + r))


def droplet(ax, x, y, d, *, fill=DONOR, zorder=5, coords="axes", dna=False, dna_color=None):
    """A liquid droplet -- for a transformation/transfer step.

    `dna=True` draws a short double helix INSIDE the drop, which is how a droplet says it carries
    purified dsDNA rather than a buffer or a cell suspension. The helix is drawn by
    `molecules.helix_path`, the module that owns duplexes -- it is NOT re-derived here, because a
    second helix implementation is exactly the fork this library exists to prevent. The import is
    lazy because `molecules` imports THIS module, so a top-level import would be circular.

    🔴 It is an ICON-register helix: it carries no per-base claim and must never be labelled with
    a sequence or a coordinate (the schematic module rule 11). It is drawn thin and on the drop's 45 deg
    diameter, which is its longest straight run -- a horizontal helix wastes ~half the available
    length and reads as a squiggle rather than as DNA.
    """
    tr = inch_axes(ax, x, y, coords)
    r = d / 2.0
    th = np.linspace(-math.pi * 0.78, math.pi * 0.78, 90)
    pts = [(r * math.sin(t), -r * math.cos(t)) for t in th] + [(0.0, 1.55 * r)]
    body = _body(ax, lambda: Polygon(pts, closed=True), tr, fill=fill, zorder=zorder,
                 lw=STROKE["secondary"], gloss=(-0.28 * r, 0.10 * r, 0.42 * r, 0.66 * r, -18),
                 extent=(-r, r, -r, 1.55 * r))
    if dna:
        import molecules as M
        col = INK if dna_color is None else dna_color
        e = 0.58 * r
        small = d < 0.30
        arts = M.helix_path(ax, [(-e, -e), (e, e)], tr=tr,
                            amp=(0.190 if small else 0.150) * r,
                            per_turn=(0.62 if small else 0.40) * r,
                            color=col, rungs=not small,
                            zorder=zorder + 1.4, lw=max(STROKE_MIN_PT, STROKE["hairline"]))
        for a in (arts if isinstance(arts, (list, tuple)) else [arts]):
            try:
                a.set_clip_path(body)
                a.set_clip_box(None)
                exempt(a)
            except Exception:
                pass
    return body


def arrow(ax, x0, y0, x1, y1, *, color=MUTE, lw=None, head=None, coords="axes", zorder=3,
          dashed=False, conn=None, tr=None):
    """A recessive process arrow (tokens: `process_arrow`).  Solid = canonical, dashed = alternative."""
    from matplotlib.patches import FancyArrowPatch
    lw = STROKE["secondary"] if lw is None else lw
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                        connectionstyle=conn or "arc3,rad=0",
                        mutation_scale=(10 if head is None else head) * (lw / 0.6),
                        lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=zorder,
                        clip_on=False, joinstyle="round",
                        linestyle=(0, (2.6, 1.9)) if dashed else "solid",
                        transform=tr if tr is not None else
                        {"axes": ax.transAxes, "data": ax.transData,
                         "figure": ax.figure.transFigure}[coords])
    ax.add_patch(a)
    return a


def _union_outline(r1, r2, neck, ovality, angle, bumps=()):
    """Closed path around the UNION of a mother circle and a bud circle, with a real neck notch.

    Geometry, given a target neck HALF-width h::

        d = sqrt(r1**2 - h**2) + sqrt(r2**2 - h**2)
    """
    h = neck * r2
    h = min(h, 0.92 * min(r1, r2))
    a, b = math.sqrt(max(r1 * r1 - h * h, 0)), math.sqrt(max(r2 * r2 - h * h, 0))
    d = a + b
    al = math.atan2(h, a)
    be = math.atan2(h, -b)
    def _r_at(t):
        rr = r1
        for bt, bh, bw in bumps:
            dth = (t - bt + math.pi) % (2 * math.pi) - math.pi
            rr += r1 * bh * math.exp(-(dth / bw) ** 2)
        return rr
    pts = [(_r_at(t) * math.cos(t), _r_at(t) * math.sin(t))
           for t in np.linspace(al, 2 * math.pi - al, 320)]
    pts += [(d + r2 * math.cos(t), r2 * math.sin(t))
            for t in np.linspace(2 * math.pi - be, 2 * math.pi + be, 110)]
    cx = d * 0.30
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    out = []
    for px, py in pts:
        qx, qy = px - cx, py * ovality
        out.append((qx * ca - qy * sa, qx * sa + qy * ca))
    return Path(out + [out[0]], closed=True), (d - cx, cx)


def yeast(ax, x, y, d=0.34, *, state="unedited", fill=None, mark="body", bud=True,
          neck=0.30, bud_ratio=0.56, angle=38.0, ovality=0.94, nucleus=True, nucleus_d=0.34,
          vacuole=False,
          genome=None, lw=None, edge=None, coords="axes", zorder=5, shadow=0.030, gloss=True,
          pool=False, at=(0.0, 0.0), tr=None, detail=1, wall=True, scars=0,
          loci=(), scar_seed=0, view="section", edit_at=None, bump_edge=True):
    """A budded *S. cerevisiae* cell.  `d` = MOTHER DIAMETER in inches.

    STATE BY FILL (`the schematic module` rule 5) -- one organism glyph, never a second badge:
      * ``state="unedited"``  the cell's own warm `cell` ground, no payload.
      * ``state="edited"``    filled with the PAYLOAD colour, which the caller passes as `fill`.
        ``mark="body"``    the cell body carries the colour  (reads at any size).
        ``mark="nucleus"`` the NUCLEUS carries it, body stays pale (keeps a dense field legible,
                           and gives the genome and its barcode somewhere to live).

    THREE LEVELS OF DETAIL, and the level is a claim about what the reader is meant to look at::

        detail=0   body + rim only. Pool members and anything under ~0.12 in.
        detail=1   + nucleus (the default; what a schematic needs).
        detail=2   + cell wall, vacuole, mitochondria, granules, nucleolus -- a HERO cell, for a
                   callout or a mechanism zoom.

    🔴 `detail=2` is for ONE cell that the reader is being asked to study. `the schematic module` rule 3
    is explicit that pooled complexity is shown by MULTIPLICITY, not detail -- twenty organelle-
    bearing cells in a flask is noise, and at flask scale the organelles are sub-pixel anyway. The
    level auto-degrades below `_DETAIL_MIN_IN` rather than drawing mush.

    `scars` draws BUD SCARS — the chitin rings left on the wall by each completed division, so the
    count IS the cell's replicative age. They are the most yeast-specific feature there is, they sit
    on the OUTSIDE of the wall, and they therefore belong at `detail=1`, not with the organelles:
    a cutaway is needed to see a vacuole and not to see a scar. (BioRender's own icon ladder agrees
    — it drops the nucleus first and keeps the bud scar last.)

    The two are mutually exclusive by construction: you cannot see through a wall and also see the
    scars on it. Asking for scars in a section, or a nucleus on a surface, silently drops the part
    that does not belong to the view rather than drawing an impossible cell.

    🔴 Passing `loci` SUPPRESSES the nucleus. A genotype view is a different claim from an
    anatomy view: the marks are the content, and a nucleus drawn underneath them reads as one more
    locus. BioRender's own haploid/diploid icons draw no nucleus either.

    `loci` draws GENOTYPE MARKS inside the cell: a sequence of ``(shape, colour, present)`` where
    shape is "box" or "dot". One entry per ALLELE, so a haploid locus is one entry and a diploid
    locus is two — which is how ploidy is read off the drawing rather than asserted in a label. This
    workspace needs exactly that: MATa vs MATalpha, and telling an a/alpha diploid from a haploid.

    `pool=True` marks the glyph as a multiplicity member: lighter stroke, no shadow, and exempt
    from the stroke floor per rule 3.  ``flask`` sets it for you, and places each member with
    ``at=`` in the flask's own inch frame -- there is ONE cell implementation, deliberately.

    🔴 There used to be a second one, `_yeast_at`, for pool members. It drifted: it measured the
    nucleus from the GLYPH centre while this function measures it from the MOTHER's, so every
    barcoded cell in every flask had its nucleus pinched into the bud neck or pushed outside the
    outline entirely (with `clip_on=False` letting it escape). A forked glyph is the exact failure
    this module was written to end; the fix was to delete the fork, not to move the offset.
    
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    ax0, ay0 = at
    if edit_at is not None:
        if genome is None:
            raise ValueError(
                "yeast(edit_at=...) needs genome=<colour>: an edit is a position IN the genome, and "
                "a cell drawn without its genome rail has nowhere to put one. Pass genome= as well, "
                "or drop edit_at and let the caption carry the edit (97 section 2.4).")
        _e = edit_at if isinstance(edit_at, (tuple, list)) else (edit_at, EDIT)
        _e_frac, _e_col = float(_e[0]), _e[1]
        if not 0.0 <= _e_frac <= 1.0:
            raise ValueError(f"yeast(edit_at=) fraction must be in [0, 1]; got {_e_frac}")
    if d < _DETAIL_MIN_IN and detail > 1:
        detail = 1
    _surface = view == "surface"
    if genome is not None:
        vacuole = False
        detail = min(detail, 1)
        if nucleus_d == 0.34:
            nucleus_d = 0.56
    if _surface:
        nucleus = vacuole = False
        genome = None
        loci = ()
        detail = min(detail, 1)
        if ovality == 0.94:
            ovality = 0.70
    else:
        scars = 0
    r1 = d / 2.0
    r2 = bud_ratio * r1 if bud else 0.0
    body_fill = CELL
    nuc_fill = NUCLEOPLASM if genome is not None else shade(CELL, -0.20)
    if state == "edited" and fill:
        if mark == "nucleus" and genome is not None:
            body_fill = shade(fill, -0.28)
        elif mark == "nucleus":
            nuc_fill = fill
        else:
            body_fill = shade(fill, -0.28)
    if lw is None:
        lw = float(np.clip(0.055 * d * 72.0, DETAIL_MIN_PT if pool else STROKE_MIN_PT,
                           STROKE["primary"]))

    ca0, sa0 = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    _scar_pts, _bumps = [], []
    if scars and _surface:
        _rng0 = np.random.RandomState(scar_seed)
        _t = 0
        while len(_scar_pts) < int(scars) and _t < 600:
            _t += 1
            _u, _v = _rng0.uniform(-1, 1), _rng0.uniform(-1, 1)
            _rho = math.hypot(_u, _v)
            if _rho > 0.94 or (_u > 0.52 and abs(_v) < 0.38):
                continue
            if any((_u - a) ** 2 + (_v - b) ** 2 < 0.085 for a, b, _, _e in _scar_pts):
                continue
            _sr = 0.074 * d * (0.78 + 0.48 * _rng0.rand())
            _edge = _rho > 0.78 and bump_edge
            if _edge:
                _k = 0.96 / max(_rho, 1e-6)
                _u, _v = _u * _k, _v * _k
                _rho = math.hypot(_u, _v)
            _scar_pts.append((_u, _v, _sr, _edge))
            if _edge:
                _ang = math.atan2(_v * ovality, _u)
                _sw = _sr / r1
                _bumps.append((_ang, max(_rho + 0.72 * _sw - 1.0, 0.02), max(1.9 * _sw, 0.24)))
            elif _rho > 0.46 and bump_edge:
                _h = 0.062 * (_sr / (0.074 * d)) * min((_rho - 0.46) / 0.40, 1.0)
                _bumps.append((math.atan2(_v * ovality, _u), _h, 0.44))
    if bud:
        path, (_, _mshift) = _union_outline(r1, r2, neck, ovality, angle, bumps=_bumps)
    else:
        _th = np.linspace(0, 2 * math.pi, 320)

        def _r0(t):
            rr = r1
            for bt, bh, bw in _bumps:
                dth = (t - bt + math.pi) % (2 * math.pi) - math.pi
                rr += r1 * bh * math.exp(-(dth / bw) ** 2)
            return rr
        _pts = [(_r0(t) * math.cos(t), _r0(t) * ovality * math.sin(t)) for t in _th]
        _ca, _sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        path = Path([(px * _ca - py * _sa, px * _sa + py * _ca) for px, py in _pts]
                    + [(_pts[0][0] * _ca - _pts[0][1] * _sa,
                        _pts[0][0] * _sa + _pts[0][1] * _ca)], closed=True)
        _mshift = 0.0
    if ax0 or ay0:
        path = Path([(px + ax0, py + ay0) for px, py in path.vertices], closed=True)
    ext = 1.05 * (r1 + (2 * r2 if bud else 0))
    _ex0, _ex1 = ax0 - ext, ax0 + ext
    _ey0, _ey1 = ay0 - ext, ay0 + ext
    _carries_state = body_fill != CELL
    _grad = ((-0.40, 0.56) if _surface else
             ((-0.10, 0.16) if _carries_state else (-0.16, 0.34)))
    _mcx, _mcy = ax0 - _mshift * ca0, ay0 - _mshift * sa0
    if pool:
        _gloss_spec = (_mcx - 0.26 * r1, _mcy + 0.34 * r1, 0.52 * r1, 0.34 * r1, 0.0, 0.42)
    elif _surface:
        _gloss_spec = (_mcx - 0.26 * r1, _mcy + 0.30 * r1 * ovality,
                       1.02 * r1, 0.46 * r1 * ovality, angle - 14.0, 0.72)
    else:
        _gloss_spec = (_mcx - 0.30 * r1, _mcy + 0.40 * r1, 0.86 * r1, 0.46 * r1, -18.0,
                       0.30 if _carries_state else 0.55)
    rim = _body(ax, path, tr, fill=body_fill, zorder=zorder, lw=lw,
                grad=_grad,
                grad_center=(0.31, 0.76) if _surface else (0.35, 0.72),
                grad_falloff=0.92 if _surface else 1.15,
                edge=edge or (shade(body_fill, -0.28) if pool else _rim(body_fill)),
                shadow=0.0 if pool else shadow,
                gloss=(_gloss_spec if gloss else None),
                extent=(_ex0, _ex1, _ey0, _ey1))
    if pool:
        exempt(rim)

    if d < _DETAIL_MIN_IN and detail > 1:
        detail = 1
    _surface = view == "surface"
    if genome is not None:
        vacuole = False
        detail = min(detail, 1)
        if nucleus_d == 0.34:
            nucleus_d = 0.56
    if _surface:
        nucleus = vacuole = False
        genome = None
        loci = ()
        detail = min(detail, 1)
        if ovality == 0.94:
            ovality = 0.70
    else:
        scars = 0
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))

    def _place(u, v):
        """(along the bud axis, across it), measured from the MOTHER's centre -> local inches.

        🔴 `_union_outline` re-centres the glyph on the mother-bud midline, so the glyph's origin is
        `_mshift` inches toward the bud from the mother. Everything placed INSIDE the cell must
        subtract that, or it drifts toward the neck as the bud grows.
        """
        u -= _mshift
        return (ax0 + u * ca - v * sa, ay0 + u * sa + v * ca)

    cyto = None
    if detail >= 2:
        if wall:
            inner = Path([(ax0 + (px - ax0) * 0.915, ay0 + (py - ay0) * 0.915)
                          for px, py in path.vertices], closed=True)
            cyto = PathPatch(inner, transform=tr, facecolor=shade(body_fill, 0.20),
                             edgecolor=shade(body_fill, -0.20),
                             lw=max(DETAIL_MIN_PT, lw * 0.45), zorder=zorder + 0.35,
                             clip_on=False, joinstyle="round")
            ax.add_patch(cyto)
            exempt(cyto)

        def _clip(artist):
            """Every organelle is clipped to the cytoplasm, so nothing can escape the wall."""
            if cyto is not None:
                artist.set_clip_path(cyto)
                artist.set_clip_box(None)
            exempt(artist)
            return artist

        for u, v, ang, ln in ((0.50, 0.42, 40, 1.00), (-0.62, -0.20, -18, 0.88),
                              (0.16, 0.70, 72, 0.80)):
            mx, my = _place(u * r1, v * r1)
            e = Ellipse((mx, my), 0.185 * d * ln, 0.072 * d, angle=angle + ang,
                        facecolor=shade(body_fill, -0.14), edgecolor=shade(body_fill, -0.34),
                        lw=max(DETAIL_MIN_PT, lw * 0.40), transform=tr, zorder=zorder + 0.42,
                        clip_on=True)
            ax.add_patch(_clip(e))
        for u, v, sz in ((0.70, 0.10, 0.8), (-0.30, 0.62, 1.0), (0.46, -0.52, 0.85),
                         (-0.70, 0.22, 0.75), (0.06, 0.34, 0.7)):
            gx2, gy2 = _place(u * r1, v * r1)
            c = Circle((gx2, gy2), 0.021 * d * sz, facecolor=shade(body_fill, -0.26),
                       edgecolor="none", transform=tr, zorder=zorder + 0.44, clip_on=True)
            ax.add_patch(_clip(c))
        if vacuole is False:
            vacuole = True

    if scars and _surface:
        for u, v, sr, _is_edge in _scar_pts:
            rho = math.hypot(u, v)
            f = math.sqrt(max(1.0 - rho * rho, 0.05))
            if _is_edge:
                f = max(f, 0.52)
            sx, sy = _place(u * r1, v * r1 * ovality)
            tang = math.degrees(math.atan2(v * ovality, u)) + 90.0 + angle
            _off = 0.22 * sr
            drop = Ellipse((sx + _off * 0.9, sy - _off * 0.9 * f), 2 * sr * 1.04,
                           2 * sr * f * 1.04, angle=tang, facecolor=shade(body_fill, -0.34),
                           edgecolor="none", alpha=0.38, transform=tr, zorder=zorder + 0.352,
                           clip_on=False)
            ax.add_patch(drop)
            exempt(drop)
            ring = Ellipse((sx, sy), 2 * sr, 2 * sr * f, angle=tang,
                           facecolor=shade(body_fill, 0.22),
                           edgecolor=shade(body_fill, -0.30),
                           lw=max(DETAIL_MIN_PT, lw * 0.42), transform=tr,
                           zorder=zorder + 0.36, clip_on=False)
            ax.add_patch(ring)
            exempt(ring)
            for _dx, _dy, _amt, _al in ((_off, -_off * f, -0.24, 0.85),
                                        (-_off, _off * f, 0.42, 0.90)):
                cr = Ellipse((sx + _dx, sy + _dy), 2 * sr, 2 * sr * f, angle=tang,
                             facecolor=shade(body_fill, _amt), edgecolor="none", alpha=_al,
                             transform=tr, zorder=zorder + 0.365, clip_on=True)
                cr.set_clip_path(ring)
                cr.set_clip_box(None)
                ax.add_patch(cr)
                exempt(cr)
            pit = Ellipse((sx, sy), 2 * sr * 0.52, 2 * sr * 0.52 * f, angle=tang,
                          facecolor=shade(body_fill, -0.16), edgecolor=shade(body_fill, -0.26),
                          lw=max(DETAIL_MIN_PT, lw * 0.32), transform=tr,
                          zorder=zorder + 0.372, clip_on=False)
            ax.add_patch(pit)
            exempt(pit)
            lip = Ellipse((sx - 0.10 * sr, sy + 0.10 * sr * f), 2 * sr * 0.30,
                          2 * sr * 0.30 * f, angle=tang, facecolor=shade(body_fill, 0.55),
                          edgecolor="none", alpha=0.60, transform=tr, zorder=zorder + 0.375,
                          clip_on=False)
            ax.add_patch(lip)
            exempt(lip)

    if vacuole:
        vx, vy = _place(-0.34 * r1, 0.30 * r1)
        vac = Ellipse((vx, vy), 1.00 * r1, 0.88 * r1, angle=angle + 18,
                      facecolor=shade(body_fill, 0.34), edgecolor=shade(body_fill, -0.16),
                      lw=max(DETAIL_MIN_PT, lw * 0.50), transform=tr,
                      zorder=zorder + 0.4, clip_on=False)
        ax.add_patch(vac)
        exempt(vac)

    if nucleus and detail >= 1 and not loci:
        if detail >= 2:
            nx, ny = _place(0.30 * r1, -0.40 * r1)
        else:
            nx, ny = _place(-0.05 * r1 if bud else 0.0, -0.17 * r1 if bud else 0.0)
        nd = (nucleus_d * 0.85 if detail >= 2 else nucleus_d) * d
        exempt(_body(ax, lambda: Circle((nx, ny), nd / 2.0), tr, fill=nuc_fill,
                     zorder=zorder + 0.5, lw=max(DETAIL_MIN_PT, lw * 0.7), edge=_rim(nuc_fill),
                     gloss=(nx - 0.14 * nd, ny + 0.16 * nd, 0.36 * nd, 0.19 * nd, -30)
                     if gloss else None,
                     extent=(nx - nd, nx + nd, ny - nd, ny + nd)))
        if detail >= 2 and nuc_fill != shade(CELL, -0.20):
            no = Circle((nx + 0.16 * nd, ny - 0.13 * nd), 0.17 * nd,
                        facecolor=shade(nuc_fill, -0.26), edgecolor="none", transform=tr,
                        zorder=zorder + 0.7, clip_on=False)
            ax.add_patch(no)
            exempt(no)
        if genome:
            gw = 0.62 * nd
            bars = (0.16, 0.07, 0.12, 0.06, 0.18, 0.07, 0.10)
            k = gw / (sum(bars) + 0.09 * (len(bars) - 1))
            bx = nx - gw / 2.0
            for w in bars:
                rct = Rectangle((bx, ny - 0.055 * nd), w * k, 0.11 * nd, facecolor=genome,
                                edgecolor="none", transform=tr, zorder=zorder + 0.8, clip_on=False)
                ax.add_patch(rct)
                exempt(rct)
                bx += (w + 0.09) * k
            for dy in (-0.075 * nd, 0.075 * nd):
                ln, = ax.plot([nx - gw / 2.0, nx + gw / 2.0], [ny + dy, ny + dy], color=INK,
                              lw=max(DETAIL_MIN_PT, lw * 0.55), transform=tr, zorder=zorder + 0.75,
                              clip_on=False, solid_capstyle="round")
                exempt(ln)
            if edit_at is not None:
                ex = nx - gw / 2.0 + _e_frac * gw
                ln, = ax.plot([ex, ex], [ny - 0.105 * nd, ny + 0.105 * nd], color=_e_col,
                              lw=max(DETAIL_MIN_PT, lw * 1.15), transform=tr,
                              zorder=zorder + 0.85, clip_on=False, solid_capstyle="butt")
                exempt(ln)
    if loci:
        n = len(loci)
        cols = 2 if n > 2 else n
        rows = int(math.ceil(n / cols))
        mw, mh = 0.20 * d, 0.13 * d
        for i, spec in enumerate(loci):
            shape, col, present = (list(spec) + [True])[:3]
            gx, gy = _place((-(cols - 1) / 2.0 + i % cols) * 0.30 * d,
                            ((rows - 1) / 2.0 - i // cols) * 0.26 * d)
            face = col if present else "none"
            edge = shade(col, -0.25) if present else col
            if shape == "box":
                from matplotlib.patches import Rectangle as _R
                a = _R((gx - mw / 2, gy - mh / 2), mw, mh, facecolor=face, edgecolor=edge,
                       lw=max(DETAIL_MIN_PT, lw * 0.6), transform=tr, zorder=zorder + 0.9,
                       clip_on=False)
            else:
                a = Circle((gx, gy), mh * 0.62, facecolor=face, edgecolor=edge,
                           lw=max(DETAIL_MIN_PT, lw * 0.6), transform=tr, zorder=zorder + 0.9,
                           clip_on=False)
            ax.add_patch(a)
            exempt(a)
    return rim


_FLASK_CTRL = [(0.1640, 1.000), (0.1560, 0.983), (0.1500, 0.960), (0.1500, 0.860),
               (0.1508, 0.800), (0.1530, 0.760), (0.1585, 0.722), (0.1690, 0.686),
               (0.1850, 0.648), (0.2065, 0.606),
               (0.2641, 0.506), (0.3217, 0.406), (0.3793, 0.306), (0.4237, 0.229),
               (0.4750, 0.140),
               (0.4920, 0.072), (0.4980, 0.036), (0.4900, 0.013), (0.4680, 0.003),
               (0.4380, 0.000)]


def _catmull2(pts, per_seg=16):
    """Catmull-Rom through an OPEN sequence of 2-D points (the closed-loop `_catmull` above works on
    radii and cannot be reused here)."""
    P = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(len(P) - 3):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        for k in range(per_seg):
            t = k / per_seg
            t2, t3 = t * t, t * t * t
            out.append(tuple(
                0.5 * ((2 * p1[d]) + (-p0[d] + p2[d]) * t
                       + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t2
                       + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t3) for d in (0, 1)))
    out.append(pts[-1])
    return out


_FLASK_R = _catmull2(_FLASK_CTRL)


def _hw_at(y):
    """Half-width of the smooth silhouette at height `y`, both in flask-height units.

    🔴 The packer and the liquid MUST read the silhouette through this, not through remembered
    constants. The straight-sided profile had its half-width hard-coded in three places; the moment
    the outline curved, cells would have hung outside the glass and the meniscus would have been the
    wrong length, and no checker can see either.
    """
    prof = _FLASK_R
    if y >= prof[0][1]:
        return prof[0][0]
    for (w1, y1), (w2, y2) in zip(prof, prof[1:]):
        if y1 >= y >= y2:
            t = 0.0 if y1 == y2 else (y1 - y) / (y1 - y2)
            return w1 + (w2 - w1) * t
    return prof[-1][0]


def _flask_path(w, h):
    right = [(px * w, py * h - h / 2.0) for px, py in _FLASK_R]
    left = [(-px * w, py * h - h / 2.0) for px, py in reversed(_FLASK_R)]
    pts = right + left
    return Path(pts + [pts[0]], closed=True)


def _liquid_path(w, h, level):
    body = [(px * w, py * h - h / 2.0) for px, py in _FLASK_R if py <= level]
    hw = _hw_at(level)
    pts = ([(hw * w, level * h - h / 2.0)] + body
           + [(-px, py) for px, py in reversed(body)]
           + [(-hw * w, level * h - h / 2.0)])
    return Path(pts + [pts[0]], closed=True)


_SEP = 1.02


def _pack(n, w, h, level, rad, seed, tries=6000):
    """Deterministic dart-throwing so the cells settle as a DENSE population, not scattered dots."""
    rng = np.random.RandomState(seed)
    pts, k = [], 0
    while len(pts) < n and k < tries:
        k += 1
        t = rng.rand() ** 1.25
        m = rad / h
        py = m * 0.92 + t * max(level - 1.85 * m, 0.02)
        hw = _hw_at(py)
        px = (rng.rand() * 2 - 1) * max(hw - rad / w * 0.92, 0.02)
        X, Y = px * w, py * h
        if all((X - a) ** 2 + (Y - b) ** 2 > (_SEP * rad) ** 2 for a, b, _ in pts):
            pts.append((X, Y, 0.86 + 0.30 * rng.rand()))
    return pts


def flask(ax, x, y, h=0.80, *, cells=22, medium=None, state="unedited", fill=None, mark="body",
          pool_gloss=True, graduations=0,
          cell_d=None, seed=3, level=0.42, coords="axes", zorder=4, lw=None, label=None,
          shadow=0.034):
    """An Erlenmeyer culture holding budded cells at density.  `h` = flask HEIGHT in inches,
    CENTRED on (x, y) like every other glyph here.

    The flask is STRUCTURE: `ink` glass, a pale neutral or `medium` tint for the broth.  Hue is
    spent only on the cells, and only when they carry a payload.
    """
    tr = inch_axes(ax, x, y, coords)
    w = 0.86 * h
    lw = STROKE["primary"] if lw is None else lw
    cd = (0.150 * h) if cell_d is None else cell_d
    fp = _flask_path(w, h)

    if shadow and EFFECTS:
        sh = PathPatch(fp, transform=Affine2D().translate(0.30 * shadow, -shadow) + tr,
                       facecolor=INK, edgecolor="none", alpha=0.13, zorder=zorder - 0.3,
                       clip_on=False)
        ax.add_patch(sh)
        exempt(sh)

    glass = PathPatch(fp, transform=tr, facecolor="#F1F5F8", edgecolor="none",
                      zorder=zorder, clip_on=False)
    ax.add_patch(glass)
    exempt(glass)

    med = shade(AGAR, 0.58) if medium is None else medium
    lp = _liquid_path(w, h, level)
    liq = PathPatch(lp, transform=tr, facecolor=med, edgecolor="none", zorder=zorder + 0.2,
                    clip_on=False)
    ax.add_patch(liq)
    exempt(liq)
    if STYLE == "volume":
        _gradient(ax, liq, shade(med, -0.14), shade(med, 0.30),
                  (-0.52 * w, 0.52 * w, -h / 2.0, level * h - h / 2.0), tr, zorder + 0.25,
                  radial=False)
    mhw = _hw_at(level) * w
    ml, = ax.plot([-mhw, mhw], [level * h - h / 2.0, level * h - h / 2.0],
                  color=shade(med, -0.30), lw=max(STROKE_MIN_PT, lw * 0.55), transform=tr,
                  zorder=zorder + 0.3, solid_capstyle="butt")
    ml.set_clip_path(liq)
    ml.set_clip_box(None)
    exempt(ml)

    mem = sorted(_pack(cells, w, h, level, cd * 0.82, seed), key=lambda t: -t[1])
    zc0 = zorder + 1.0
    for i, (cx, cy, sc) in enumerate(mem):
        yeast(ax, x, y, cd * sc, tr=tr, at=(cx, cy - h / 2.0), coords=coords,
              state=state, fill=fill, mark=mark, pool=True, gloss=pool_gloss, shadow=0.0,
              nucleus=(state == "edited" and bool(fill) and mark == "nucleus"),
              angle=30.0 + ((int(abs(cx) * 9973) + int(abs(cy) * 7919)) % 8) * 42.0,
              zorder=zc0 + i)
    zglass = zc0 + len(mem) + 1.0

    if graduations:
        for gi in range(int(graduations)):
            gy = (0.16 + 0.13 * gi) * h - h / 2.0
            if gy > level * h - h / 2.0 - 0.02 * h:
                break
            ghw = _hw_at((gy + h / 2.0) / h) * w
            gl_, = ax.plot([ghw * 0.30, ghw * 0.62], [gy, gy], color=shade(INK, 0.55),
                           lw=max(STROKE_MIN_PT, lw * 0.42), transform=tr, zorder=zglass - 0.2,
                           solid_capstyle="round")
            gl_.set_clip_path(glass)
            gl_.set_clip_box(None)
            exempt(gl_)

    gl = PathPatch(fp, transform=tr, facecolor="none", edgecolor=INK, lw=lw, zorder=zglass,
                   joinstyle="round", capstyle="round", clip_on=False)
    ax.add_patch(gl)
    st, = ax.plot([-0.60 * _hw_at(0.86) * w, -0.72 * _hw_at(0.34) * w],
                  [0.86 * h - h / 2.0, 0.34 * h - h / 2.0],
                  color="#FFFFFF", alpha=0.55,
                  lw=lw * 1.7, transform=tr, zorder=zglass - 0.1, solid_capstyle="round")
    st.set_clip_path(glass)
    st.set_clip_box(None)
    exempt(st)
    if label:
        text_in(ax, x, y, 0.0, -h / 2.0 - 0.085, label, coords=coords, va="top",
                size=TYPE["annotation"])
    return gl


def sun(ax, x, y, d=0.20, *, coords="axes", zorder=6, lw=None, rays=8, color=None):
    """LIGHT.  A disc with eight rays; shape carries the meaning, the fill only supports it."""
    tr = inch_axes(ax, x, y, coords)
    r = d / 2.0
    lw = STROKE["secondary"] if lw is None else lw
    face = SUN_FILL if color is None else color
    for k in range(rays):
        t = math.radians(22.5 + k * (360.0 / rays))
        ln, = ax.plot([1.34 * r * math.cos(t), 1.96 * r * math.cos(t)],
                      [1.34 * r * math.sin(t), 1.96 * r * math.sin(t)],
                      color=SUN_RIM, lw=lw * 1.15, transform=tr, zorder=zorder,
                      clip_on=False, solid_capstyle="round")
    _body(ax, lambda: Circle((0, 0), r), tr, fill=face, zorder=zorder + 0.2, lw=lw, edge=SUN_RIM,
          gloss=(-0.28 * r, 0.34 * r, 0.74 * r, 0.40 * r, -30), extent=(-r, r, -r, r))
    return tr


def moon(ax, x, y, d=0.20, *, coords="axes", zorder=6, lw=None, color=None, bite=0.40,
         inner=0.82, tilt=28.0):
    """DARK.  A closed crescent: outer arc of radius r, inner arc of radius `inner`*r offset by
    `bite`*r.  Both arcs are real geometry, so the shape survives greyscale and any background."""
    tr = inch_axes(ax, x, y, coords)
    r = d / 2.0
    lw = STROKE["secondary"] if lw is None else lw
    face = MOON_FILL if color is None else color
    dd, ri = bite * r, inner * r
    xi = (dd * dd + r * r - ri * ri) / (2 * dd)
    yi = math.sqrt(max(r * r - xi * xi, 1e-9))
    a = math.atan2(yi, xi)
    b = math.atan2(yi, xi - dd)
    pts = [(r * math.cos(t), r * math.sin(t))
           for t in np.linspace(a, 2 * math.pi - a, 140)]
    pts += [(dd + ri * math.cos(t), ri * math.sin(t))
            for t in np.linspace(2 * math.pi - b, b, 140)]
    ca, sa = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    pts = [(px * ca - py * sa, px * sa + py * ca) for px, py in pts]
    path = Path(pts + [pts[0]], closed=True)
    _body(ax, path, tr, fill=face, zorder=zorder, lw=lw,
          gloss=(-0.34 * r, 0.30 * r, 0.50 * r, 0.30 * r, 40), extent=(-r, r, -r, r))
    return tr


def condition(ax, x, y, kind, *, d=0.20, label=True, coords="axes", zorder=6, size=None,
              gap=0.055, color=None, side="right"):
    """A sun or a moon plus its word.  Light and dark are NEVER carried by a background wash alone;
    a wash is a background role and a reader printing in greyscale loses it.

    `side` puts the word "right" (default), "left" or "below" the glyph -- a badge at the right
    edge of a panel needs its word on the inside or the word runs off the canvas.
    """
    (sun if kind == "light" else moon)(ax, x, y, d, coords=coords, zorder=zorder)
    if not label:
        return
    word = "LIGHT" if kind == "light" else "DARK"
    reach = d / 2.0 * (1.96 if kind == "light" else 1.02)
    dx, dy, ha, va = {"right": (reach + gap, 0.0, "left", "center"),
                      "left": (-reach - gap, 0.0, "right", "center"),
                      "below": (0.0, -reach - gap, "center", "top")}[side]
    text_in(ax, x, y, dx, dy, word, coords=coords, ha=ha, va=va, fontweight="bold",
            size=TYPE["annotation"] if size is None else size,
            color=INK if color is None else color)


def colony(ax, x, y, d=0.05, *, fill=None, coords="axes", zorder=7, picked=False, lw=None,
           at=(0.0, 0.0), tr=None, organism="yeast", maturity=1.0):
    """One colony.  `picked=True` rings it in `edit` -- the sampled clone, not a second colour.

    The two organisms therefore differ in THREE channels, none of them hue, which is why one pair of
    tokens serves both:

    | | yeast | *E. coli* |
    |---|---|---|
    | opacity | opaque | translucent, and **less so with `maturity`** |
    | relief | domed, and more so with `maturity` | flat, and only slightly less so |
    | specularity | **matte** — no highlight at all | a sheen that **broadens and dulls** with `maturity` |

    `maturity` in 0..1 is a GROWTH axis for both organisms, not a yeast-only knob. For yeast it
    grows the dome and its contact shadow; for *E. coli* it raises opacity and turns a young wet
    pinpoint specular into a mature broad **dull shine** — which is the direction that is easy to
    get backwards, because "shinier with age" is the intuitive and wrong answer.

    Dropping the specular highlight is the counter-intuitive half: every other glyph in this module
    gets one, and a yeast colony must not, because *dull* is a stated property of the object.
    `maturity` in 0..1 grows the dome and its contact shadow toward the saturated form.
    
    """
    fill = COLONY if fill is None else fill
    lw = max(DETAIL_MIN_PT, 0.05 * d * 72.0) if lw is None else lw
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    ax0, ay0 = at
    r = d / 2.0
    yeast_like = organism != "ecoli"
    if yeast_like:
        dome = 0.42 + 0.40 * float(np.clip(maturity, 0.0, 1.0))
        p = _body(ax, lambda: Circle((ax0, ay0), r), tr, fill=fill, zorder=zorder, lw=lw,
                  edge=_rim(fill), shadow=(0.20 + 0.14 * maturity) * d,
                  gloss=None,
                  grad=(-0.34, dome), grad_center=(0.34, 0.74), grad_falloff=0.85,
                  extent=(ax0 - r, ax0 + r, ay0 - r, ay0 + r))
    else:
        m = float(np.clip(maturity, 0.0, 1.0))
        base = fill if fill is not COLONY else _mix(COLONY, AGAR, 0.56 - 0.34 * m)
        p = _body(ax, lambda: Circle((ax0, ay0), r), tr, fill=base, zorder=zorder, lw=lw * 0.8,
                  edge=shade(base, -0.20 - 0.10 * m), shadow=(0.030 + 0.030 * m) * d,
                  alpha=0.46 + 0.42 * m,
                  gloss=(ax0 - (0.34 - 0.06 * m) * r, ay0 + (0.38 - 0.06 * m) * r,
                         (0.36 + 0.30 * m) * r, (0.24 + 0.22 * m) * r, -28,
                         0.98 - 0.32 * m),
                  grad=(-0.03 - 0.05 * m, 0.05 + 0.09 * m),
                  extent=(ax0 - r, ax0 + r, ay0 - r, ay0 + r))
    exempt(p)
    if picked:
        rg = Circle(at, 0.86 * d, facecolor="none", edgecolor=EDIT,
                    lw=max(STROKE_MIN_PT, STROKE["secondary"]), transform=tr,
                    zorder=zorder + 0.5, clip_on=False)
        ax.add_patch(rg)
    return p


def plate96(ax, x, y, w=1.05, *, filled=None, fill=None, coords="axes", zorder=4, lw=None,
            label=None, rows=8, cols=12, well_fills=None):
    """A 12 x 8 microplate.  `filled` = how many wells carry the payload, filled row-major.

    This is the glyph for "96 clones sampled" -- a count a reader can literally verify by eye,
    which a scatter of dots can never be.

    `well_fills` (added 2026-09-18) takes a full rows*cols list of colours in ROW-MAJOR order and
    overrides `filled`/`fill` entirely. It exists for a plate whose wells carry MORE THAN ONE
    payload identity -- e.g. a PCR plate split by amplicon direction, where the top half and the
    bottom half are different conditions. Drawing that by overlaying a second set of circles on a
    plate glyph is what forks a vocabulary; this keeps one plate, one geometry, one shadow.
    """
    tr = inch_axes(ax, x, y, coords)
    h = w * (rows + 1.3) / (cols + 1.3)
    lw = STROKE["secondary"] if lw is None else lw
    from matplotlib.patches import FancyBboxPatch
    pad = 0.055 * w
    sh = FancyBboxPatch((-w / 2 + 0.014, -h / 2 - 0.026), w, h,
                        boxstyle="round,pad=0,rounding_size=%.4f" % (0.045 * w),
                        facecolor=INK, edgecolor="none", alpha=0.13, transform=tr,
                        zorder=zorder - 0.3, clip_on=False)
    ax.add_patch(sh)
    exempt(sh)
    body = FancyBboxPatch((-w / 2, -h / 2), w, h,
                          boxstyle="round,pad=0,rounding_size=%.4f" % (0.045 * w),
                          facecolor=shade(PAPER, -0.03), edgecolor=INK, lw=lw,
                          transform=tr, zorder=zorder, clip_on=False)
    ax.add_patch(body)
    dx = (w - 2 * pad) / cols
    dy = (h - 2 * pad) / rows
    wd = 0.74 * min(dx, dy)
    n = 0
    for ri in range(rows):
        for ci in range(cols):
            cx = -w / 2 + pad + dx * (ci + 0.5)
            cy = h / 2 - pad - dy * (ri + 0.5)
            if well_fills is not None:
                f = well_fills[n]
            else:
                on = filled is not None and n < filled
                f = (fill or BARCODE) if on else shade(MUTE, 0.45)
            wl = Circle((cx, cy), wd / 2, facecolor=f, edgecolor=shade(f, -0.35),
                        lw=max(DETAIL_MIN_PT, lw * 0.45), transform=tr, zorder=zorder + 0.5, clip_on=False)
            ax.add_patch(wl)
            exempt(wl)
            n += 1
    if label:
        text_in(ax, x, y, 0.0, -h / 2 - 0.085, label, coords=coords, va="top",
                size=TYPE["annotation"])
    return body


def _ann_sector(r_in, r_out, a0, a1, n=64):
    """A closed annular sector — the honest primitive for anything drawn ON a circular backbone.

    🔴 A feature on a plasmid is an ARC OF AN ANNULUS, not a fat line. Drawn as a thick `plot` its
    ends are square in SCREEN space rather than radial, and a barcode rung of constant linewidth
    subtends a shrinking angle as the radius grows — so the gaps taper shut at the inner edge and
    fan open at the outer, and the whole barcode reads as a comb. Drawing the rungs thinner
    reproduces the same defect at a smaller scale; the fix is fixed-ANGLE geometry.
    """
    th = np.radians(np.linspace(a0, a1, n))
    outer = [(r_out * math.cos(t), r_out * math.sin(t)) for t in th]
    inner = [(r_in * math.cos(t), r_in * math.sin(t)) for t in th[::-1]]
    return Path(outer + inner + [outer[0]], closed=True)


def plasmid(ax, x, y, d=0.60, *, arcs=(), bars=None, coords="axes", zorder=5, lw=None,
            width=None, label=None, backbone=None, bar_color=None, leaders=(), hatch_arcs=(),
            marks=(), leader_len=0.16, leader_size=None, cuts=(), sites=(), cut_gap=11.0,
            cut_size=None):
    """A plasmid: an `ink` backbone circle carrying coloured arcs for the features it holds.

    `the schematic module` rule "structure is never coloured" -- the circle is ink; hue is spent only on
    the molecular identities riding on it.  ``arcs`` = ``[(deg0, deg1, colour), ...]``.
    ``bars=(deg0, deg1)`` draws the barcode as real alternating rungs on the backbone (the project
    identity motif, section 4), not as a coloured arc.

    The `ASPECT` argument every earlier copy of this function needed is gone: the inch transform
    makes the backbone round in any axes.
    """
    tr = inch_axes(ax, x, y, coords)
    r = d / 2.0
    lw = STROKE["primary"] if lw is None else lw
    aw = (0.135 * d) if width is None else width

    cuts = tuple(cuts)
    _cut_low = 0.0
    if not cuts:
        bb = Circle((0, 0), r, facecolor="none", edgecolor=backbone or INK, lw=lw, transform=tr,
                    zorder=zorder, clip_on=False)
        ax.add_patch(bb)
    else:
        cs = sorted(float(c[0] if isinstance(c, (tuple, list)) else c) % 360.0 for c in cuts)
        if len(cs) > 1 and min(cs[i + 1] - cs[i] for i in range(len(cs) - 1)) < cut_gap * 1.5:
            raise ValueError(
                f"plasmid(cuts={cs}): two cuts are closer than {cut_gap * 1.5:.0f} deg, so their "
                "gaps merge and the fragment between them disappears. Move them apart or shrink "
                "cut_gap -- a released fragment the reader cannot see is worse than no cut.")
        bb = None
        for i, c in enumerate(cs):
            a0 = c + cut_gap / 2.0
            a1 = (cs[i + 1] if i + 1 < len(cs) else cs[0] + 360.0) - cut_gap / 2.0
            th = np.radians(np.linspace(a0, a1, max(12, int(abs(a1 - a0) * 1.6))))
            ln, = ax.plot(r * np.cos(th), r * np.sin(th), color=backbone or INK, lw=lw,
                          transform=tr, zorder=zorder, clip_on=False, solid_capstyle="butt")
            if bb is None:
                bb = ln
        for c in cuts:
            if not isinstance(c, (tuple, list)) or len(c) < 2 or not c[1]:
                continue
            v = math.radians(float(c[0]))
            lx, ly = (r + aw * 0.9 + 0.055) * math.cos(v), (r + aw * 0.9 + 0.055) * math.sin(v)
            right = math.cos(v) >= -0.15
            vert = abs(math.cos(v)) < 0.35
            ax.text(lx, ly, c[1], transform=tr, color=INK,
                    ha="center" if vert else ("left" if right else "right"),
                    va="top" if (vert and math.sin(v) < 0) else
                       ("bottom" if vert else "center"),
                    fontsize=cut_size or TYPE["annotation"], style="italic",
                    clip_on=False, zorder=zorder + 0.7)
            if vert and math.sin(v) < 0:
                _cut_low = max(_cut_low, 0.055 + cap_mm(cut_size or TYPE["annotation"]) / 25.4
                               + 0.030)
    for a0, a1, col in arcs:
        pp = PathPatch(_ann_sector(r - aw / 2, r + aw / 2, a0, a1), transform=tr, facecolor=col,
                       edgecolor=_rim(col), lw=max(STROKE_MIN_PT, STROKE["hairline"]),
                       zorder=zorder + 0.3, clip_on=False, joinstyle="round")
        ax.add_patch(pp)
    for s0, s1, n_sites, col_s in sites:
        n_sites = int(n_sites)
        pitch = (s1 - s0) / n_sites
        if abs(pitch) * math.radians(1.0) * r < 0.012:
            raise ValueError(
                f"plasmid(sites=): {n_sites} sites across {abs(s1 - s0):.0f} deg of a {d:.2f} in "
                "backbone gives boxes under 0.012 in wide -- they read as a dashed line, not as a "
                "countable array. Widen the span, drop the count, or draw it on a Track instead.")
        for k in range(n_sites):
            g0 = s0 + pitch * (k + 0.30)
            g1 = s0 + pitch * (k + 0.90)
            pp = PathPatch(_ann_sector(r - aw / 2, r + aw / 2, g0, g1), transform=tr,
                           facecolor=col_s, edgecolor=_rim(col_s),
                           lw=max(STROKE_MIN_PT, STROKE["hairline"]), zorder=zorder + 0.32,
                           clip_on=False, joinstyle="round")
            ax.add_patch(pp)
    if bars is not None:
        b0, b1 = bars
        col = bar_color or BARCODE
        base = PathPatch(_ann_sector(r - aw / 2, r + aw / 2, b0, b1), transform=tr, facecolor=col,
                         edgecolor=_rim(col), lw=max(STROKE_MIN_PT, STROKE["hairline"]),
                         zorder=zorder + 0.4, clip_on=False, joinstyle="round")
        ax.add_patch(base)
        span = abs(b1 - b0)
        n = max(5, int(span / 5.2))
        step = span / n
        for i in range(n):
            if i % 2:
                continue
            g0 = b0 + step * (i + 0.62)
            g1 = g0 + step * (0.40 if i % 4 else 0.62)
            gp = PathPatch(_ann_sector(r - aw / 2 + aw * 0.06, r + aw / 2 - aw * 0.06, g0, g1),
                           transform=tr, facecolor=PAPER, edgecolor="none",
                           zorder=zorder + 0.45, clip_on=False)
            ax.add_patch(gp)
            exempt(gp)
    for m in marks:
        deg, col = m[0], m[1]
        span = m[2] if len(m) > 2 else 3.2
        th = np.radians(np.linspace(deg - span / 2, deg + span / 2, 12))
        for f in np.linspace(-1.32, 1.32, 9):
            rr = r + f * aw * 0.85
            ln, = ax.plot(rr * np.cos(th), rr * np.sin(th), color=col,
                          lw=max(STROKE_MIN_PT, aw * 30.0), transform=tr,
                          zorder=zorder + 0.5, clip_on=False, solid_capstyle="butt")
            exempt(ln)

    for deg, text, col in leaders:
        v = math.radians(deg)
        x0, y0 = (r + aw * 0.9) * math.cos(v), (r + aw * 0.9) * math.sin(v)
        x1, y1 = (r + aw * 0.9 + leader_len) * math.cos(v), (r + aw * 0.9 + leader_len) * math.sin(v)
        ln, = ax.plot([x0, x1], [y0, y1], color=col, lw=max(STROKE_MIN_PT, STROKE["hairline"]),
                      transform=tr, zorder=zorder + 0.6, clip_on=False, solid_capstyle="round")
        right = math.cos(v) >= -0.15
        ax.text(x1 + (0.035 if right else -0.035), y1, text, transform=tr, color=INK,
                fontsize=leader_size or TYPE["annotation"], ha="left" if right else "right",
                va="center", fontweight="bold", clip_on=False, zorder=zorder + 0.7,
                linespacing=1.25)
    for a0, a1, col in hatch_arcs:
        pp = PathPatch(_ann_sector(r - aw / 2, r + aw / 2, a0, a1), transform=tr,
                       facecolor=shade(col, 0.80), edgecolor=col,
                       lw=max(STROKE_MIN_PT, STROKE["hairline"]), zorder=zorder + 0.35,
                       clip_on=False, joinstyle="round", linestyle=(0, (2.2, 1.6)))
        ax.add_patch(pp)
    if label:
        text_in(ax, x, y, 0.0, -r - 0.085 - _cut_low, label, coords=coords, va="top",
                size=TYPE["annotation"])
    return bb


def spot_dilution(ax, x, y, w=1.60, *, rows=4, steps=5, growth=None, coords="axes", zorder=4,
                  lw=None, label=None, row_labels=None, step_labels=False, agar=None,
                  spot_d=None, seed=11, organism="yeast"):
    """A rectangular agar plate carrying rows of serially-diluted spots -- the standard yeast
    growth assay, and the readable way to show that one strain grows and another does not.

    `growth` is one value per row in 0..1: 1.0 grows to the last dilution, 0.4 dies partway, 0.0
    is a dead row.  It is the ROW that carries the phenotype, so a reader compares rows, which is
    exactly what the assay is for.  Each spot fades and shrinks with dilution, and the last
    surviving spot in a row breaks into individual colonies rather than staying a disc.
    """
    tr = inch_axes(ax, x, y, coords)
    lw = STROKE["primary"] if lw is None else lw
    ag = AGAR if agar is None else agar
    h = w * (rows + 1.35) / (steps + 1.6)
    from matplotlib.patches import FancyBboxPatch
    rr = 0.035 * w
    sh = FancyBboxPatch((-w / 2 + 0.30 * 0.014, -h / 2 - 0.014), w, h,
                        boxstyle="round,pad=0,rounding_size=%.4f" % rr, facecolor=INK,
                        edgecolor="none", alpha=0.13, transform=tr, zorder=zorder - 0.3,
                        clip_on=False)
    ax.add_patch(sh)
    exempt(sh)
    body = FancyBboxPatch((-w / 2, -h / 2), w, h,
                          boxstyle="round,pad=0,rounding_size=%.4f" % rr, facecolor=ag,
                          edgecolor=INK, lw=lw, transform=tr, zorder=zorder, clip_on=False)
    ax.add_patch(body)

    padx, pady = 0.085 * w, 0.10 * h
    dx = (w - 2 * padx) / steps
    dy = (h - 2 * pady) / rows
    d0 = 0.72 * min(dx, dy) if spot_d is None else spot_d
    rng = np.random.RandomState(seed)
    g = [1.0] * rows if growth is None else list(growth)
    for ri in range(rows):
        cy = h / 2 - pady - dy * (ri + 0.5)
        reach = g[ri] * steps
        for si in range(steps):
            cx = -w / 2 + padx + dx * (si + 0.5)
            live = reach - si
            if live <= 0:
                continue
            sd = d0 * (0.87 ** si) * min(1.0, 0.42 + 0.58 * live)
            if live < 1.0:
                for _ in range(max(3, int(9 * live))):
                    a, rq = rng.rand() * 2 * math.pi, math.sqrt(rng.rand()) * sd * 0.46
                    colony(ax, x, y, sd * 0.30, tr=tr, zorder=zorder + 1, organism=organism,
                           at=(cx + rq * math.cos(a), cy + rq * math.sin(a)), maturity=0.5)
            else:
                colony(ax, x, y, sd, tr=tr, at=(cx, cy), organism=organism,
                       lw=max(STROKE_MIN_PT, lw * 0.45), zorder=zorder + 1 + si * 0.01,
                       maturity=1.0 - 0.5 * si / max(steps - 1, 1))
        if row_labels:
            text_in(ax, x, y, -w / 2 - 0.07, cy, row_labels[ri], coords=coords, ha="right",
                    size=TYPE["annotation"])
    if step_labels:
        for si in range(steps):
            cx = -w / 2 + padx + dx * (si + 0.5)
            text_in(ax, x, y, cx, h / 2 + 0.055, "10$^{-%d}$" % si, coords=coords, va="bottom",
                    size=TYPE["annotation"])
    if label:
        text_in(ax, x, y, 0.0, -h / 2 - 0.085, label, coords=coords, va="top",
                size=TYPE["annotation"])
    return body


def tetrad(ax, x, y, d=0.42, *, spores=(), coords="axes", zorder=5, lw=None, label=None,
           ascus=True, fill=None, at=(0.0, 0.0), tr=None):
    """Four haploid spores in an ascus, arranged 2 x 2.

    `spores` is four `loci` lists — one per spore — so **segregation is drawn, not asserted**. That
    is the whole reason this glyph exists: a tetrad's content IS which alleles went where, and a
    reader should be able to check 2:2 by counting rather than by trusting a caption.

    🔴 Non-Mendelian elements do NOT segregate 2:2. The 2-micron plasmid, mitochondria and any
    other cytoplasmic element are inherited by ALL FOUR spores, so a caller drawing one must give
    it to every spore. The glyph cannot enforce that — it draws what it is handed — but drawing a
    cytoplasmic marker 2:2 is a claim about genetics that would be wrong.
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    ax0, ay0 = at
    lw = STROKE["secondary"] if lw is None else lw
    face = CELL if fill is None else fill
    sd = d * 0.52
    off = sd * 0.50

    if ascus:
        env = Ellipse((ax0, ay0), d * 1.18, d * 1.18, facecolor=shade(face, 0.52),
                      edgecolor=shade(face, -0.42), lw=lw * 1.25, transform=tr, zorder=zorder,
                      clip_on=False)
        ax.add_patch(env)

    spec = list(spores) + [()] * (4 - len(spores))
    for i, (sx, sy) in enumerate(((-off, off), (off, off), (-off, -off), (off, -off))):
        yeast(ax, x, y, sd, tr=tr, at=(ax0 + sx, ay0 + sy), coords=coords, bud=False,
              nucleus=False, detail=0, gloss=True, shadow=0.0,
              zorder=zorder + 1 + i, loci=spec[i], lw=max(DETAIL_MIN_PT, lw * 0.8))
    if label:
        text_in(ax, x, y, ax0, ay0 - d * 0.66, label, coords=coords, va="top",
                size=TYPE["annotation"])
    return env if ascus else None


def tube(ax, x, y, h=0.90, *, turbidity=1.0, medium=None, level=0.58, coords="axes", zorder=4,
         lw=None, label=None, shadow=0.030, at=(0.0, 0.0), tr=None):
    """A round-bottomed culture tube.  `h` = tube HEIGHT in inches, centred on (x, y).

    🔴 `turbidity` in 0..1 darkens the broth, and that is the point: **a dilution series is drawn as
    a VALUE ramp, not as a set of labels.** The reader sees the concentration fall. It costs no
    accent, it survives greyscale by construction, and it is the one place in this vocabulary where
    a continuous quantity is encoded in a fill.

    A tube is not a small flask. It is the vessel for a dilution step, where a flask is the vessel
    for a culture — so the two glyphs carry different steps and should not be substituted.
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    ax0, ay0 = at
    w = 0.155 * h
    lw = STROKE["secondary"] if lw is None else lw
    base = shade(AGAR, 0.55) if medium is None else medium
    broth = shade(base, -0.55 * float(np.clip(turbidity, 0.0, 1.0)))

    def _outline():
        r = w / 2.0
        pts = [(ax0 - r, ay0 + h / 2)]
        pts += [(ax0 - r, ay0 - h / 2 + r)]
        pts += [(ax0 + r * math.cos(t), ay0 - h / 2 + r + r * math.sin(t))
                for t in np.linspace(math.pi, 2 * math.pi, 40)]
        pts += [(ax0 + r, ay0 + h / 2)]
        return Path(pts + [pts[0]], closed=True)

    if shadow and EFFECTS:
        sh = PathPatch(_outline(), transform=Affine2D().translate(0.30 * shadow, -shadow) + tr,
                       facecolor=INK, edgecolor="none", alpha=0.12, zorder=zorder - 0.3,
                       clip_on=False)
        ax.add_patch(sh)
        exempt(sh)
    glass = PathPatch(_outline(), transform=tr, facecolor="#F1F5F8", edgecolor="none",
                      zorder=zorder, clip_on=False)
    ax.add_patch(glass)
    exempt(glass)

    r = w / 2.0
    top = ay0 - h / 2 + level * h
    lp = [(ax0 - r, top), (ax0 - r, ay0 - h / 2 + r)]
    lp += [(ax0 + r * math.cos(t), ay0 - h / 2 + r + r * math.sin(t))
           for t in np.linspace(math.pi, 2 * math.pi, 40)]
    lp += [(ax0 + r, top)]
    liq = PathPatch(Path(lp + [lp[0]], closed=True), transform=tr, facecolor=broth,
                    edgecolor="none", zorder=zorder + 0.2, clip_on=False)
    ax.add_patch(liq)
    exempt(liq)
    if STYLE == "volume":
        _gradient(ax, liq, shade(broth, -0.12), shade(broth, 0.22),
                  (ax0 - r, ax0 + r, ay0 - h / 2, top), tr, zorder + 0.25, radial=False)
    ml, = ax.plot([ax0 - r * 0.99, ax0 + r * 0.99], [top, top], color=shade(broth, -0.28),
                  lw=max(STROKE_MIN_PT, lw * 0.55), transform=tr, zorder=zorder + 0.3)
    ml.set_clip_path(liq)
    ml.set_clip_box(None)
    exempt(ml)
    st, = ax.plot([ax0 - r * 0.45, ax0 - r * 0.45], [ay0 + h * 0.40, ay0 - h * 0.18],
                  color="#FFFFFF", alpha=0.55, lw=lw * 1.5, transform=tr, zorder=zorder + 0.9,
                  solid_capstyle="round")
    st.set_clip_path(glass)
    st.set_clip_box(None)
    exempt(st)
    gl = PathPatch(_outline(), transform=tr, facecolor="none", edgecolor=INK, lw=lw,
                   zorder=zorder + 1.0, joinstyle="round", clip_on=False)
    ax.add_patch(gl)
    rim, = ax.plot([ax0 - r, ax0 + r], [ay0 + h / 2, ay0 + h / 2], color=INK, lw=lw * 1.3,
                   transform=tr, zorder=zorder + 1.1, solid_capstyle="round")
    if label:
        text_in(ax, x, y, ax0, ay0 - h / 2 - 0.075, label, coords=coords, va="top",
                size=TYPE["annotation"])
    return gl


def _tick_marks(ax, tr, x0, ys, half_w, *, color, lw, zorder, clip):
    """Graduation ticks on a vessel wall. Cheap, and they say 'this is a graduated tube'."""
    for yy in ys:
        t, = ax.plot([x0 - half_w, x0 - half_w * 0.45], [yy, yy], color=color, lw=lw,
                     transform=tr, zorder=zorder, solid_capstyle="butt")
        t.set_clip_path(clip)
        t.set_clip_box(None)
        exempt(t)


def conical(ax, x, y, h=0.95, *, volume=50, fill=None, level=0.55, turbidity=0.0,
            coords="axes", zorder=4, lw=None, label=None, shadow=0.030, at=(0.0, 0.0), tr=None,
            cap=True, ticks=True, patch=True):
    """A CONICAL centrifuge tube -- a Falcon.  `h` = OVERALL height in inches (cap included),
    centred on (x, y).

    🔴 PROPORTIONS ARE FROM THE PRODUCT WE ORDER, not from taste: Corning/Falcon 352098, whose
    photograph shows a tube ~114 mm long x ~30 mm across (3.8:1), a cap about an eighth of the
    length and slightly WIDER than the body, a cone that is only about a fifth of the length, and
    a small ROUNDED tip rather than a needle point. A tall thin tube with a long sharp cone -- the
    first version of this glyph -- is not what a Falcon looks like.

    `volume` in {15, 50} changes only the body aspect; a 15 mL conical is visibly narrower.
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    ax0, ay0 = at
    lw = STROKE["secondary"] if lw is None else lw
    base = shade(AGAR, 0.55) if fill is None else fill
    broth = shade(base, -0.55 * float(np.clip(turbidity, 0.0, 1.0)))

    cap_h = 0.125 * h if cap else 0.0
    tube_h = h - cap_h
    aspect = 3.80 if float(volume) >= 50 else 6.90
    w = tube_h / aspect
    r = w / 2.0
    y_bot = ay0 - h / 2.0
    y_top = y_bot + tube_h
    y_cone = y_bot + 0.215 * tube_h
    tip = r * 0.16

    def _outline():
        pts = [(ax0 - r, y_top), (ax0 - r, y_cone), (ax0 - tip, y_bot + 0.020 * tube_h)]
        pts += [(ax0 - tip * 0.5, y_bot), (ax0 + tip * 0.5, y_bot)]
        pts += [(ax0 + tip, y_bot + 0.020 * tube_h), (ax0 + r, y_cone), (ax0 + r, y_top)]
        return Path(pts + [pts[0]], closed=True)

    if shadow and EFFECTS:
        sh = PathPatch(_outline(), transform=Affine2D().translate(0.30 * shadow, -shadow) + tr,
                       facecolor=INK, edgecolor="none", alpha=0.12, zorder=zorder - 0.3,
                       clip_on=False)
        ax.add_patch(sh)
        exempt(sh)

    glass = PathPatch(_outline(), transform=tr, facecolor="#F1F5F8", edgecolor="none",
                      zorder=zorder, clip_on=False)
    ax.add_patch(glass)
    exempt(glass)

    top = y_bot + level * tube_h
    lp = [(ax0 - r, top), (ax0 - r, y_cone), (ax0 - tip, y_bot + 0.020 * tube_h),
          (ax0 - tip * 0.5, y_bot), (ax0 + tip * 0.5, y_bot),
          (ax0 + tip, y_bot + 0.020 * tube_h), (ax0 + r, y_cone), (ax0 + r, top)]
    liq = PathPatch(Path(lp + [lp[0]], closed=True), transform=tr, facecolor=broth,
                    edgecolor="none", zorder=zorder + 0.2, clip_on=False)
    liq.set_clip_path(glass)
    liq.set_clip_box(None)
    ax.add_patch(liq)
    exempt(liq)
    if STYLE == "volume":
        _gradient(ax, liq, shade(broth, -0.12), shade(broth, 0.22),
                  (ax0 - r, ax0 + r, y_bot, top), tr, zorder + 0.25, radial=False)
    ml, = ax.plot([ax0 - r * 0.99, ax0 + r * 0.99], [top, top], color=shade(broth, -0.28),
                  lw=max(STROKE_MIN_PT, lw * 0.55), transform=tr, zorder=zorder + 0.3)
    ml.set_clip_path(glass)
    ml.set_clip_box(None)
    exempt(ml)

    if patch:
        from matplotlib.patches import FancyBboxPatch as _FBP
        wp = _FBP((ax0 - r * 0.62, y_top - tube_h * 0.30), r * 1.24, tube_h * 0.22,
                  boxstyle="round,pad=0,rounding_size=%.4f" % (0.05 * w),
                  facecolor="#FFFFFF", edgecolor="none", alpha=0.78, transform=tr,
                  zorder=zorder + 0.55, clip_on=False)
        ax.add_patch(wp)
        exempt(wp)

    if ticks:
        ys = [y_cone + (y_top - y_cone) * f for f in (0.10, 0.26, 0.42, 0.58, 0.74)]
        _tick_marks(ax, tr, ax0, ys, r, color=shade(INK, 0.55),
                    lw=max(STROKE_MIN_PT, lw * 0.5), zorder=zorder + 0.8, clip=glass)

    st, = ax.plot([ax0 - r * 0.52, ax0 - r * 0.52], [y_top - 0.07 * tube_h, y_cone + 0.05 * tube_h],
                  color="#FFFFFF", alpha=0.50, lw=lw * 1.6, transform=tr, zorder=zorder + 0.9,
                  solid_capstyle="round")
    st.set_clip_path(glass)
    st.set_clip_box(None)
    exempt(st)

    gl = PathPatch(_outline(), transform=tr, facecolor="none", edgecolor=INK, lw=lw,
                   zorder=zorder + 1.0, joinstyle="round", clip_on=False)
    ax.add_patch(gl)

    if cap:
        from matplotlib.patches import FancyBboxPatch as _FBP
        cw = w * 1.14
        cp = _FBP((ax0 - cw / 2, y_top - cap_h * 0.10), cw, cap_h,
                  boxstyle="round,pad=0,rounding_size=%.4f" % (0.22 * cap_h),
                  facecolor=shade(MUTE, -0.02), edgecolor=INK, lw=lw,
                  transform=tr, zorder=zorder + 1.2, clip_on=False)
        ax.add_patch(cp)
        for f in np.linspace(0.18, 0.82, 5):
            k, = ax.plot([ax0 - cw / 2 + cw * f, ax0 - cw / 2 + cw * f],
                         [y_top + cap_h * 0.12, y_top + cap_h * 0.76],
                         color=shade(MUTE, -0.34), lw=max(STROKE_MIN_PT, lw * 0.42),
                         transform=tr, zorder=zorder + 1.3, solid_capstyle="butt")
            exempt(k)

    if label:
        text_in(ax, x, y, ax0, ay0 - h / 2 - 0.075, label, coords=coords, va="top",
                size=TYPE["annotation"])
    return gl


def microfuge(ax, x, y, h=0.44, *, volume=2.0, fill=None, level=0.42, coords="axes", zorder=4,
              lw=None, label=None, shadow=0.024, at=(0.0, 0.0), tr=None, hinge="right",
              ticks=True):
    """A MICROFUGE tube -- the 1.5 / 2.0 mL snap-cap tube the lysis runs in.

    `h` = OVERALL height in inches including the closed cap, centred on (x, y).

    🔴 PROPORTIONS FROM THE PRODUCT: Thermo 3453 / the standard 2 mL snap-cap microtube, whose
    photograph shows a tube ~38 mm x ~11 mm (3.5:1) with a nearly cylindrical upper body, a
    ROUNDED BULLET bottom over the lower ~40 %, and a flat disc cap on a short hinge arm. It is
    NOT a stubby cone: the first version of this glyph was 1.9:1 and pointed, and read as a
    thimble.
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    ax0, ay0 = at
    lw = STROKE["secondary"] if lw is None else lw
    liq = shade(AGAR, 0.30) if fill is None else fill

    cap_h = 0.105 * h
    tube_h = h - cap_h
    aspect = 3.50 if float(volume) >= 2.0 else 3.85
    w = tube_h / aspect
    r = w / 2.0
    y_bot = ay0 - h / 2.0
    y_top = y_bot + tube_h
    y_round = y_bot + 0.42 * tube_h
    H = y_round - y_bot

    def _nose(n=34):
        """A blunt superellipse nose -- a 2 mL tube bottom is ROUND, not conical."""
        out = []
        for t in np.linspace(0.0, 1.0, n):
            xx = r * (1.0 - t ** 2.6) ** (1.0 / 2.6)
            out.append((xx, y_round - t * H))
        return out

    def _outline():
        nose = _nose()
        pts = [(ax0 - r, y_top), (ax0 - r, y_round)]
        pts += [(ax0 - xx, yy) for xx, yy in nose]
        pts += [(ax0 + xx, yy) for xx, yy in reversed(nose)]
        pts += [(ax0 + r, y_round), (ax0 + r, y_top)]
        return Path(pts + [pts[0]], closed=True)

    if shadow and EFFECTS:
        sh = PathPatch(_outline(), transform=Affine2D().translate(0.30 * shadow, -shadow) + tr,
                       facecolor=INK, edgecolor="none", alpha=0.12, zorder=zorder - 0.3,
                       clip_on=False)
        ax.add_patch(sh)
        exempt(sh)

    glass = PathPatch(_outline(), transform=tr, facecolor="#F1F5F8", edgecolor="none",
                      zorder=zorder, clip_on=False)
    ax.add_patch(glass)
    exempt(glass)

    top = y_bot + level * tube_h
    nose = _nose()
    lp = [(ax0 - r, top), (ax0 - r, y_round)]
    lp += [(ax0 - xx, yy) for xx, yy in nose]
    lp += [(ax0 + xx, yy) for xx, yy in reversed(nose)]
    lp += [(ax0 + r, y_round), (ax0 + r, top)]
    lq = PathPatch(Path(lp + [lp[0]], closed=True), transform=tr, facecolor=liq,
                   edgecolor="none", zorder=zorder + 0.2, clip_on=False)
    ax.add_patch(lq)
    exempt(lq)
    if STYLE == "volume":
        _gradient(ax, lq, shade(liq, -0.12), shade(liq, 0.22),
                  (ax0 - r, ax0 + r, y_bot, top), tr, zorder + 0.25, radial=False)
    ml, = ax.plot([ax0 - r * 0.99, ax0 + r * 0.99], [top, top], color=shade(liq, -0.28),
                  lw=max(STROKE_MIN_PT, lw * 0.55), transform=tr, zorder=zorder + 0.32)
    ml.set_clip_path(glass)
    ml.set_clip_box(None)
    exempt(ml)

    if ticks:
        ys = [y_round + (y_top - y_round) * f for f in (0.28, 0.58, 0.86)]
        _tick_marks(ax, tr, ax0, ys, r, color=shade(INK, 0.55),
                    lw=max(STROKE_MIN_PT, lw * 0.5), zorder=zorder + 0.8, clip=glass)

    st, = ax.plot([ax0 - r * 0.50, ax0 - r * 0.50], [y_top - 0.06 * tube_h, y_round + 0.02 * tube_h],
                  color="#FFFFFF", alpha=0.50, lw=lw * 1.4, transform=tr, zorder=zorder + 0.9,
                  solid_capstyle="round")
    st.set_clip_path(glass)
    st.set_clip_box(None)
    exempt(st)

    gl = PathPatch(_outline(), transform=tr, facecolor="none", edgecolor=INK, lw=lw,
                   zorder=zorder + 1.0, joinstyle="round", clip_on=False)
    ax.add_patch(gl)

    from matplotlib.patches import FancyBboxPatch as _FBP
    cw = w * 1.08
    lid = _FBP((ax0 - cw / 2, y_top - cap_h * 0.18), cw, cap_h,
               boxstyle="round,pad=0,rounding_size=%.4f" % (0.40 * cap_h),
               facecolor=shade(MUTE, 0.02), edgecolor=INK, lw=lw,
               transform=tr, zorder=zorder + 1.2, clip_on=False)
    ax.add_patch(lid)
    dimple = Ellipse((ax0, y_top + cap_h * 0.30), cw * 0.44, cap_h * 0.42,
                     facecolor="none", edgecolor=shade(MUTE, -0.30),
                     lw=max(STROKE_MIN_PT, lw * 0.42), transform=tr, zorder=zorder + 1.3,
                     clip_on=False)
    ax.add_patch(dimple)
    exempt(dimple)
    sgn = 1.0 if hinge == "right" else -1.0
    fx = ax0 + sgn * cw / 2
    arm = Path([(fx - sgn * cw * 0.04, y_top + cap_h * 0.06),
                (fx + sgn * cw * 0.34, y_top + cap_h * 0.20),
                (fx + sgn * cw * 0.30, y_top - cap_h * 0.26),
                (fx - sgn * cw * 0.06, y_top - cap_h * 0.22)], closed=True)
    ap = PathPatch(arm, transform=tr, facecolor=shade(MUTE, 0.02), edgecolor=INK, lw=lw,
                   zorder=zorder + 1.15, joinstyle="round", clip_on=False)
    ax.add_patch(ap)

    if label:
        text_in(ax, x, y, ax0, ay0 - h / 2 - 0.070, label, coords=coords, va="top",
                size=TYPE["annotation"])
    return gl


def vial(ax, x, y, h=0.34, *, fill=None, level=0.62, coords="axes", zorder=5, lw=None, label=None,
         at=(0.0, 0.0), tr=None, cap_frac=0.16):
    """A capped reagent vial.  `h` = vial HEIGHT in inches, centred on (x, y).

    For LiAc, PEG, carrier ssDNA, a lysis enzyme, a buffer — anything ADDED at a step.

    A vial IS a glyph because its CONTENTS join the reaction; a heat block's do not.
    
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    ax0, ay0 = at
    w = 0.46 * h
    lw = STROKE["secondary"] if lw is None else lw
    liq = shade(AGAR, 0.30) if fill is None else fill
    from matplotlib.patches import FancyBboxPatch
    body_h = h * (1 - cap_frac)
    y0 = ay0 - h / 2

    def _body_path():
        return FancyBboxPatch((ax0 - w / 2, y0), w, body_h,
                              boxstyle="round,pad=0,rounding_size=%.4f" % (0.14 * w),
                              facecolor="#F1F5F8", edgecolor="none", transform=tr,
                              zorder=zorder, clip_on=False)
    g = _body_path()
    ax.add_patch(g)
    exempt(g)
    lh = body_h * level
    lq = FancyBboxPatch((ax0 - w / 2, y0), w, lh,
                        boxstyle="round,pad=0,rounding_size=%.4f" % (0.14 * w),
                        facecolor=liq, edgecolor="none", transform=tr, zorder=zorder + 0.2,
                        clip_on=False)
    lq.set_clip_path(g)
    lq.set_clip_box(None)
    ax.add_patch(lq)
    exempt(lq)
    out = FancyBboxPatch((ax0 - w / 2, y0), w, body_h,
                         boxstyle="round,pad=0,rounding_size=%.4f" % (0.14 * w),
                         facecolor="none", edgecolor=INK, lw=lw, transform=tr,
                         zorder=zorder + 0.6, clip_on=False)
    ax.add_patch(out)
    cap = FancyBboxPatch((ax0 - w * 0.34, y0 + body_h - lw / 144), w * 0.68, h * cap_frac,
                         boxstyle="round,pad=0,rounding_size=%.4f" % (0.10 * w),
                         facecolor=shade(MUTE, -0.10), edgecolor=INK, lw=lw, transform=tr,
                         zorder=zorder + 0.7, clip_on=False)
    ax.add_patch(cap)
    if label:
        text_in(ax, x, y, ax0, ay0 - h / 2 - 0.065, label, coords=coords, va="top",
                size=TYPE["annotation"])
    return out


def condition_mark(ax, x, y, kind, text, *, d=0.17, coords="axes", zorder=6, size=None,
                   gap=0.055, color=None, at=(0.0, 0.0), tr=None):
    """A small CONDITION mark plus its value — `kind` in {"temp", "time", "light", "dark"}.

    "120 °C, 15 min" carries everything the drawn autoclave carries, at a fraction of the ink, and
    it does not date when the lab buys a different machine. So the thermometer and the hourglass
    survive as small marks on a step; the equipment does not survive at all.

    This is the same shape as `condition()` for light and dark: the mark is the cue, the value is
    the content, and the vessel it applies to is drawn elsewhere.
    
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    ax0, ay0 = at
    col = INK if color is None else color
    lw = max(STROKE_MIN_PT, STROKE["secondary"] * 0.9)
    if kind == "temp":
        bw = d * 0.26
        st, = ax.plot([ax0, ax0], [ay0 - d * 0.18, ay0 + d * 0.46], color=col, lw=lw * 1.5,
                      transform=tr, zorder=zorder, solid_capstyle="round")
        b = Circle((ax0, ay0 - d * 0.30), bw * 0.62, facecolor=EDIT, edgecolor=col, lw=lw,
                   transform=tr, zorder=zorder + 0.2, clip_on=False)
        ax.add_patch(b)
        f, = ax.plot([ax0, ax0], [ay0 - d * 0.26, ay0 + d * 0.20], color=EDIT, lw=lw * 0.9,
                     transform=tr, zorder=zorder + 0.3, solid_capstyle="round")
        exempt(f)
    elif kind == "time":
        hw, hh = d * 0.24, d * 0.38
        pts = [(ax0 - hw, ay0 + hh), (ax0 + hw, ay0 + hh), (ax0, ay0),
               (ax0 + hw, ay0 - hh), (ax0 - hw, ay0 - hh), (ax0, ay0)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=shade(MUTE, 0.35), edgecolor=col,
                             lw=lw, transform=tr, zorder=zorder, clip_on=False,
                             joinstyle="round"))
    else:
        (sun if kind == "light" else moon)(ax, x, y, d, coords=coords, zorder=zorder)
    reach = d / 2.0 * (1.96 if kind == "light" else 1.02)
    text_in(ax, x, y, ax0 + reach + gap, ay0, text, coords=coords, ha="left",
            size=TYPE["annotation"] if size is None else size, color=INK)


def _streak_paths(r, quadrants=4, seed=0):
    """Colony positions for a quadrant isolation streak, with a confluence flag per colony.

    🔴 A STREAK IS A PROGRESSION, NOT A PATTERN. Quadrant 1 is the inoculum smear — cells so dense
    they merge into a continuous line. Each later quadrant is drawn from the previous with a flamed
    loop, so the spacing along the stroke grows until, in the last quadrant, colonies are DISCRETE.
    **The last quadrant is the entire purpose**: a streak that does not end in single colonies has
    not isolated anything, and drawing one that does not is a claim the plate never supported.

    The first build drew four short clumps, which read as four smears and showed no progression at
    all. These are long parallel STROKES sweeping across the plate, three or four per quadrant,
    with the along-stroke gap growing quadrant by quadrant — which is what a loop actually leaves.
    """
    rng = np.random.RandomState(seed)
    out = []
    for q in range(quadrants):
        base = math.radians(150 - q * 74)
        n_lines = 4 if q == 0 else 3
        gap = (0.34, 1.25, 2.30, 3.60)[min(q, 3)]
        conf = q == 0
        for li in range(n_lines):
            t = (li + 0.5) / n_lines
            a0 = base + math.radians(-30 + 60 * t)
            r0, r1_ = 0.20 * r + 0.10 * r * t, 0.84 * r
            n_cell = max(2, int((r1_ - r0) / (gap * 0.085 * r)))
            for k in range(n_cell):
                u = (k + 0.5) / n_cell
                rr = r0 + (r1_ - r0) * u
                th = a0 + rng.uniform(-0.045, 0.045) + 0.22 * (u - 0.5)
                out.append((rr * math.cos(th), rr * math.sin(th), q, conf))
    return out


def _streak_zigzag(r, quadrants=4, seed=0):
    """The loop's actual PATH, quadrant by quadrant — the serial streak as it is performed.

    Each quadrant is a tight back-and-forth zigzag; each new one starts by crossing the tail of the
    previous, which is what carries cells forward, and is rotated so the four sweep round the dish.
    Returns one polyline per quadrant, so the caller can draw the early ones as continuous growth
    and the late ones as the discrete colonies they resolve into.
    """
    rng = np.random.RandomState(seed)
    out = []
    for q in range(quadrants):
        rot = math.radians(140 - q * 76)
        n_sw = (9, 7, 6, 4)[min(q, 3)]
        span, depth = 0.66 * r, 0.22 * r
        cxq, cyq = 0.44 * r * math.cos(rot), 0.44 * r * math.sin(rot)
        pts = []
        for k in range(n_sw):
            u = k / max(n_sw - 1, 1)
            xx = -span / 2 + span * u
            for yy in ((depth, -depth) if k % 2 == 0 else (-depth, depth)):
                jx = rng.uniform(-0.012, 0.012) * r
                pts.append((xx + jx, yy * (0.72 + 0.28 * rng.rand())))
        ca, sa = math.cos(rot + math.pi / 2), math.sin(rot + math.pi / 2)
        out.append([(cxq + px * ca - py * sa, cyq + px * sa + py * ca) for px, py in pts])
    return out


def plate(ax, x, y, d=0.80, *, colonies=26, picked=0, fill=None, colony_d=None, seed=5,
          coords="axes", zorder=4, lw=None, label=None, agar=None, organism="yeast",
          lawn=False, view="top", pattern="spread", quadrants=4, tilt=0.42, at=(0.0, 0.0),
          tr=None, streak_style="colonies"):
    """A petri dish.  `d` = dish diameter in inches.

    `view="top"` looks straight down; `view="side"` is the three-quarter view, where the agar
    surface becomes an ellipse of aspect `tilt` and the dish gains a wall and a lid rim. **The side
    view is not a decoration of the top view** — it is what you draw when the plate is one object in
    a bench sequence, because a flat circle at the end of an arrow reads as a symbol rather than a
    thing on a bench.

    `pattern` is `"spread"` (random, count is the message), `"streak"` (isolation, the progression
    to single colonies is the message) or `"none"`.
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    ax0, ay0 = at
    r = d / 2.0
    lw = STROKE["primary"] if lw is None else lw
    ag = AGAR if agar is None else agar
    cd = 0.085 * d if colony_d is None else colony_d
    side = view == "side"
    ry = r * tilt if side else r

    if side:
        wall_h = 0.46 * r
        lip_dy = 0.30 * r
        th = np.linspace(math.pi, 2 * math.pi, 90)
        sh = Ellipse((ax0 + 0.016, ay0 - wall_h - 0.030), 2 * r, 2 * ry, facecolor=INK,
                     edgecolor="none", alpha=0.12, transform=tr, zorder=zorder - 0.3,
                     clip_on=False)
        ax.add_patch(sh)
        exempt(sh)
        lip_pts = [(ax0 + r * math.cos(t), ay0 + lip_dy + ry * math.sin(t)) for t in th]
        base_pts = [(ax0 + r * math.cos(t), ay0 - wall_h + ry * math.sin(t)) for t in th[::-1]]
        wall = PathPatch(Path(lip_pts + base_pts + [lip_pts[0]], closed=True), transform=tr,
                         facecolor="#EDF2F6", edgecolor=INK, lw=lw,
                         zorder=zorder + 0.1, clip_on=False, joinstyle="round")
        ax.add_patch(wall)
        agar_side = PathPatch(Path([(ax0 + r * 0.985 * math.cos(t), ay0 + ry * 0.985 * math.sin(t))
                                    for t in th]
                                   + [(ax0 + r * 0.985 * math.cos(t),
                                       ay0 - wall_h + ry * 0.985 * math.sin(t))
                                      for t in th[::-1]], closed=True),
                             transform=tr, facecolor=shade(ag, -0.10), edgecolor="none",
                             zorder=zorder + 0.15, clip_on=False)
        ax.add_patch(agar_side)
        exempt(agar_side)
        lip = Ellipse((ax0, ay0 + lip_dy), 2 * r, 2 * ry, facecolor="none", edgecolor=INK,
                      lw=lw, transform=tr, zorder=zorder + 0.95, clip_on=False)
        ax.add_patch(lip)
        lipi = Ellipse((ax0, ay0 + lip_dy), 2 * r * 0.945, 2 * ry * 0.945, facecolor="none",
                       edgecolor=shade(INK, 0.55), lw=max(STROKE_MIN_PT, lw * 0.5),
                       transform=tr, zorder=zorder + 0.95, clip_on=False)
        ax.add_patch(lipi)
        exempt(lipi)
    dish = _body(ax, lambda: Ellipse((ax0, ay0), 2 * r, 2 * ry), tr, fill=ag, zorder=zorder + 0.3,
                 lw=lw, edge=INK, shadow=0.0 if side else 0.030 * d,
                 gloss=(ax0 - 0.34 * r, ay0 + 0.52 * ry, 0.72 * r, 0.24 * ry, -22, 0.42),
                 extent=(ax0 - r, ax0 + r, ay0 - ry, ay0 + ry))
    rim = Ellipse((ax0, ay0), 2 * 0.905 * r, 2 * 0.905 * ry, facecolor="none",
                  edgecolor=shade(ag, -0.30), lw=max(STROKE_MIN_PT, lw * 0.55), transform=tr,
                  zorder=zorder + 0.7, clip_on=False)
    ax.add_patch(rim)

    if lawn:
        lwc = shade(COLONY, -0.06)
        cov = Ellipse((ax0, ay0), 2 * 0.86 * r, 2 * 0.86 * ry, facecolor=lwc,
                      edgecolor=shade(lwc, -0.22), lw=max(STROKE_MIN_PT, lw * 0.5),
                      transform=tr, zorder=zorder + 1, clip_on=False)
        ax.add_patch(cov)
        exempt(cov)
        rng = np.random.RandomState(seed)
        for _ in range(90):
            a, rr = rng.rand() * 2 * math.pi, math.sqrt(rng.rand()) * 0.80 * r
            t = Ellipse((ax0 + rr * math.cos(a), ay0 + rr * tilt * math.sin(a) if side
                         else ay0 + rr * math.sin(a)), 0.060 * r * (0.6 + rng.rand()),
                        0.060 * r * (0.6 + rng.rand()) * (tilt if side else 1.0),
                        facecolor=shade(lwc, -0.10), edgecolor="none", transform=tr,
                        zorder=zorder + 1.1, clip_on=False)
            ax.add_patch(t)
            exempt(t)
    elif pattern == "streak" and streak_style == "serial":
        polys = _streak_zigzag(r, quadrants, seed)
        for q, pts in enumerate(polys):
            xs = [ax0 + px for px, py in pts]
            ys = [ay0 + (py * tilt if side else py) for px, py in pts]
            if q < quadrants - 1:
                width = cd * 72.0 * (0.40 - 0.06 * q)
                back, = ax.plot(xs, ys, color=_rim(COLONY), lw=width + 1.0, transform=tr,
                                zorder=zorder + 1 + q * 0.1, clip_on=False,
                                solid_capstyle="round", solid_joinstyle="round")
                fore, = ax.plot(xs, ys, color=COLONY, lw=width, transform=tr,
                                zorder=zorder + 1.02 + q * 0.1, clip_on=False,
                                solid_capstyle="round", solid_joinstyle="round")
                exempt(back)
                exempt(fore)
            else:
                seg = np.linspace(0, len(pts) - 1.001, 9)
                for j, t in enumerate(seg):
                    i0 = int(t)
                    fr = t - i0
                    px = pts[i0][0] + (pts[i0 + 1][0] - pts[i0][0]) * fr
                    py = pts[i0][1] + (pts[i0 + 1][1] - pts[i0][1]) * fr
                    colony(ax, x, y, cd * 0.92, tr=tr,
                           at=(ax0 + px, ay0 + (py * tilt if side else py)),
                           organism=organism, zorder=zorder + 1.4 + j * 0.01, maturity=0.95,
                           fill=fill if fill else None)
    elif pattern == "streak":
        _pts = _streak_paths(r, quadrants, seed)
        _lines = {}
        for cx, cy, q, conf in _pts:
            if conf:
                _lines.setdefault(q, []).append((cx, cy * tilt if side else cy))
        for q, pts in _lines.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            sm, = ax.plot([ax0 + v for v in xs], [ay0 + v for v in ys], color=COLONY,
                          lw=cd * 72.0 * 0.92, solid_capstyle="round", solid_joinstyle="round",
                          transform=tr, zorder=zorder + 1, clip_on=False)
            exempt(sm)
            smr, = ax.plot([ax0 + v for v in xs], [ay0 + v for v in ys],
                           color=_rim(COLONY), lw=cd * 72.0 * 0.92 + 1.1,
                           solid_capstyle="round", solid_joinstyle="round", transform=tr,
                           zorder=zorder + 0.98, clip_on=False)
            exempt(smr)
        i = 0
        for cx, cy, q, conf in _pts:
            if conf:
                continue
            i += 1
            cy2 = cy * tilt if side else cy
            colony(ax, x, y, cd * (0.86 + 0.26 * ((i * 37) % 7) / 7.0), tr=tr,
                   at=(ax0 + cx, ay0 + cy2), organism=organism,
                   zorder=zorder + 1.2 + i * 0.004, maturity=0.95,
                   fill=fill if fill else None)
    elif pattern == "spread":
        rng = np.random.RandomState(seed)
        pts, lim = [], 0.80 * r
        while len(pts) < colonies and len(pts) < 400:
            a, rr = rng.rand() * 2 * math.pi, math.sqrt(rng.rand()) * lim
            X, Y = rr * math.cos(a), rr * math.sin(a)
            if all((X - u) ** 2 + (Y - v) ** 2 > (1.55 * cd) ** 2 for u, v in pts):
                pts.append((X, Y))
            elif rng.rand() < 0.004:
                break
        for i, (cx, cy) in enumerate(pts):
            f = fill if (fill and i < picked) else None
            colony(ax, x, y, cd * (0.82 + 0.36 * rng.rand()), fill=f, tr=tr,
                   at=(ax0 + cx, ay0 + (cy * tilt if side else cy)),
                   zorder=zorder + 1 + i * 0.01, picked=i < picked, organism=organism,
                   maturity=0.55 + 0.45 * rng.rand())
    if label:
        text_in(ax, x, y, ax0, ay0 - ry - (0.30 * r if side else 0.0) - 0.085, label,
                coords=coords, va="top", size=TYPE["annotation"])
    return dish


def array_plate(ax, x, y, w=1.30, *, density=96, filled=None, fill=None, coords="axes", zorder=4,
                lw=None, label=None, agar=None, organism="yeast", seed=3, at=(0.0, 0.0),
                tr=None):
    """An SBS-footprint rectangular plate carrying a ROBOT-PINNED ARRAY.

    `density` is 96, 384 or 1536 — the standard pinning densities, which are 12x8, 24x16 and 48x32
    on the same 128 x 86 mm footprint. That is why the plate does not change size between them:
    **the density is the whole message, and a plate that grew with its grid would hide it.**

    🔴 THE GLYPH DEGRADES ITSELF, because 1536 colonies at figure size are ~0.15 pt apart. Below the
    point where a reader could resolve one, drawing 1536 discrete circles is a lie about what the
    page can show — it becomes a texture with the grid still legible as rows and columns. The
    `density` label is then doing the work the dots cannot, which is the honest division of labour.
    🔴 All three densities share ONE outer boundary: the corner colonies sit at identical positions
    and only the colony size changes, because each density is a 2x interleave of the previous one.
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    ax0, ay0 = at
    lw = STROKE["secondary"] if lw is None else lw
    ag = AGAR if agar is None else agar
    GRID = {96: (12, 8), 384: (24, 16), 1536: (48, 32)}
    if density not in GRID:
        raise ValueError("density must be 96, 384 or 1536 (SBS pinning formats), got %r" % density)
    cols, rows = GRID[density]
    h = w * 86.0 / 128.0
    from matplotlib.patches import FancyBboxPatch
    rr = 0.030 * w
    sh = FancyBboxPatch((ax0 - w / 2 + 0.014, ay0 - h / 2 - 0.026), w, h,
                        boxstyle="round,pad=0,rounding_size=%.4f" % rr, facecolor=INK,
                        edgecolor="none", alpha=0.12, transform=tr, zorder=zorder - 0.3,
                        clip_on=False)
    ax.add_patch(sh)
    exempt(sh)
    body = FancyBboxPatch((ax0 - w / 2, ay0 - h / 2), w, h,
                          boxstyle="round,pad=0,rounding_size=%.4f" % rr, facecolor=ag,
                          edgecolor=INK, lw=lw, transform=tr, zorder=zorder, clip_on=False)
    ax.add_patch(body)
    pad = 0.085 * w
    x0i, x1i = ax0 - w / 2 + pad, ax0 + w / 2 - pad
    y0i, y1i = ay0 - h / 2 + pad, ay0 + h / 2 - pad
    dx = (x1i - x0i) / max(cols - 1, 1)
    dy = (y1i - y0i) / max(rows - 1, 1)
    cdia = 0.72 * min(dx, dy)
    resolvable = cdia * 72.0 >= 1.6
    rng = np.random.RandomState(seed)
    n = 0
    for ri in range(rows):
        for ci in range(cols):
            cx = x0i + dx * ci
            cy = y1i - dy * ri
            on = filled is None or n < filled
            n += 1
            if not on:
                continue
            f = fill if fill else COLONY
            if resolvable:
                colony(ax, x, y, cdia, tr=tr, at=(cx, cy), organism=organism,
                       zorder=zorder + 1 + n * 0.001, fill=f,
                       maturity=0.6 + 0.4 * rng.rand())
            else:
                c = Circle((cx, cy), cdia / 2.0, facecolor=f, edgecolor=shade(f, -0.30),
                           lw=max(0.20, lw * 0.22), transform=tr, zorder=zorder + 1,
                           clip_on=False)
                ax.add_patch(c)
                exempt(c)
    if label:
        text_in(ax, x, y, ax0, ay0 - h / 2 - 0.075, label, coords=coords, va="top",
                size=TYPE["annotation"])
    return body


def _counterform(face):
    """The colour to PUNCH a hole with, given the body it is punched in."""
    r, g, b = _rgb(face)
    pr, pg, pb = _rgb(PAPER)
    near_paper = (abs(r - pr) + abs(g - pg) + abs(b - pb)) < 0.18
    return _mix(face, _rim(face), 0.55) if near_paper else PAPER


def pombe(ax, x, y, d=0.22, *, fill=None, lw=None, edge=None, coords="axes", zorder=5,
          septum=False, dividing=False, nucleus=True, tilt=-25.0, at=(0.0, 0.0), tr=None, pool=False):
    """*S. pombe*: a tilted rod with hemispherical ends.  `d` = CELL LENGTH in inches.

    * **No septum.** In the sources a fission-yeast septum is a band 0.024 to 0.028 of cell width, which
      is 0.06 to 0.13 pt at 3 to 6 mm -- below even Nature's 0.25 pt floor, let alone this package's 0.6 pt
      one. Drawn at a stroke it can survive, it IS the pill's seam. `septum=True` is kept only for a
      caller drawing a cell far larger than a species mark, and is off by default.
    * **No gloss.** `pill_blue.svg` renders its body with a linear gradient across the short axis; the
      first version put a specular highlight in the same place. Both PhyloPic fission yeasts carry no
      interior tone at all.
    * **Tilted.** Measured tilts: fission yeast -19.3, +14.5, -22.6, -35.5 and -39.4 degrees (4 sources);
      every rod that is NOT a fission yeast sits within 7.4 degrees of horizontal (both pills, a
      bacterial rod, *B. subtilis*). The classes do not overlap, so the tilt alone separates them, at
      any size, for free.
    * **Aspect 3.4**, `w = 0.294 d`. Measured pombe aspects are 3.23 to 3.99 (mean 3.5); a pill is 2.93
      to 2.99. The first version's 2.38 was too stubby to be either.

    `nucleus` punches a background-coloured counterform at or above 0.157 in (4 mm), which is where the
    measurements say it still reads: diameter 0.55 of the cell width at mid-length, the way PhyloPic
    punches a mouse eye, never a stroked circle, which would cost a second rim.

    `dividing=True` draws the DBCLS construction instead: two compartments, each separately closed and
    rimmed. That is the honest way to show a septum at this size, because a pill's seam is a line on one
    body while a septum is a break between two. It needs 4 mm, and the sources draw a dividing cell
    LONGER (aspect 3.99) rather than cutting one cell in half.
    
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    lw = (STROKE["secondary"] if not pool else STROKE["hairline"]) if lw is None else lw
    face = CELL if fill is None else fill
    ax0, ay0 = at
    w = 0.294 * d
    r = w / 2.0
    ca, sa = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))

    def place(pts):
        return [(ax0 + px * ca - py * sa, ay0 + px * sa + py * ca) for px, py in pts]

    def capsule(cx, length):
        half = max(length / 2.0 - r, 1e-6)
        pts = [(cx + half + r * math.cos(t), r * math.sin(t))
               for t in np.linspace(-math.pi / 2, math.pi / 2, 90)]
        pts += [(cx - half + r * math.cos(t), r * math.sin(t))
                for t in np.linspace(math.pi / 2, 1.5 * math.pi, 90)]
        return pts

    if dividing and d >= 0.157:
        gap = 0.3 / 72.0
        seg = (d - gap) / 2.0
        for cx in (-(seg + gap) / 2.0, (seg + gap) / 2.0):
            pts = place(capsule(cx, seg))
            _body(ax, Path(pts + [pts[0]], closed=True), tr, fill=face, zorder=zorder, lw=lw, edge=edge,
                  gloss=None, extent=None)
    else:
        pts = place(capsule(0.0, d))
        _body(ax, Path(pts + [pts[0]], closed=True), tr, fill=face, zorder=zorder, lw=lw, edge=edge,
              gloss=None, extent=None)
        if septum:
            a_, b_ = place([(0.0, -r * 0.92)])[0], place([(0.0, r * 0.92)])[0]
            ax.plot([a_[0], b_[0]], [a_[1], b_[1]], color=_rim(face), lw=lw * 0.9, transform=tr,
                    zorder=zorder + 0.3, clip_on=False, solid_capstyle="butt")
        if nucleus and d >= 0.157:
            c_ = place([(0.0, 0.0)])[0]
            ax.add_patch(Circle(c_, 0.55 * w / 2.0, facecolor=_counterform(face), edgecolor="none",
                                transform=tr, zorder=zorder + 0.25, clip_on=False))
    return tr


def mouse(ax, x, y, d=0.22, *, fill=None, lw=None, edge=None, coords="axes", zorder=5,
          facing="right", tail=True, eye=True, at=(0.0, 0.0), tr=None, pool=False):
    """Mouse: a two-lobe body and head with a muzzle, one ear lobe and a low tail.  `d` = RUMP-TO-NOSE.

    * **The tail does not rise.** Of 13 marks the tip sits low in 9 and high in 4. The first version
      finished 0.28 d ABOVE the body axis, which is the minority choice and reads as a decorative flick;
      it now dips to -0.08 d and returns to +0.05 d, one inflection, the way three of the sources draw it.
    * **The tail length was already right** (exposed run 0.42 of body against a published median of 0.39),
      so it is unchanged. An earlier draft of the reference told me to lengthen it; that was withdrawn on
      re-measurement, and the length stays.
    * **The head was 35 % too long.** Ear-rear to nose over rump to nose measures 0.31 in the reference
      silhouette against 0.42 here, so `r_head` drops from 0.205 d to 0.17 d and the muzzle, not the
      skull, carries the length.

    The EAR is the first mouse-versus-rat discriminator (0.51 of head diameter here, 0.49 measured in a
    mouse, 0.29 in the rat the same illustrator drew), so its size is unchanged; what changed is that it
    now sits proud of the skull instead of behind it. The MUZZLE is the second discriminator: short and
    blunt for a mouse, long and tapering for a rat.
    
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    lw = (STROKE["secondary"] if not pool else STROKE["hairline"]) if lw is None else lw
    face = CELL if fill is None else fill
    ax0, ay0 = at
    s = 1.0 if facing == "right" else -1.0
    r_body, r_head = 0.27 * d, 0.17 * d
    path, (head_x, _cx) = _union_outline(r_body, r_head, 0.86, 0.88, 0.0)
    hx, hy = s * head_x, 0.02 * d
    verts = [(ax0 + s * px, ay0 + py) for px, py in path.vertices]
    nose = [(ax0 + hx - s * 0.02 * d, ay0 + hy + 0.10 * d),
            (ax0 + hx + s * (r_head + 0.07 * d), ay0 + hy - 0.04 * d),
            (ax0 + hx - s * 0.02 * d, ay0 + hy - 0.12 * d)]
    _body(ax, Path(nose + [nose[0]], closed=True), tr, fill=face, zorder=zorder - 0.4, lw=lw, edge=edge,
          gloss=None, extent=None)
    _body(ax, Path(verts + [verts[0]], closed=True), tr, fill=face, zorder=zorder, lw=lw, edge=edge,
          gloss=(ax0 - 0.10 * d, ay0 + 0.13 * d, 0.46 * d, 0.20 * d, 0) if not pool else None,
          extent=(ax0 - 0.5 * d, ax0 + 0.5 * d, ay0 - 0.3 * d, ay0 + 0.3 * d))
    er = 0.105 * d
    _body(ax, lambda: Circle((ax0 + hx - s * 0.055 * d, ay0 + hy + 0.195 * d), er), tr, fill=face,
          zorder=zorder + 0.25, lw=lw, edge=edge, extent=None)
    if eye and d >= 0.20:
        ax.add_patch(Circle((ax0 + hx + s * 0.055 * d, ay0 + hy + 0.035 * d), 0.0225 * d,
                            facecolor=_counterform(face), edgecolor="none", transform=tr,
                            zorder=zorder + 0.3, clip_on=False))
    if tail:
        t = np.linspace(0, 1, 56)
        tx = ax0 - s * (0.20 + 0.40 * t) * d
        dip = -0.22 * np.sin(math.pi * np.clip(t / 0.55, 0.0, 1.0))
        rise = 0.13 * np.clip((t - 0.55) / 0.45, 0.0, None) ** 1.6
        ty = ay0 + np.where(t < 0.55, dip, rise) * d
        assert np.all(np.isfinite(ty)), "mouse tail: non-finite y"

        for seg in range(len(t) - 1):
            ax.plot(tx[seg:seg + 2], ty[seg:seg + 2], color=_rim(face),
                    lw=max(lw * (1.15 - 0.6 * seg / len(t)), STROKE_MIN_PT), transform=tr,
                    zorder=zorder - 0.3, clip_on=False, solid_capstyle="round")
    return tr


def human(ax, x, y, d=0.22, *, fill=None, lw=None, edge=None, coords="axes", zorder=5,
          at=(0.0, 0.0), tr=None, pool=False):
    """Human: head over shoulders.  `d` = TOTAL HEIGHT of the bust in inches.

    A bust, not a standing figure: at 4 mm a whole body is 1.3 mm wide and its limbs are mush beside a
    mouse and a cell. Recorded deliberately, because PhyloPic -- the library actually used for species
    marks in comparative figures -- draws humans as full-body side views in 5 of 5 cases. The bust is
    the UI convention, chosen here for legibility at mark size.
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    lw = (STROKE["secondary"] if not pool else STROKE["hairline"]) if lw is None else lw
    face = CELL if fill is None else fill
    ax0, ay0 = at
    hr = 0.25 * d
    hy = 0.5 * d - hr
    gap, sh_h, sh_w = 0.09 * d, 0.41 * d, 0.88 * d
    top, bot = -0.5 * d + sh_h, -0.5 * d
    plat, arc = 0.18 * d, 0.35 * d
    corner = 0.058 * d
    pts = [(ax0 - sh_w / 2.0, ay0 + bot + corner)]
    pts += [(ax0 - sh_w / 2.0 + arc - arc * math.cos(t), ay0 + top - arc + arc * math.sin(t))
            for t in np.linspace(0, math.pi / 2, 40)]
    pts += [(ax0 + plat / 2.0, ay0 + top)]
    pts += [(ax0 + sh_w / 2.0 - arc + arc * math.sin(t), ay0 + top - arc + arc * math.cos(t))
            for t in np.linspace(0, math.pi / 2, 40)]
    pts += [(ax0 + sh_w / 2.0, ay0 + bot + corner)]
    pts += [(ax0 + sh_w / 2.0 - corner + corner * math.cos(t), ay0 + bot + corner - corner * math.sin(t))
            for t in np.linspace(0, math.pi / 2, 14)]
    pts += [(ax0 - sh_w / 2.0 + corner, ay0 + bot)]
    pts += [(ax0 - sh_w / 2.0 + corner - corner * math.sin(t), ay0 + bot + corner - corner * math.cos(t))
            for t in np.linspace(0, math.pi / 2, 14)]
    _body(ax, Path(pts + [pts[0]], closed=True), tr, fill=face, zorder=zorder, lw=lw, edge=edge,
          gloss=None, extent=(ax0 - sh_w / 2, ax0 + sh_w / 2, ay0 + bot, ay0 + top))
    _body(ax, lambda: Circle((ax0, ay0 + hy), hr), tr, fill=face, zorder=zorder + 0.2, lw=lw, edge=edge,
          gloss=(ax0 - 0.30 * hr, ay0 + hy + 0.34 * hr, 0.70 * hr, 0.40 * hr, -30) if not pool else None,
          extent=(ax0 - hr, ax0 + hr, ay0 + hy - hr, ay0 + hy + hr))
    return tr


def thaliana(ax, x, y, d=0.22, *, fill=None, lw=None, edge=None, coords="axes", zorder=5, n=7,
             at=(0.0, 0.0), tr=None, pool=False):
    """*Arabidopsis thaliana*: a rosette seen from above.  `d` = ROSETTE DIAMETER in inches.

    The plant biologist's icon for Arabidopsis is the vegetative rosette from above, not a flowering
    side view: at 4 mm a bolt and its siliques are a hairline and a smudge, while a ring of leaves is
    still a ring. What keeps it from reading as a FLOWER, the one shape it could be mistaken for, is
    construction rather than detail:
    """
    tr = inch_axes(ax, x, y, coords) if tr is None else tr
    lw = (STROKE["secondary"] if not pool else STROKE["hairline"]) if lw is None else lw
    face = CELL if fill is None else fill
    ax0, ay0 = at
    R = d / 2.0
    r_in, s_pet, w_pet = 0.0, 0.24, 0.065 * R
    ss = np.linspace(0.0, 1.0, 70)
    uu = np.clip((ss - s_pet) / (1.0 - s_pet), 0.0, 1.0)
    prof = np.sqrt(uu * (1.0 - uu)) * (0.60 + 0.80 * uu)
    prof = prof / prof.max()
    for i in range(n):
        L = R * (1.0 - 0.38 * i / (n - 1))
        th = math.radians(18.0 + 137.5 * i)
        ca, sa = math.cos(th), math.sin(th)
        rr = r_in + ss * (L - r_in)
        ww = np.maximum(0.34 * L * prof, np.where(ss < 0.97, w_pet, 0.0))
        half = list(zip(rr, ww)) + list(zip(rr[::-1], -ww[::-1]))
        pts = [(ax0 + px * ca - py * sa, ay0 + px * sa + py * ca) for px, py in half]
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        _body(ax, Path(pts + [pts[0]], closed=True), tr, fill=face, zorder=zorder + 0.01 * i, lw=lw, edge=edge,
              gloss=None, extent=(min(xs), max(xs), min(ys), max(ys)))
    return tr


SPECIES = {"cerevisiae": "yeast", "pombe": "pombe", "mouse": "mouse", "human": "human", "thaliana": "thaliana"}


def species(ax, x, y, which, *, d=0.22, coords="axes", zorder=5, fill=None, lw=None,
            at=(0.0, 0.0), tr=None, pool=False, **kw):
    """One species mark by name: "cerevisiae", "pombe", "mouse", "human" or "thaliana".

    `d` is the mark's own natural dimension (a length for the two cells, a height for the bust), so a
    row of marks drawn at one `d` is NOT visually equal-weight; `species_row` solves that instead of
    asking every caller to.
    """
    if which not in SPECIES:
        raise ValueError(f"species(): unknown {which!r}; known marks are {sorted(SPECIES)}")
    if which == "cerevisiae":
        st = dict(state="edited", fill=fill, mark="body") if fill is not None else {}
        return yeast(ax, x, y, d, view="surface", detail=0, nucleus=False, coords=coords,
                     zorder=zorder, lw=lw, at=at, tr=tr, pool=pool, **st, **kw)
    return {"pombe": pombe, "mouse": mouse, "human": human, "thaliana": thaliana}[which](
        ax, x, y, d, coords=coords, zorder=zorder, fill=fill, lw=lw, at=at, tr=tr, pool=pool, **kw)


_SP_WEIGHT = {"cerevisiae": 1.00, "pombe": 1.46, "mouse": 1.34, "human": 0.92, "thaliana": 1.40}


def species_row(ax, x, y, names, *, d=0.20, pitch=None, coords="axes", zorder=5, label=True,
                size=None, color=None, vertical=False, gap=0.055, fill=None, lw=None):
    """A row (or column) of species marks with their names.

    🔴 The marks are NOT drawn at one `d`. A rod is long and thin, a bust is tall and narrow, and a
    budded ovoid is round: at equal nominal size the rod dominates and the bust disappears. Each mark
    is scaled by its own optical weight (`_SP_WEIGHT`) so the ROW reads as one family, which is the
    whole point of a species cue.

    Latin binomials are italic and vernacular names are not, so the caller writes "pombe" and gets
    *S. pombe* while "mouse" stays upright. That rule is the reason this helper exists rather than a
    loop at each call site: it was got wrong by hand twice in the deck.
    """
    lab = {"cerevisiae": ("S. cerevisiae", True), "pombe": ("S. pombe", True),
           "mouse": ("mouse", False), "human": ("human", False),
           "thaliana": ("A. thaliana", True)}
    fs = TYPE["annotation"] if size is None else size
    if pitch is None:
        pitch = d * 1.30 if vertical else d * 1.75
        if label and not vertical:
            fig = ax.figure
            fig.canvas.draw()
            r_ = fig.canvas.get_renderer()
            w_in = 0.0
            for nm in names:
                t_ = ax.text(0, 0, lab[nm][0], fontsize=fs, style="italic" if lab[nm][1] else "normal")
                w_in = max(w_in, t_.get_window_extent(renderer=r_).width / fig.dpi)
                t_.remove()
            pitch = max(pitch, w_in + gap)
    tr = inch_axes(ax, x, y, coords)
    for i, nm in enumerate(names):
        dx = 0.0 if vertical else i * pitch
        dy = -i * pitch if vertical else 0.0
        species(ax, x, y, nm, d=d * _SP_WEIGHT[nm], coords=coords, zorder=zorder, fill=fill, lw=lw,
                at=(dx, dy), tr=tr)
        if label:
            word, ital = lab[nm]
            text_in(ax, x, y, dx + (gap + d * 0.62 if vertical else 0.0),
                    dy + (0.0 if vertical else -(d * 0.62 + gap)), word, coords=coords,
                    ha="left" if vertical else "center", va="center" if vertical else "top",
                    size=fs, color=INK if color is None else color,
                    style="italic" if ital else "normal")
