"""panels.py -- LAYER A of the RNA figure standard: argument-role colour for data panels,
the default type scale, and the type / stroke floors with their checkers.

FLOORS (tokens.json -> typography.cap_min_mm, geometry.stroke_min_pt), enforced by
`check_type_floor` / `check_stroke_floor` -- shared with the schematic module so both
styles bound type and stroke the same way. `fig.texts` (provenance footer, stamp, concise legend --
the figure standard fixes those sizes) is auto-exempt; anything else smaller must call `exempt()`.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "TOK", "TOKENS_PATH", "FOCAL", "REFERENCE", "MATCH", "STRATUM_A", "STRATUM_B", "CONTROL", "NEUTRAL",
    "INK", "HAIRLINE", "SUBTLE", "MUTE", "WASH", "PAPER", "FOCAL_TINT", "REFERENCE_TINT", "MATCH_TINT",
    "ROLES", "TYPE", "SLIDE", "STROKE", "CAP_MIN_MM", "STROKE_MIN_PT", "CAP_RATIO",
    "COL_SINGLE", "COL_DOUBLE",
    "role", "cycle", "ramp", "tint", "apply", "bars", "points", "cap_mm", "pt_for_cap_mm",
    "reps_as_points", "check_point_spill",
    "exempt", "check_type_floor", "check_stroke_floor", "check_on_page", "check_floors",
    "VERM", "BLUE", "GREEN", "ORANGE", "SKY", "PURPLE", "GREY",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
TOKENS_PATH = os.environ.get("RNA_FIGURE_TOKENS") or os.path.join(_HERE, "_views", "rna_tokens.json")
with open(TOKENS_PATH) as _fh:
    TOK = json.load(_fh)

_SEM, _STR = TOK["palette"]["semantic"], TOK["palette"]["structural"]
_A = list(TOK["layers"]["A"]["tokens"])
_B = list(TOK["layers"]["B"]["tokens"])

FOCAL, REFERENCE, MATCH = _SEM["focal"], _SEM["reference"], _SEM["match"]
STRATUM_A, STRATUM_B, CONTROL = _SEM["stratum_a"], _SEM["stratum_b"], _SEM["control"]
NEUTRAL = _STR["neutral"]
INK, HAIRLINE, SUBTLE, MUTE, WASH, PAPER = (_STR["ink"], _STR["hairline"], _STR["subtle"],
                                            _STR["mute"], _STR["wash"], _STR["paper"])
FOCAL_TINT, REFERENCE_TINT, MATCH_TINT = _STR["focal_tint"], _STR["reference_tint"], _STR["match_tint"]

ROLES: Dict[str, str] = {**{k: _SEM[k] for k in _A}, "neutral": NEUTRAL}

VERM, BLUE, GREEN, ORANGE, SKY, PURPLE, GREY = FOCAL, REFERENCE, MATCH, STRATUM_A, STRATUM_B, CONTROL, NEUTRAL

TYPE = dict(TOK["typography"]["manuscript_pt"])
SLIDE = dict(TOK["typography"]["slide_pt"])
STROKE = dict(TOK["geometry"]["stroke_pt"])
CAP_MIN_MM = TOK["typography"]["cap_min_mm"]
STROKE_MIN_PT = TOK["geometry"]["stroke_min_pt"]
CAP_RATIO = TOK["typography"]["cap_height_ratio"]
COL_SINGLE = TOK["geometry"]["column_in"]["single"]
COL_DOUBLE = TOK["geometry"]["column_in"]["double"]

_MM_PER_PT = 25.4 / 72.0
_TINTS = {"focal": FOCAL_TINT, "reference": REFERENCE_TINT, "match": MATCH_TINT}


def role(name: str) -> str:
    """Hex for a layer-A role (or 'neutral'). REFUSES a layer-B token -- a molecular identity
    must not be handed to a data series -- and refuses an unknown name instead of defaulting."""
    if name in _B:
        raise ValueError(f"{name!r} is a layer-B molecular identity, not an argument role; "
                         f"data series take one of {sorted(ROLES)}")
    try:
        return ROLES[name]
    except KeyError:
        raise ValueError(f"unknown role {name!r}; roles are {sorted(ROLES)}") from None


def cycle(n: Optional[int] = None) -> List[str]:
    ""
    order = TOK["layers"]["A"].get("cycle") or ["reference", "focal", "stratum_a", "stratum_b", "control", "match"]
    if n is not None:
        if n > 4:
            raise ValueError(f"{n} series exceed the 4-accent budget: use two roles + neutral/mute, facet, "
                             "or a sequential ramp of one role (P.ramp)")
        return [ROLES[k] for k in order[:n]]
    return [ROLES[k] for k in order]


def ramp(name: str = "focal", n: int = 256):
    ""
    from matplotlib.colors import LinearSegmentedColormap
    ramps = TOK.get("ramps") or {}
    if name == "signed":
        stops = ramps.get("signed") or ["reference", "paper", "focal"]
    else:
        stops = ramps.get(name) or ["paper", name]
    hexes = []
    for tokn in stops:
        if tokn in _B:
            raise ValueError(f"{tokn!r} is a layer-B identity; a heatmap ramp is a layer-A role")
        hexes.append(PAPER if tokn == "paper" else role(tokn))
    return LinearSegmentedColormap.from_list(f"rf_{name}", hexes, N=n)


_TINTLESS = ("stratum_a", "stratum_b", "control", "neutral")


def tint(name: str) -> str:
    """The band/fill tint under a role's line: the tint token for focal / reference / match,
    `wash` for the four roles that have none (stratum_a, stratum_b, control, neutral). REFUSES a
    layer-B token or an unknown name -- a typo that silently became grey would be a gate that
    lies (audit 879, F3.7)."""
    if name in _B:
        raise ValueError(f"{name!r} is a layer-B molecular identity; bands belong to data series")
    if name in _TINTS:
        return _TINTS[name]
    if name in _TINTLESS:
        return WASH
    raise ValueError(f"unknown role {name!r}; roles are {sorted(ROLES)}")


def apply(matplotlib_module, *, slide: bool = False, dpi: int = 300) -> dict:
    """The default rcParams: family (Arial first, for the bold-weight reason in tokens.json),
    floored type sizes, stroke tiers, out-ticks, no top/right spines, exact-geometry saving
    (`savefig.bbox=None` -- 'tight' silently changes an 89 / 183 mm figure's width).

    Sets `font.sans-serif` to the token family. MEASURED on matplotlib 3.9.4: the rcParams
    family list does NOT fall back per glyph, so a Text carrying a glyph Arial lacks (★ ⊘ ▮ ⇢
    ✱ ∼ →) must pass its own `fontfamily=[family[0], "DejaVu Sans"]`; `pileup` does.
    """
    T = SLIDE if slide else TYPE
    fam = TOK["typography"]["family"]
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": fam,
        "font.size": T["in_figure"],
        "axes.labelsize": T["axis_label"],
        "axes.titlesize": T["axis_label"],
        "xtick.labelsize": T["tick_label"],
        "ytick.labelsize": T["tick_label"],
        "legend.fontsize": T["annotation"],
        "axes.linewidth": STROKE["hairline"],
        "xtick.major.width": STROKE["hairline"],
        "ytick.major.width": STROKE["hairline"],
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": HAIRLINE,
        "xtick.color": HAIRLINE,
        "ytick.color": HAIRLINE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": STROKE["secondary"],
        "lines.solid_capstyle": "round",
        "patch.linewidth": STROKE["hairline"],
        "mathtext.fontset": "custom",
        "mathtext.rm": fam[0],
        "mathtext.it": fam[0] + ":italic",
        "mathtext.bf": fam[0] + ":bold",
        "savefig.bbox": None,
        "savefig.dpi": dpi,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    matplotlib_module.rcParams.update(rc)
    return rc


_HATCH = {"stratum_b": "////", "control": None}


def bars(ax, x, heights, *, roles: Sequence[str], width: float = 0.8, labels: Optional[Sequence[str]] = None,
         kind: str = "fill", edge: bool = True, alpha: float = 1.0, **kw):
    """Grouped bars whose colours are ROLES, with the declared greyscale / CVD cues applied:

    `heights` is one sequence per role (grouped) or a single sequence when `roles` has one
    entry. Returns the BarContainers.
    
    """
    roles = list(roles)
    from matplotlib.colors import to_hex
    strata_hex = {STRATUM_A.upper(), STRATUM_B.upper()}

    def _face(p):
        fc = p.get_facecolor()
        return None if (len(fc) >= 4 and fc[3] == 0) else to_hex(fc).upper()
    strata_on_axes = any(_face(p) in strata_hex for p in ax.patches)
    focal_on_axes = any(_face(p) == FOCAL.upper() for p in ax.patches)
    strata_now = any(r.startswith("stratum") for r in roles)
    msg = ("FOCAL beside stratum fills must be an annotation (arrow, outline, marker, text), never a "
           "bar fill -- 806 rule of use; the declared cue for the vermillion/amber pair. Pass "
           "kind='annotation' to draw FOCAL as an outline.")
    if "focal" in roles and (strata_now or strata_on_axes) and kind != "annotation":
        raise ValueError(msg)
    if strata_now and focal_on_axes:
        raise ValueError(msg)
    x = np.asarray(x, dtype=float)
    hs = [np.asarray(heights)] if len(roles) == 1 and np.ndim(heights) == 1 else [np.asarray(h) for h in heights]
    if len(hs) != len(roles):
        raise ValueError(f"{len(hs)} height series for {len(roles)} roles")
    n = len(roles)
    w = width / n
    out = []
    for i, (r, h) in enumerate(zip(roles, hs)):
        col = role(r)
        offs = (i - (n - 1) / 2) * w
        style = dict(color=col, edgecolor=INK if (edge and r != "neutral") else "none",
                     linewidth=STROKE["hairline"] if edge else 0, alpha=alpha, hatch=_HATCH.get(r))
        if r == "neutral":
            style["alpha"] = min(alpha, 0.85)
        if r == "focal" and kind == "annotation":
            style.update(color="none", edgecolor=col, linewidth=STROKE["primary"], hatch=None)
        if _HATCH.get(r):
            style["edgecolor"] = INK
        style.update(kw)
        out.append(ax.bar(x + offs, h, w, label=(labels[i] if labels else None), **style))
    return out


_MARKER = {"focal": "D", "reference": "o", "stratum_a": "o", "stratum_b": "o", "control": "s",
           "match": "^", "neutral": "."}


def points(ax, x, y, *, role_name: str, s: float = 14, label: Optional[str] = None, **kw):
    """Scatter points whose colour is a role and whose MARKER is the declared second channel:
    focal a DIAMOND (so it is never the same shape as a stratum point -- the vermillion/amber
    pair's cue holds for points as well as bars; audit 879, F4.1), stratum_b an OPEN circle (its
    greyscale twin stratum_a is filled), control a square, match a triangle, neutral a small dot."""
    col = role(role_name)
    style = dict(s=s, label=label, marker=_MARKER.get(role_name, "o"), linewidths=STROKE["hairline"])
    if role_name == "stratum_b":
        style.update(facecolors="none", edgecolors=col)
    else:
        style.update(c=col, edgecolors=INK if role_name != "neutral" else "none")
    style.update(kw)
    return ax.scatter(x, y, **style)


def cap_mm(pt: float) -> float:
    """On-page cap height in mm for a font size in points."""
    return pt * CAP_RATIO * _MM_PER_PT


def pt_for_cap_mm(mm: Optional[float] = None) -> float:
    """Font size in points whose cap height is `mm` (default: the default floor)."""
    return (CAP_MIN_MM if mm is None else mm) / (CAP_RATIO * _MM_PER_PT)


_EXEMPT_GID = "the RNA figure standard-floor-exempt"
_MAG_EXEMPT_GID = "magestic-floor-exempt"


def exempt(artist):
    """Mark an artist as deliberately below a floor so the checkers skip it. Say why in a
    comment at the call site. `fig.texts` (footer, stamp, legend) is auto-exempt."""
    artist.set_gid(_EXEMPT_GID)
    return artist


def _is_exempt(artist) -> bool:
    return artist.get_gid() in (_EXEMPT_GID, _MAG_EXEMPT_GID)


def check_type_floor(fig, *, floor_mm: Optional[float] = None, verbose: bool = True):
    """Every Text in `fig` whose cap height is below the floor. Returns [(where, pt, mm)]."""
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
                bad.append((t.get_text().replace("\n", " ")[:40], round(pt, 2), round(cap_mm(pt), 2)))
        if not ax.axison:
            continue
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            if not t.get_text().strip() or _is_exempt(t) or not t.get_visible():
                continue
            if cap_mm(t.get_fontsize()) < floor - tol:
                bad.append(("tick:" + t.get_text()[:20], round(t.get_fontsize(), 2),
                            round(cap_mm(t.get_fontsize()), 2)))
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                if t.get_text().strip() and not _is_exempt(t) and cap_mm(t.get_fontsize()) < floor - tol:
                    bad.append(("legend:" + t.get_text()[:20], round(t.get_fontsize(), 2),
                                round(cap_mm(t.get_fontsize()), 2)))
    if verbose:
        print("the RNA figure standard: type floor %.2f mm cap (= %.1f pt) -- %s"
              % (floor, pt_for_cap_mm(floor), "OK" if not bad else "%d BELOW" % len(bad)))
        for w, pt, mm in bad[:12]:
            print("   %-42s %.1f pt  cap %.2f mm" % (w, pt, mm))
    return bad


def check_stroke_floor(fig, *, floor_pt: Optional[float] = None, verbose: bool = True):
    """Every stroked artist below the stroke floor. Unstroked patches (edge alpha 0) are skipped."""
    floor = STROKE_MIN_PT if floor_pt is None else floor_pt
    bad = []
    for ax in fig.axes:
        for a in list(ax.patches) + list(ax.lines) + list(ax.collections):
            if _is_exempt(a) or not a.get_visible():
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
        print("the RNA figure standard: stroke floor %.2f pt -- %s" % (floor, "OK" if not bad else "%d BELOW" % len(bad)))
        for k, lw in bad[:12]:
            print("   %-42s %.2f pt" % (k, lw))
    return bad


def check_on_page(fig, *, tol_px: float = 0.5, verbose: bool = True) -> List[tuple]:
    ""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    def _drawn_tick_labels(axis):
        lo, hi = sorted(axis.get_view_interval())
        out = []
        for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
            if lo <= tick.get_loc() <= hi:
                out += [lab for lab in (tick.label1, tick.label2) if lab.get_visible()]
        return out
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        texts += _drawn_tick_labels(ax.xaxis) + _drawn_tick_labels(ax.yaxis)
        leg = ax.get_legend()
        if leg is not None:
            texts += list(leg.get_texts())
    bad = []
    for t in texts:
        if t is None or not t.get_text().strip() or not t.get_visible():
            continue
        bb = t.get_window_extent(r)
        over = {"left": -bb.x0, "right": bb.x1 - W, "bottom": -bb.y0, "top": bb.y1 - H}
        side, o = max(over.items(), key=lambda kv: kv[1])
        if o > tol_px:
            bad.append((t.get_text().replace("\n", " ")[:40], side, round(float(o), 1)))
    if verbose:
        print("the RNA figure standard: on-page text -- %s" % ("OK" if not bad else "%d OFF the page" % len(bad)))
        for w, side, o in bad[:12]:
            print("   %-42s %s  %.1f px past the %s edge" % (w, side, o, side))
    return bad


def check_floors(fig, **kw) -> bool:
    """Both floors AND the page: type >= the cap floor, stroke >= the stroke floor, every text on
    the page. True when the figure is clean."""
    return not (check_type_floor(fig, **kw) or check_stroke_floor(fig, **kw) or check_on_page(fig, **kw))


_MIN_SPREAD = 0.15


def reps_as_points(ax, rows, *, letters: Sequence[str], groups: Sequence[str], roles: Dict[str, str],
                   value: str = "share", ylabel: Optional[str] = None, bar_w: float = 0.34, gap: float = 0.06,
                   key: bool = True, key_labels: Optional[Dict[str, str]] = None,
                   s_max: float = 14.0, s_min: float = 4.0) -> List[dict]:
    """    * the point CENTRES spread over at most `tokens.geometry.points.spill_reach` (0.42) x
      `bar_w` to either side of the bar centre (a sorted swarm), and less when the marker's
      EDGE would otherwise cross the bar's edge -- the reach adapts to the marker;
    * the marker shrinks with n (`s_max` pt^2 up to four points, then `s_max * 4 / n`, never
      below `s_min`), and further when the bar cannot hold it with a 30 % spread;
    * a bar that cannot hold even the `s_min` marker with that spread is REFUSED (ValueError):
      widen the bar or the axes, or facet the groups -- never let it overflow;
    * the group key is written ABOVE the axes, never on the data.

    `check_point_spill(result)` re-measures the RENDERED artists, and figproof.qa is run with
    include_data=True on a points panel so text-on-points is a gate, not an advisory.

    Returns one dict per bar: {"group", "letter", "n", "mean", "sd", "s_pt", "reach", "bar",
    "points"} -- the artists included, so a checker can measure them.
    
    """
    from matplotlib.markers import MarkerStyle
    reach_frac = float(TOK["geometry"].get("points", {}).get("spill_reach", 0.42))
    G = list(groups)
    ng = len(G)
    letters = list(letters)
    xs = np.arange(len(letters), dtype=float)
    fig = ax.figure
    ax.set_xlim(-0.5 - bar_w, len(letters) - 0.5 + bar_w)
    px_per_unit = ax.bbox.width / (ax.get_xlim()[1] - ax.get_xlim()[0])
    pt_per_unit = px_per_unit * 72.0 / fig.dpi
    lw = STROKE["hairline"]
    out: List[dict] = []
    ymax = 0.0
    for gi, g in enumerate(G):
        off = (gi - (ng - 1) / 2) * (bar_w + gap)
        col = role(roles[g])
        marker = _MARKER.get(roles[g], "o")
        ms = MarkerStyle(marker)
        ext = ms.get_path().transformed(ms.get_transform()).get_extents()
        half_unit = max(abs(ext.x0), abs(ext.x1))

        def half_units(s):
            return (half_unit * np.sqrt(s) + lw / 2) / pt_per_unit

        def s_for_half(h):
            return max(0.0, (h * pt_per_unit - lw / 2) / half_unit) ** 2

        for li, L in enumerate(letters):
            vals = np.array([r[value] for r in rows
                             if r.get("group") == g and r.get("letter") == L and r.get(value) is not None],
                            dtype=float)
            n = len(vals)
            if n == 0:
                continue
            m = float(vals.mean())
            bar = ax.bar(xs[li] + off, m, width=bar_w, color=col, alpha=0.35, edgecolor="none", zorder=2)[0]
            sd = None
            if n >= 3:
                sd = float(vals.std(ddof=1))
                ax.errorbar(xs[li] + off, m, yerr=sd, fmt="none", ecolor=col, elinewidth=lw, capsize=2, zorder=3)
                ymax = max(ymax, m + sd)
            min_reach = _MIN_SPREAD * bar_w if n > 1 else 0.0
            room = bar_w / 2 - min_reach
            if s_for_half(room) < s_min:
                raise ValueError(
                    f"reps_as_points: {n} points cannot stay inside a bar {bar_w} wide at letter {L!r} ({g}): "
                    f"the bar is {bar_w * pt_per_unit:.1f} pt wide and a {s_min} pt^2 marker with a "
                    f"{2 * _MIN_SPREAD:.0%} swarm spread needs {2 * (half_units(s_min) + min_reach) * pt_per_unit:.1f} pt "
                    f"-- widen the bar or the axes, or facet the groups")
            s_pt = s_max if n <= 4 else max(s_min, s_max * 4.0 / n)
            s_pt = float(min(s_pt, s_for_half(room)))
            reach = min(reach_frac * bar_w, bar_w / 2 - half_units(s_pt)) if n > 1 else 0.0
            order = np.argsort(vals)
            jit = np.zeros(n)
            jit[order] = np.linspace(-reach, reach, n) if n > 1 else 0.0
            pc = points(ax, xs[li] + off + jit, vals, role_name=roles[g], s=s_pt, edgecolors=PAPER,
                        linewidths=lw, zorder=4)
            ymax = max(ymax, float(vals.max()))
            out.append({"group": g, "letter": L, "n": n, "mean": m, "sd": sd, "s_pt": s_pt,
                        "reach": float(reach), "bar": bar, "points": pc})
    ax.set_xticks(xs)
    ax.set_xticklabels(letters, fontweight="bold", fontsize=TYPE["in_figure"])
    ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=TYPE["axis_label"])
    if key:
        renderer = fig.canvas.get_renderer()
        x = 0.0
        for g in G:
            t = ax.text(x, 1.02, (key_labels or {}).get(g, g), transform=ax.transAxes, ha="left", va="bottom",
                        color=role(roles[g]), fontsize=TYPE["annotation"], fontweight="bold")
            x += 0.03 + t.get_window_extent(renderer).width / ax.bbox.width
    return out


def _swarm_extent_px(pc, renderer):
    """(x0, x1) in display px of a scatter swarm: the offsets through the axes transform, plus
    the marker's half-width (its path extents scaled by sqrt(size), plus half the edge). Measured
    by hand because `PathCollection.get_window_extent` returns an empty bbox for scatter markers."""
    ax = pc.axes
    fig = ax.figure
    xs = ax.transData.transform(pc.get_offsets())[:, 0]
    ext = pc.get_paths()[0].get_extents()
    half_unit = max(abs(ext.x0), abs(ext.x1))
    s = float(np.atleast_1d(pc.get_sizes())[0])
    lw = float(np.atleast_1d(pc.get_linewidths())[0])
    half_px = (half_unit * np.sqrt(s) + lw / 2) * fig.dpi / 72.0
    return float(xs.min() - half_px), float(xs.max() + half_px)


def check_point_spill(result: Sequence[dict], *, tol_px: float = 0.5, verbose: bool = True) -> List[tuple]:
    """Re-measure a `reps_as_points` result on the RENDERED page: every swarm's pixel extent
    (marker edges included) must lie inside its own bar's. Returns [(group, letter,
    overshoot_px)] -- empty when clean. A checker that cannot fail is not a checker:
    tests/test_panels.py moves a swarm past its bar and asserts it is flagged."""
    bad = []
    for e in result:
        fig = e["bar"].figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        p0, p1 = _swarm_extent_px(e["points"], renderer)
        bb = e["bar"].get_window_extent(renderer)
        over = max(bb.x0 - p0, p1 - bb.x1)
        if over > tol_px:
            bad.append((e["group"], e["letter"], round(float(over), 2)))
    if verbose:
        print("the RNA figure standard: point spill -- %s" % ("OK" if not bad else "%d bar(s) overrun" % len(bad)))
        for g, L, o in bad[:12]:
            print("   %-20s %s  %.2f px past the bar" % (g, L, o))
    return bad
