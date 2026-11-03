"""figproof.gates — every check `finish()` runs, each returning a list of problems (empty = pass).

Each gate has a planted failing test in tests/test_figproof.py; a gate that cannot fail is not a gate.
"""
from __future__ import annotations

import contextlib
import io
import math

import matplotlib.colors as mcolors
from matplotlib.collections import Collection
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

from . import tokens as T

EXEMPT_GIDS = {"figproof-exempt", "the RNA figure standard-floor-exempt", "magestic-floor-exempt"}
DATA_TEXT_GID = "figproof-data-text"
NEUTRAL_CHROMA = 6.0


def to_hex(c):
    """'#RRGGBB' for any matplotlib colour, or None when it is fully transparent or unset."""
    if c is None:
        return None
    try:
        r, g, b, a = mcolors.to_rgba(c)
    except (ValueError, TypeError):
        return None
    if a == 0:
        return None
    return "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))


def _lab(hx):
    def lin(v):
        v = v / 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(int(hx[i:i + 2], 16)) for i in (1, 3, 5))
    X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116
    fx, fy, fz = f(X), f(Y), f(Z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def chroma(hx):
    _, a, b = _lab(hx)
    return math.hypot(a, b)


def is_exempt(artist):
    return artist.get_gid() in EXEMPT_GIDS


def _texts(fig):
    """Every visible, non-empty Text the figure will draw (axes texts, labels, in-view tick labels, legends, figure texts)."""
    out = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        cand = list(ax.texts) + [ax.title]
        if ax.axison:
            cand += [ax.xaxis.label, ax.yaxis.label]
            for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
                lo, hi = min(lim), max(lim)
                for tick in axis.get_major_ticks():
                    if lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                        cand += [tick.label1, tick.label2]
        leg = ax.get_legend()
        if leg is not None:
            cand += list(leg.get_texts())
        out += cand
    out += list(fig.texts)
    for leg in fig.legends:
        out += list(leg.get_texts())
    return [t for t in out if isinstance(t, Text) and t.get_visible() and t.get_text().strip()]


def floors_standard(fig, panels_mod):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = panels_mod.check_floors(fig)
    return [] if ok else [ln.strip() for ln in buf.getvalue().splitlines() if ln.strip() and "OK" not in ln]


def type_band_journal(fig, letters, lo=5.0, hi=7.0, letter_pt=8.0, stroke_min_pt=0.6, panels_mod=None):
    """Journal type band: text 5 to 7 pt; panel letters 8 pt bold, upright, lowercase."""
    bad, letter_ids = [], {id(t) for t in letters}
    for t in _texts(fig):
        if is_exempt(t):
            continue
        pt = t.get_fontsize()
        s = t.get_text().replace("\n", " ")[:40]
        if id(t) in letter_ids:
            if abs(pt - letter_pt) > 1e-6 or t.get_fontweight() not in ("bold", 700) or not s.islower():
                bad.append(f"panel letter {s!r} is {pt} pt {t.get_fontweight()}; journal mode wants 8 pt bold lowercase")
        elif not (lo - 1e-6 <= pt <= hi + 1e-6):
            bad.append(f"{s!r} is {pt:g} pt; journal text must be {lo:g} to {hi:g} pt")
    if panels_mod is not None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            strokes = panels_mod.check_stroke_floor(fig, floor_pt=stroke_min_pt)
        bad += [f"stroke below {stroke_min_pt} pt: {s}" for s in (strokes or [])][:10]
    return bad


def overlaps(fig, figqa_mod):
    ov = figqa_mod.find_overlaps(fig, include_data=False)
    return [f"{o.as_dict().get('a_label')} overlaps {o.as_dict().get('b_label')}" for o in ov]


def containment(fig, figqa_mod):
    r = figqa_mod.check_canvas_containment(fig)
    return [] if r.get("status") == "PASS" else [r.get("summary", "an artist lies outside the canvas")]


def axes_layer(ax):
    """The colour layer an axes speaks: an explicit `figproof.layer(ax, ...)`; else `role` for an axis-on data panel; else None (an
    untagged axis-off axes, where a hex takes its single token meaning and a hex with two must be resolved by a tag)."""
    return getattr(ax, "_hf_layer", None) or ("role" if ax.axison else None)


def _artist_colours(artist):
    cols = []
    if isinstance(artist, Text):
        cols.append(artist.get_color())
    elif isinstance(artist, Line2D):
        if artist.get_linestyle() not in ("None", "none", "", " ") and artist.get_linewidth() > 0:
            cols.append(artist.get_color())
        if artist.get_marker() not in (None, "None", "none", "", " "):
            cols += [artist.get_markerfacecolor(), artist.get_markeredgecolor()]
    elif isinstance(artist, Patch):
        cols.append(artist.get_facecolor() if artist.get_fill() else None)
        if artist.get_linewidth() > 0 or artist.get_hatch():
            cols.append(artist.get_edgecolor())
    elif isinstance(artist, Collection):
        cols += list(artist.get_facecolors())
        lws = artist.get_linewidths()
        if len(lws) and max(lws) > 0:
            cols += list(artist.get_edgecolors())
    return [h for h in (to_hex(c) for c in cols) if h]


def drawn_colours(fig):
    """{HEX: [(axes label, axes layer), ...]} from artist properties (not pixels). Images are not read (blind spot)."""
    out = {}
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        lay = axes_layer(ax)
        label = getattr(ax, "_hf_label", None) or f"axes {i}"
        arts = list(ax.patches) + list(ax.lines) + list(ax.collections) + list(ax.texts)
        for a in arts:
            if isinstance(a, AxesImage) or not a.get_visible():
                continue
            for hx in _artist_colours(a):
                out.setdefault(hx, []).append((label, lay))
    return out


def hex_meanings(fig):
    """    In an axes of layer L, a hex means its token in L when it has one ("identity.barcode" in a schematic, "role.stratum_a" in a bar
    chart); a hex that is a token only in another colour layer means "foreign:<that token>" there (an RNA role colour hard-coded into
    a MAGESTIC alignment panel); a colour that is no token has no meaning and is ignored. Two meanings for one hex in one figure fail.
    Marks and roles share panels by design (the RNA figure standard's cross-layer sets), which this rule allows: a mark hex has one meaning wherever
    it is drawn.
    """
    bad = []
    drawn = drawn_colours(fig)
    for hx, where in drawn.items():
        toks = [(lay, tok) for lay, tok in T.meanings_of(hx) if lay in T.COLOUR_LAYERS]
        if not toks:
            continue
        meanings = {}
        for label, lay in where:
            if lay is None:
                if len(toks) == 1:
                    m = f"{toks[0][0]}.{toks[0][1]}"
                else:
                    bad.append(f"{hx} is {' and '.join(f'{l}.{t}' for l, t in toks)}; {label} is an untagged axis-off panel, so tag it "
                               "with figproof.layer(ax, 'identity' or 'role')")
                    continue
            else:
                own = [f"{l}.{t}" for l, t in toks if l == lay]
                m = own[0] if own else "foreign:" + "/".join(f"{l}.{t}" for l, t in toks)
            meanings.setdefault(m, set()).add(label)
        if len(meanings) > 1:
            desc = "; ".join(f"{m} in {', '.join(sorted(labels))}" for m, labels in sorted(meanings.items()))
            bad.append(f"{hx} carries {len(meanings)} meanings in one figure: {desc}")
    present = set(drawn)
    for a, b, ra, rb, de in T.never_together():
        if a in present and b in present:
            bad.append(f"never-together pair in one figure: {ra} {a} and {rb} {b} (worst dE {de})")
    return bad


def strokes_at_most(fig, cap_pt):
    ""
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    bad = []
    for a in fig.findobj():
        if not a.get_visible() or is_exempt(a):
            continue
        widths = []
        if isinstance(a, Line2D):
            if a.get_linestyle() not in ("None", "none", "", " "):
                widths.append(("line", a.get_linewidth()))
            if a.get_marker() not in (None, "None", "none", "", " "):
                widths.append(("marker edge", a.get_markeredgewidth()))
        elif isinstance(a, Patch):
            ec = a.get_edgecolor()
            if ec is not None and len(ec) == 4 and ec[3] > 0:
                widths.append(("edge", a.get_linewidth()))
        elif isinstance(a, Collection):
            ecs = a.get_edgecolor()
            if len(ecs) and max(c[3] for c in ecs) > 0:
                widths += [("edge", w) for w in a.get_linewidths()]
        for what, w in widths:
            if w and w > cap_pt + 1e-6:
                bad.append(f"{type(a).__name__} {what} is {w:g} pt; Extended Data strokes must be at most {cap_pt:g} pt")
    return bad[:12]


def slide_panels(fig):
    """    Three things fail a slide render:
      * a panel letter drawn anywhere on it, which points into a legend the audience cannot see;
      * a data panel that carries no name (a column header and row label from `figproof.slides.grid`, or its own title);
      * a grid with no axis labels at all. Labels are checked per FIGURE rather than per axes, because a shared grid deliberately
        labels only its outer panels and requiring one on every panel would push a caller into repeating the same words four times.
    
    """
    bad, named = [], 0
    letters = [t for t in fig.texts
               if len(t.get_text().strip()) == 1 and t.get_text().strip().isalpha() and t.get_text().strip().islower()
               and t.get_fontweight() in ("bold", 700) and not is_exempt(t)]
    if letters:
        shown = ", ".join(sorted(t.get_text().strip() for t in letters))
        bad.append(f"panel letter(s) {shown} on a slide render; a slide's legend goes to the notes, so a letter points at something the "
                   "audience cannot read. Name the panels instead, with figproof.slides.grid")
    data_axes = [a for a in fig.axes if a.get_visible() and a.axison and not getattr(a, "_hf_schematic", False)]
    for ax in data_axes:
        if getattr(ax, "_hf_panel_name", "") or ax.get_title():
            named += 1
        else:
            bad.append("a slide panel says nothing on the canvas about what it shows; give it a column header and a row label "
                       "(figproof.slides.grid) or a title")
    if data_axes and not any(ax.get_xlabel() for ax in data_axes):
        bad.append("no x axis label anywhere on the slide; name the quantity")
    if data_axes and not any(ax.get_ylabel() for ax in data_axes):
        bad.append("no y axis label anywhere on the slide; name the quantity")
    return bad[:8]


def declarable_warnings(fig):
    present = set(drawn_colours(fig))
    return [f"{ra} {a} and {rb} {b} share a figure (worst dE {de}); declare the form cue that separates them"
            for a, b, ra, rb, de in T.declarable() if a in present and b in present]


def coloured_text(fig):
    ""
    bad = []
    neutral = {h.upper() for h in T.layer("neutral").values()}
    for t in _texts(fig):
        if is_exempt(t) or t.get_gid() == DATA_TEXT_GID:
            continue
        hx = to_hex(t.get_color())
        if hx and hx not in neutral and chroma(hx) >= NEUTRAL_CHROMA:
            bad.append(f"coloured text {t.get_text()[:30]!r} in {hx}; set it in ink beside a key marker, or figproof.exempt() it")
    return bad


def abstract_one_panel(fig, allowed=1):
    """Cell Press graphical abstract: "the abstract should consist of one single panel". A HARD gate.

    Counts visible axes that carry drawn content. A graphical abstract is typically ONE axis-off canvas; a second visible axes means
    the author has laid out panels, which is the thing Cell's guide rules out (its own worked example merges a multi-panel draft into
    one). An empty helper axes (nothing drawn) is not a panel and is ignored.
    """
    drawn = [a for a in fig.axes if a.get_visible() and (a.lines or a.patches or a.collections or a.images or a.texts)]
    if len(drawn) > allowed:
        return [f"abstract has {len(drawn)} panels; Cell requires {allowed} single panel"]
    return []


def abstract_text_budget(fig, budget_words):
    """Words on a graphical abstract. ADVISORY ONLY — returns warnings, never problems."""
    words = sum(len(t.get_text().split()) for t in _texts(fig))
    if budget_words is None:
        return [f"abstract carries {words} words (no default budget set yet; Cell says only 'use text sparingly')"]
    if words > budget_words:
        return [f"abstract carries {words} words, over the default budget of {budget_words}"]
    return []


ANNOTATION_MAX_WORDS = 4

FINITE_VERBS = frozenset("""
is are was were am be been being has have had does do did
can cannot could will would shall should may might must
isn't aren't wasn't weren't doesn't don't didn't can't won't
shows reveals indicates suggests demonstrates confirms explains causes drives requires increases decreases reduces
loses keeps binds cleaves degrades recruits produces remains becomes depends
""".split())


def _words(s):
    """Tokens that carry a letter: 'n = 24' is one word, 'internal 5′ ends' three, a 20-nt sequence one."""
    return [w.strip(".,;:!?()[]{}\"'‘’“”") for w in s.split() if any(ch.isalpha() for ch in w)]


def _has_finite_verb(s):
    ws = [w.lower() for w in _words(s)]
    return len(ws) >= 2 and any(w in FINITE_VERBS for w in ws)


def caption_annotations(fig, max_words=ANNOTATION_MAX_WORDS):
    """§2b / §2g: free text INSIDE a panel names a drawn thing. Longer than `max_words`, or holding a finite verb, it is legend text.

    Checks `ax.texts` only (annotations included). Never checks the legend block, the footer, the stamp, axis labels, tick labels, slide
    headers or row labels (all of those are figure texts or axis labels), exempt text, or text marked as data (`figproof.data_text`).
    """
    out = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for t in ax.texts:
            s = t.get_text().strip()
            if not s or not t.get_visible() or is_exempt(t) or t.get_gid() == DATA_TEXT_GID:
                continue
            n = len(_words(s))
            if n > max_words:
                out.append(f"caption-like annotation ({n} words) {s[:48]!r}: a label names a thing in 2-{max_words} words; move the "
                           "rest to the legend (the figure standard, §2g)")
            elif _has_finite_verb(s):
                out.append(f"annotation {s[:48]!r} is a sentence (finite verb); name the thing and move the claim to the legend (§2b)")
    return out[:12]


def panel_headings(fig, slide=False):
    """Headings, by mode."""
    out = []
    if not slide:
        for ax in fig.axes:
            s = ax.get_title().strip() if ax.get_visible() else ""
            if s and not is_exempt(ax.title):
                out.append(f"axes title {s[:48]!r}: no per-axes titles outside a slide (the figure standard); fold it into the x-axis "
                           "label and the legend's bold lead")
        return out[:12]
    heads = [ax.title for ax in fig.axes if ax.get_visible() and ax.get_title().strip()]
    heads += [t for t in fig.texts if getattr(t, "_hf_role", None) == "header"]
    for t in heads:
        s = t.get_text().strip().replace("\n", " ")
        if is_exempt(t):
            continue
        if _has_finite_verb(s) or s.endswith((".", "?", "!")):
            out.append(f"slide panel header {s[:48]!r} reads as a sentence; a header names what the panel shows, and the claim "
                       "belongs in the slide title (§2i)")
    return out[:12]


def cards(fig, tol_in=0.01):
    """Slide geometry: a card never overlaps a sibling or any other filled box, and its own text never leaves it.

    Three rules, over the cards registered by `figproof.slides.card`:

      * a card must lie on the page, clear of the foot band a 16:9 slide keeps for its running label;
      * a card must not intersect another card, nor any other visible filled rectangle on the figure. **Nesting is
        allowed**: if one box contains the other the author meant it (an accent rule inside its card, a tint behind a
        group), and only a partial intersection is a collision;
      * every text a card drew must lie inside that card.

    A figure that registers no cards is not checked at all, so this cannot touch a figure that has not adopted the API.
    Known blind spot: boxes nobody registered are checked only against registered cards, never against each other.
    
    """
    recs = getattr(fig, "_hf_cards", [])
    if not recs:
        return []
    from .slides import CARD_FOOT_BAND_IN, _bbox_top_in
    r = fig.canvas.get_renderer()
    W, H = (float(v) for v in fig.get_size_inches())
    bad = []

    def label(rec):
        return (rec.get("label") or "a card")

    for rec in recs:
        x, y, w, h = rec["rect"]
        if x < -tol_in or y < -tol_in or x + w > W + tol_in or y + h > H + tol_in:
            bad.append(f"{label(rec)!r} at ({x:g}, {y:g}) {w:g}x{h:g} in runs off the {W:g}x{H:g} in page")
        elif y + h > H - CARD_FOOT_BAND_IN + tol_in:
            bad.append(f"{label(rec)!r} ends {y + h:.2f} in down the page, inside the {CARD_FOOT_BAND_IN:g} in foot band "
                       f"the slide keeps for its running label")

    def overlap(a, b):
        ax0, ay0, aw, ah = a
        bx0, by0, bw, bh = b
        ox = min(ax0 + aw, bx0 + bw) - max(ax0, bx0)
        oy = min(ay0 + ah, by0 + bh) - max(ay0, by0)
        if ox <= tol_in or oy <= tol_in:
            return 0.0
        contains = ((ax0 <= bx0 + tol_in and ay0 <= by0 + tol_in and ax0 + aw >= bx0 + bw - tol_in
                     and ay0 + ah >= by0 + bh - tol_in)
                    or (bx0 <= ax0 + tol_in and by0 <= ay0 + tol_in and bx0 + bw >= ax0 + aw - tol_in
                        and by0 + bh >= ay0 + ah - tol_in))
        return 0.0 if contains else min(ox, oy)

    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            d = overlap(recs[i]["rect"], recs[j]["rect"])
            if d:
                bad.append(f"{label(recs[i])!r} and {label(recs[j])!r} overlap by {d:.2f} in; a card never grows into a "
                           f"sibling — place them with figproof.slides.row() / stack(), which take the row height from "
                           f"the tallest card and return the y the next element starts at")

    own = {id(rec["patch"]) for rec in recs}
    others = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for p in ax.patches:
            if id(p) in own or not p.get_visible() or to_hex(p.get_facecolor()) is None:
                continue
            bb = _bbox_top_in(fig, p.get_window_extent(r))
            others.append((p, (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1])))
    for rec in recs:
        for p, rect in others:
            d = overlap(rec["rect"], rect)
            if d:
                bad.append(f"{label(rec)!r} overlaps a filled {type(p).__name__} by {d:.2f} in without containing it")

    for rec in recs:
        x, y, w, h = rec["rect"]
        for t in rec["texts"]:
            if not t.get_visible():
                continue
            tx0, ty0, tx1, ty1 = _bbox_top_in(fig, t.get_window_extent(r))
            if tx0 < x - tol_in or ty0 < y - tol_in or tx1 > x + w + tol_in or ty1 > y + h + tol_in:
                bad.append(f"{t.get_text()[:34]!r} lies outside {label(rec)!r}")
    return bad[:10]
