"""figproof.slides — the slide grid: column headers, row labels and a schematic of what is measured, in place of panel letters.

Use it inside a slide figure:

    fig = fp.figure("slide", title="Direct RNA keeps the full-length ends that cDNA loses", ...)
    g = slides.grid(fig, (0.6, 0.35, 8.0, 4.0),
                    col_headers=["direct RNA", "PCR-cDNA"],
                    row_labels=["full-length (TSS) ends", "internal 5′ ends: breaks and decay fragments"],
                    row_schematics=[schematics.full_length, schematics.broken_5p])
    g.ax(0, 0).plot(...)
    fp.finish(fig, "NN_slug_fig", outdir)

`grid` reserves the header band and the left gutter, draws the headers, labels and schematics, and hands back the data axes. It never
places the figure: `box` is (x, y, w, h) in inches from the body's top-left, exactly as `fig.fp.axes` takes it.

It also takes a PLAIN matplotlib figure, where `box` measures from the figure's top-left, so a script that predates the toolkit can adopt
the headers, the row labels and the schematics without being rebuilt. What it does not get that way is the gates: only `fp.finish` checks
that the slide carries no letters and that every panel is named. The schematics in `figproof.schematics` take a plain axes too, and can be
used on their own.
"""
from __future__ import annotations

import textwrap

from matplotlib.patches import FancyBboxPatch, Rectangle

from . import tokens as T

HEADER_H_IN = 0.30
GUTTER_W_IN = 1.55
SCHEMATIC_H_IN = 0.40
GAP_IN = (0.45, 0.49)
EM_PER_CHAR = 0.52


class _Place:
    """Where the grid puts things, on either kind of figure."""

    def __init__(self, fig):
        self.fig = fig
        fp = getattr(fig, "fp", None)
        self.fp = fp
        if fp is not None:
            self.W, self.H, self.top, self.mode = fp.W, fp.H, fp.top, fp.mode
        else:
            self.W, self.H = (float(v) for v in fig.get_size_inches())
            self.top, self.mode = 0.0, None

    def axes(self, x, y, w, h, **kw):
        if self.fp is not None:
            return self.fp.axes(x, y, w, h, **kw)
        return self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], **kw)

    def y_frac(self, y_in):
        """Inches down from the grid's origin -> a figure fraction, for a free text label."""
        return 1 - (self.top + y_in) / self.H


class SlideGrid:
    """What `grid()` returns: the data axes, and the names that let the gates check the panel says what it shows."""

    def __init__(self, fig, axes, names, col_headers, row_labels):
        self.fig, self._axes, self.names = fig, axes, names
        self.col_headers, self.row_labels = col_headers, row_labels

    def ax(self, row, col=0):
        return self._axes[row][col]

    @property
    def axes(self):
        return [a for row in self._axes for a in row]


def _wrap(text, width_in, pt):
    n = max(8, int(width_in * 72.0 / (pt * EM_PER_CHAR)))
    return "\n".join(textwrap.wrap(text, n)) or text


def grid(fig, box, *, col_headers=(), row_labels=(), row_schematics=(), gutter_in=GUTTER_W_IN, header_in=HEADER_H_IN,
         schematic_in=SCHEMATIC_H_IN, gap=GAP_IN, sharex=True, sharey=True):
    """A rows x cols grid of data axes inside `box` = (x, y, w, h) inches, with the header band, the row gutter and the schematics
    drawn under the default clearances. Returns a `SlideGrid`.

    `row_schematics` are callables `fn(ax, width_in)` from `figproof.schematics` (or any callable with that signature); pass `None` for a
    row that has no schematic. Column-only layouts pass `row_labels=[""]`.
    
    """
    place = _Place(fig)
    if place.mode is not None and place.mode != "slide":
        raise ValueError(f"slides.grid is for slide mode; this figure is {place.mode!r}. Other modes carry panel letters and a legend.")
    rows = max(1, len(row_labels) or 1)
    cols = max(1, len(col_headers) or 1)
    if row_schematics and len(row_schematics) != rows:
        raise ValueError(f"row_schematics has {len(row_schematics)} entries for {rows} rows")
    from . import panels as P
    ink = T.colour("neutral.ink")
    head_pt, label_pt = P.TYPE["in_figure"], P.TYPE["in_figure"]
    x0, y0, w, h = box
    has_gutter = any(row_labels) or bool(row_schematics)
    gw = gutter_in if has_gutter else 0.0
    hh = header_in if any(col_headers) else 0.0
    gx, gy = gap
    panel_w = (w - gw - gx * (cols - 1)) / cols
    panel_h = (h - hh - gy * (rows - 1)) / rows
    if panel_w <= 0 or panel_h <= 0:
        raise ValueError(f"box {box} leaves no room for a {rows}x{cols} grid after the gutter and header")
    if place.fp is not None and y0 + h > place.fp.body_h + 1e-6:
        raise ValueError(f"the grid's box ends {y0 + h:.2f} in down the body, which is only {place.fp.body_h:.2f} in tall; shorten the "
                         "box or raise body_height, and leave about 0.5 in under the last row for its tick and axis labels")

    axes, names, first = [], {}, None
    for r in range(rows):
        row_axes = []
        py = y0 + hh + r * (panel_h + gy)
        for c in range(cols):
            px = x0 + gw + c * (panel_w + gx)
            kw = {}
            if first is not None:
                if sharex:
                    kw["sharex"] = first
                if sharey:
                    kw["sharey"] = first
            ax = place.axes(px, py, panel_w, panel_h, **kw)
            first = first or ax
            ax._hf_no_letter = True
            name = " / ".join(s for s in ((row_labels[r] if r < len(row_labels) else ""),
                                          (col_headers[c] if c < len(col_headers) else "")) if s)
            ax._hf_panel_name = name
            names[(r, c)] = name
            row_axes.append(ax)
        axes.append(row_axes)

    for c, head in enumerate(col_headers):
        if not head:
            continue
        px = x0 + gw + c * (panel_w + gx)
        t = fig.text((px + panel_w / 2.0) / place.W, place.y_frac(y0 + hh - 0.08), head, ha="center", va="bottom",
                     fontsize=head_pt, fontweight="bold", color=ink)
        t._hf_role = "header"

    for r, label in enumerate(row_labels):
        py = y0 + hh + r * (panel_h + gy)
        fn = row_schematics[r] if r < len(row_schematics) else None
        if fn is not None:
            sx = place.axes(x0, py + (panel_h - schematic_in) / 2.0 - schematic_in * 0.45, gw - 0.12, schematic_in)
            sx.set_xlim(0, 1); sx.set_ylim(0, 1); sx.set_axis_off()
            sx._hf_no_letter = True
            sx._hf_schematic = True
            fn(sx, gw - 0.12)
        if label:
            ty = py + (panel_h + schematic_in * (0.9 if fn is not None else 0.0)) / 2.0
            fig.text(x0 / place.W, place.y_frac(ty), _wrap(label, gw - 0.12, label_pt), ha="left", va="center",
                     fontsize=label_pt, color=ink)

    return SlideGrid(fig, axes, names, list(col_headers), list(row_labels))


CARD_PAD_IN = 0.30
CARD_LEAD_IN = 0.30
CARD_BODY_PT = 15.0
CARD_HEAD_PT = 19.0
CARD_HEAD_LEAD = 1.22
CARD_HEAD_GAP_IN = 0.04
CARD_FOOT_BAND_IN = 0.78
CARD_RADIUS_IN = 0.10


class CardOverflow(ValueError):
    """Raised when content cannot be made to fit a card at the slide's type floor, or when a stack runs off the page.

    This is the whole point of the module: the alternative to raising is a silent collision, which is what shipped.
    """


def _slide_floor_pt():
    return float(T.typography()["slide"]["in_figure"])


def _renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        fig.canvas.draw()
        return fig.canvas.get_renderer()


def _page_in(fig):
    return tuple(float(v) for v in fig.get_size_inches())


def _bbox_top_in(fig, bbox):
    """A display bbox -> (x0, y0, x1, y1) in inches from the page's top-left, y growing DOWNWARD."""
    dpi = fig.dpi
    hpx = fig.get_size_inches()[1] * dpi
    return (bbox.x0 / dpi, (hpx - bbox.y1) / dpi, bbox.x1 / dpi, (hpx - bbox.y0) / dpi)


def card_ax(fig):
    """The one inch-coordinate overlay every card is drawn on: x right, y DOWN from the page's top-left.

    It is axis-off and unfilled, carries no panel letter, and is marked a schematic so `finish()` leaves it out of the
    notes roster. A deck that already owns such an axes can hand it over with `adopt_card_ax(fig, ax)`.
    """
    ax = getattr(fig, "_hf_card_ax", None)
    if ax is None:
        W, H = _page_in(fig)
        ax = fig.add_axes([0, 0, 1, 1], zorder=0)
        ax.set_axis_off()
        ax.set_facecolor("none")
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.invert_yaxis()
        ax._hf_no_letter = True
        ax._hf_schematic = True
        fig._hf_card_ax = ax
    return ax


def adopt_card_ax(fig, ax):
    """Use an existing inch-coordinate, top-left-origin axes for the cards (x: 0..W, y: 0..H inverted)."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    W, H = _page_in(fig)
    if abs(x0) > 1e-6 or abs(x1 - W) > 1e-6 or abs(y0 - H) > 1e-6 or abs(y1) > 1e-6:
        raise ValueError(f"a card axes must span x 0..{W:g} and y {H:g}..0 (inverted), not x {x0:g}..{x1:g} y {y0:g}..{y1:g}")
    ax._hf_no_letter = True
    fig._hf_card_ax = ax
    return ax


def _lines_for(items, w_in, pt, head_pt):
    """`items` -> the wrapped lines as (text, is_heading, pt). An item is a string, or (string, True) for a heading."""
    out = []
    for it in items:
        text, head = it if isinstance(it, (tuple, list)) else (it, False)
        size = head_pt if head else pt
        width = max(8, int(w_in * 72.0 / (size * EM_PER_CHAR)))
        for part in (textwrap.wrap(text, width) if text else [""]):
            out.append((part, bool(head), size))
    return out


def _draw_block(ax, x_in, y_in, lines, lead_in, color, head_color, zorder, body_pt=CARD_BODY_PT):
    """Draw the wrapped lines from a top-left origin. Returns the Text artists in order."""
    arts, y, prev_head = [], y_in, False
    for text, head, size in lines:
        if prev_head and not head:
            y += CARD_HEAD_GAP_IN
        arts.append(ax.text(x_in, y, text, fontsize=size, va="top", ha="left", zorder=zorder + 2,
                            color=head_color if head else color, fontweight="bold" if head else "normal"))
        y += lead_in * (max(CARD_HEAD_LEAD, size / float(body_pt)) if head else 1.0)
        prev_head = head
    return arts


def _block_extent_in(fig, arts):
    """(widest, bottom) in inches from the page's top-left, from what matplotlib ACTUALLY rendered.

    The bottom is the last line's rendered bottom, not the sum of the line advances. A line's drawn height is not its
    advance, so an advance-only sum is an estimate of the one number a card is sized against.
    """
    if not arts:
        return 0.0, 0.0
    r = _renderer(fig)
    boxes = [_bbox_top_in(fig, a.get_window_extent(r)) for a in arts]
    return max(b[2] for b in boxes), max(b[3] for b in boxes)


def measure(fig, w_in, items, *, pt=CARD_BODY_PT, head_pt=CARD_HEAD_PT, lead_in=CARD_LEAD_IN, pad_in=CARD_PAD_IN,
            accent=None, accent_w_in=0.055, **_colours):
    """The height in inches the content needs in a card `w_in` wide, measured from a real render at this type size.

    Nothing is left on the figure. This is what `row()` and `stack()` ask before they place anything.
    """
    ax = card_ax(fig)
    inner = w_in - 2 * pad_in - (accent_w_in + 0.06 if accent else 0.0)
    if inner <= 0.4:
        raise ValueError(f"a card {w_in:g} in wide leaves {inner:g} in inside its padding; widen it or cut the padding")
    arts = _draw_block(ax, 0.0, 0.0, _lines_for(items, inner, pt, head_pt), lead_in, "k", "k", 0, pt)
    try:
        _, bottom = _block_extent_in(fig, arts)
    finally:
        for a in arts:
            a.remove()
    return bottom + 2 * pad_in


def card(fig, rect, items, *, pt=CARD_BODY_PT, head_pt=CARD_HEAD_PT, lead_in=CARD_LEAD_IN, pad_in=CARD_PAD_IN,
         floor_pt=None, fc=None, ec=None, lw=0.9, radius=CARD_RADIUS_IN, accent=None, accent_w_in=0.055,
         color=None, head_color=None, zorder=1, label=None):
    """Draw one card at `rect` = (x, y, w, h) in inches from the page's top-left, and register it for `gates.cards`.

    `items` are strings, or `(string, True)` for a heading. Returns the rect as drawn.
    
    """
    x, y, w, h = (float(v) for v in rect)
    floor = _slide_floor_pt() if floor_pt is None else float(floor_pt)
    ax = card_ax(fig)
    ink = T.colour("neutral.ink")
    color = ink if color is None else color
    head_color = (color if head_color is None else head_color)
    fc = "#FFFFFF" if fc is None else fc
    ec = T.colour("neutral.rule") if ec is None else ec
    acc_w = (accent_w_in + 0.06) if accent else 0.0
    inner = w - 2 * pad_in - acc_w
    if inner <= 0.4:
        raise ValueError(f"a card {w:g} in wide leaves {inner:g} in inside its padding; widen it or cut the padding")

    size, head, lead, arts = pt, head_pt, lead_in, []
    while True:
        arts = _draw_block(ax, x + pad_in + acc_w, y + pad_in, _lines_for(items, inner, size, head), lead,
                           color, head_color, zorder, size)
        widest, bottom = _block_extent_in(fig, arts)
        need_h = (bottom - y) + pad_in
        need_w = (widest - x) + pad_in
        if need_h <= h + 1e-9 and need_w <= w + 1e-9:
            break
        for a in arts:
            a.remove()
        arts = []
        if size - 0.5 < floor:
            raise CardOverflow(
                f"card at ({x:g}, {y:g}) {w:g}x{h:g} in cannot hold its content at the {floor:g} pt slide floor: it "
                f"needs {need_h:.2f} in of height and {need_w:.2f} in of width. Cut the text, or let slides.row() / "
                f"slides.stack() size the card from the content. First item: "
                f"{(items[0] if items else items)!r:.90}")
        size -= 0.5
        head = max(size + 2.0, head - 0.5)
        lead = max(0.22, lead - 0.012)

    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                         fc=fc, ec=ec, lw=lw, zorder=zorder, transform=ax.transData)
    ax.add_patch(box)
    if accent:
        ax.add_patch(Rectangle((x, y + radius * 0.4), accent_w_in, h - radius * 0.8, fc=accent, ec="none",
                               zorder=zorder + 1, transform=ax.transData))
    cards_of(fig).append({"rect": (x, y, w, h), "texts": arts, "patch": box, "pt": size,
                          "label": label or (items[0][0] if items and isinstance(items[0], (tuple, list)) else
                                             (str(items[0])[:40] if items else ""))})
    return (x, y, w, h)


def cards_of(fig):
    """The cards registered on this figure, newest last. Clear it when you reuse a figure."""
    if not hasattr(fig, "_hf_cards"):
        fig._hf_cards = []
    return fig._hf_cards


def _blocks(blocks):
    """Normalise a `row`/`stack` child: a list of items, or a dict {"items": [...], **per-card overrides}."""
    out = []
    for b in blocks:
        if isinstance(b, dict):
            b = dict(b)
            out.append((b.pop("items"), b))
        else:
            out.append((list(b), {}))
    return out


def row(fig, x, y, w, blocks, *, gap_in=0.25, bottom_in=None, min_h_in=0.0, **kw):
    """A row of equal-width cards whose height is the TALLEST child's measured height. Returns the y below the row.

    Every child is measured before anything is drawn, so a long one lengthens the whole row rather than running into
    whatever sits underneath it. Place the next element at the returned y, never at a y guessed in advance.
    """
    items = _blocks(blocks)
    n = len(items)
    if n == 0:
        return y
    cw = (w - gap_in * (n - 1)) / n
    if cw <= 0:
        raise ValueError(f"{n} cards and {gap_in:g} in gaps do not fit in {w:g} in")
    heights = [measure(fig, cw, it, **{**kw, **ov}) for it, ov in items]
    h = max([min_h_in] + heights)
    limit = (_page_in(fig)[1] - CARD_FOOT_BAND_IN) if bottom_in is None else float(bottom_in)
    if y + h > limit + 1e-9:
        raise CardOverflow(
            f"a row of {n} cards at y = {y:g} in needs {h:.2f} in and so ends at {y + h:.2f} in, past the {limit:.2f} in "
            f"the page allows. Cut the longest card's text (it needs {max(heights):.2f} in), widen the row, or move the "
            f"row up.")
    for i, (it, ov) in enumerate(items):
        card(fig, (x + i * (cw + gap_in), y, cw, h), it, **{**kw, **ov})
    return y + h


def stack(fig, x, y, w, blocks, *, gap_in=0.20, bottom_in=None, **kw):
    """A column of cards, each as tall as its own measured content. Returns the y below the last card."""
    yy = y
    for it, ov in _blocks(blocks):
        h = measure(fig, w, it, **{**kw, **ov})
        yy = row(fig, x, yy, w, [{"items": it, **ov}], bottom_in=bottom_in, min_h_in=h, **kw) + gap_in
    return yy - gap_in


def check(fig, tol_in=0.01):
    """`gates.cards(fig)`, for a deck rendered outside `figproof.finish()`. Returns a list of problems (empty = pass)."""
    from . import gates as G
    return G.cards(fig, tol_in=tol_in)


def assert_ok(fig, name="", tol_in=0.01):
    """Raise `CardOverflow` if any card collides with a sibling, another filled box, or the page edge."""
    bad = check(fig, tol_in=tol_in)
    if bad:
        raise CardOverflow(f"{name or 'slide'}: " + "; ".join(bad))
