"""The standard every figstyle-using figure follows:
  1. ONE concise combined legend, LEFT-justified (journal default; "full" is opt-in), ~9 pt, dark grey ("0.15").
  2. Bold spans: the "Figure N | <title>." lead (through the first sentence-ending period) AND every
     "(a)"/"(b)"/"(c)" panel callout are BOLD; the rest normal. (mpl has no inline bold, so this is done by
     per-word placement — see _segments + the _layout/_render engine below.)
  3. The legend CALLS OUT each panel "(a)", "(b)", ...
  4. Panel letters on the axes = bold LOWERCASE a, b, c (the house style for lab and collaborator figures, matching the "(a)"
     callouts). Journal and supplementary figures default to bold capitals A, B, C; see figproof.figure(letters=...).
  5. No redundant top suptitle (the title lives in the legend's bold lead); fold per-panel titles into x-labels.
  6. Minimal text on the figure; full detail -> a .md sidecar per PNG.
  7. Keep the date+time / script provenance stamp (small grey, bottom-right).
"""
import datetime, textwrap, re


def panel_letters(axes, letters="abcdefgh", fontsize=14, dx=-0.13, dy=1.06, min_fig_x=0.004,
                  max_fig_y=0.998):
    """Bold LOWERCASE panel letters (figure standard, matches the '(a)' legend callouts) at each axes'
    top-left. Override `letters` (e.g. 'ABCDEFGH') only if a figure deliberately needs uppercase.

    Fix is backwards-compatible: the letter is placed exactly as before whenever it lands on-canvas;
    only when it would fall off the left edge is it clamped to `min_fig_x` in figure coordinates.
    Call this AFTER the layout is settled (after `gs.update`/`subplots_adjust`), since it reads
    `ax.get_position()`.
    
    """
    for ax, l in zip(axes, letters):
        fig = ax.get_figure()
        bb = ax.get_position()
        fx = bb.x0 + dx * bb.width
        fy = bb.y0 + dy * bb.height
        if fy > max_fig_y:
            fy = max_fig_y
            fx = max(fx, min_fig_x)
            fig.text(fx, fy, l, fontsize=fontsize, fontweight="bold", va="top", ha="left")
        elif fx < min_fig_x:
            fig.text(min_fig_x, fy, l, fontsize=fontsize,
                     fontweight="bold", va="bottom", ha="left")
        else:
            ax.text(dx, dy, l, transform=ax.transAxes, fontsize=fontsize, fontweight="bold",
                    va="bottom", ha="left")


def stamp(fig, script, fontsize=7.0):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.995, 0.004, f"{script} · {ts}", ha="right", va="bottom", fontsize=fontsize, color="0.5")


def _segments(text):
    ""
    segs = []; in_title = True
    for w in text.split():
        panel = bool(re.match(r"^\([a-z]\)$", w))
        segs.append((w, in_title or panel))
        if in_title and w.endswith("."):
            in_title = False
    return segs


def _layout(fig, text, x0, x1, fontsize, linespacing):
    ""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    Wpx = fig.get_figwidth() * fig.dpi

    def wpx(s, b):
        t = fig.text(0, 0, s, fontsize=fontsize, fontweight=("bold" if b else "normal"))
        bb = t.get_window_extent(r); t.remove(); return bb.width
    sp = wpx("x x", False) - 2 * wpx("x", False)
    toks = [(w, b, wpx(w, b)) for (w, b) in _segments(text)]
    maxpx = (x1 - x0) * Wpx
    lines = []; cur = []; cw = 0.0
    for tok in toks:
        add = tok[2] + (sp if cur else 0)
        if cur and cw + add > maxpx:
            lines.append(cur); cur = [tok]; cw = tok[2]
        else:
            cur.append(tok); cw += add
    if cur:
        lines.append(cur)
    lh = fontsize * linespacing / (72.0 * fig.get_figheight())
    return lines, sp, maxpx, lh, Wpx


def _render(fig, lines, sp, maxpx, Wpx, x0, y_top, fontsize, color, lh, full):
    """Place the pre-wrapped `lines` top-down from y_top, one word at a time (bold per token).
    `full`=True spreads inter-word gaps to full-justify every non-final line; False = single space (LEFT)."""
    y = y_top
    for i, line in enumerate(lines):
        last = (i == len(lines) - 1)
        gap = ((maxpx - sum(t[2] for t in line)) / (len(line) - 1)) if (full and not last and len(line) > 1) else sp
        x = x0
        for (w, b, pw) in line:
            fig.text(x, y, w, fontsize=fontsize, color=color, ha="left", va="top",
                     fontweight=("bold" if b else "normal"))
            x += (pw + gap) / Wpx
        y -= lh
    return y


def concise_legend(fig, text, script="", y0=0.03, fontsize=9.0, x0=0.055, x1=0.965,
                   justify="left", color="0.15", linespacing=1.5, width=None, pad=0.03):
    """    Returns the block TOP y (figure fraction) so the caller sets
        fig.subplots_adjust(bottom=returned_top + pad).

    Back-compat: existing callers `concise_legend(fig, text, script=..., width=..., fontsize=..., y0=...)`
    still work. `width` (old char-wrap width) is ACCEPTED but IGNORED — line breaks now come from pixel
    measurement against x0..x1, so a caller that passed a large `width` may wrap to more lines (taller block)
    and want a bigger bottom pad on re-render. NOTE: `text` is whitespace-tokenized, so explicit '\\n'
    paragraph breaks in `text` are flattened into one reflowed block (unlike the old centered version).
    """
    full = (justify == "full")
    lines, sp, maxpx, lh, Wpx = _layout(fig, text, x0, x1, fontsize, linespacing)
    y_top = y0 + len(lines) * lh
    _render(fig, lines, sp, maxpx, Wpx, x0, y_top, fontsize, color, lh, full)

    try:
        from matplotlib.layout_engine import ConstrainedLayoutEngine
        eng = fig.get_layout_engine()
        if isinstance(eng, ConstrainedLayoutEngine):
            b = min(max(y_top + pad, 0.0), 0.95)
            eng.set(rect=(0.0, b, 1.0, 1.0 - b))
    except Exception:
        pass

    if script:
        stamp(fig, script, fontsize=6.5)
    return y_top


def legend_block(fig, caption, methods="", script="", y0=0.028, fontsize=8.5, width=155,
                 methods_fontsize=7.0):
    """LEGACY two-part legend (caption + a small grey methods block). Retained for back-compat; it is NOT
    updated to the figure standard — it stays centered/char-wrapped with no bold lead or per-word (a)/(b)
    bolding. For NEW figures use `concise_legend` (the figure standard). Returns the block TOP y (figure
    fraction) so the caller can set `fig.subplots_adjust(bottom=returned_top + pad)`."""
    fh = fig.get_size_inches()[1]

    def wrap(t, w):
        out = []
        for p in t.split("\n"):
            out += textwrap.wrap(p, width=w) or [""]
        return out
    clines = wrap(caption, width)
    mlines = wrap(methods, int(width * 1.12)) if methods else []
    lh = fontsize / (fh * 72) * 1.4
    mlh = methods_fontsize / (fh * 72) * 1.4
    y = y0
    if mlines:
        fig.text(0.5, y, "\n".join(mlines), ha="center", va="bottom", fontsize=methods_fontsize, color="0.45")
        y += len(mlines) * mlh + 0.010
    fig.text(0.5, y, "\n".join(clines), ha="center", va="bottom", fontsize=fontsize, color="0.12")
    top = y + len(clines) * lh
    if script:
        stamp(fig, script)
    return top


def dfv_feature(start, end, strand=0, label=None, color="#cccccc", fontsize=7.0,
                thickness=14, **kwargs):
    """`dna_features_viewer.GraphicFeature` with this standard's defaults pre-applied.

    DFV's own defaults violate the figure standard in a way `figproof.qa` CANNOT catch (a font-size
    mismatch is not an overlap), so prefer this over constructing GraphicFeature directly:

      * `box_color=None`      -- drops DFV's rounded grey "tooltip" box around labels.
      * `fontdict={"fontsize": ...}` -- DFV IGNORES rcParams["font.size"] and hardcodes 11 pt.
        NOTE the key must be `fontsize`; `size` raises TypeError (DFV passes `fontsize` itself).

    Verified against dna_features_viewer 3.1.5 (2026-08-12). Import is LAZY so that merely importing
    figstyle does not require DFV to be installed.

    Reversed axes (minus-strand loci drawn 5'->3' left to right) work: call `.plot(ax=ax)` then
    `ax.set_xlim(high, low)`. Strand lives in DATA coordinates -- the arrow tip is at the LOW
    coordinate for strand=-1 -- so the direction stays correct after reversal with no manual flip.
    Arrowheads do flatten to a plain bar for a long feature in a narrow window; add your own cue.
    """
    try:
        from dna_features_viewer import GraphicFeature
    except ImportError as exc:
        raise ImportError(
            "dna_features_viewer is not installed in this interpreter "
            "(`pip install dna-features-viewer`; pure-Python, no compiled deps)."
        ) from exc
    fontdict = dict(kwargs.pop("fontdict", None) or {})
    if "size" in fontdict:
        fontdict["fontsize"] = fontdict.pop("size")
    fontdict.setdefault("fontsize", fontsize)
    kwargs.setdefault("box_color", None)
    return GraphicFeature(start=start, end=end, strand=strand, label=label, color=color,
                          thickness=thickness, fontdict=fontdict, **kwargs)


def letters_2f(fig, panels, gap_in=0.10, fontsize=14, align=True, columns=(), rows=()):
    """the figure standard (2026-09-16): bold lowercase panel letters placed in FIGURE coordinates from the rendered
    geometry — bottom-right corner `gap_in` above the panel's content top and `gap_in` left of its leftmost decoration
    (y-label / tick labels; for an axis-off schematic the ink box) — then, with align=True, one shared x per column and
    one shared y per row (panels whose reference edges lie within 0.05 in are grouped). `panels` = [(ax, "a"), ...].
    Returns {letter: (fig_x, fig_y)} of the placed bottom-right corners (inches) for gating."""
    import matplotlib
    fig.canvas.draw(); r = fig.canvas.get_renderer(); dpi = fig.dpi
    def bb(a):
        b = a.get_window_extent(renderer=r); return (b.x0 / dpi, b.y0 / dpi, b.x1 / dpi, b.y1 / dpi)
    spec = []
    for ax, letter in panels:
        if ax.axison:
            fb = bb(ax); items = [fb, bb(ax.yaxis.label)] if ax.yaxis.label.get_text() else [fb]
            ylo, yhi = sorted(ax.get_ylim())
            items += [bb(t) for t in ax.get_yticklabels() if t.get_visible() and t.get_text() and ylo <= t.get_position()[1] <= yhi]
            top = fb[3]
        else:
            arts = [x for x in list(ax.patches) + list(ax.lines) + list(ax.collections) + list(ax.texts) if x.get_visible()]
            items = [b for b in (bb(x) for x in arts) if b[2] > b[0] and b[3] > b[1]] or [bb(ax)]
            top = max(b[3] for b in items)
        left = min(b[0] for b in items)
        spec.append([ax, letter, left - gap_in, top + gap_in])
    if align:
        for i in range(len(spec)):
            for j in range(len(spec)):
                if abs(spec[i][2] - spec[j][2]) < 0.05: spec[i][2] = spec[j][2] = min(spec[i][2], spec[j][2])
                if abs(spec[i][3] - spec[j][3]) < 0.05: spec[i][3] = spec[j][3] = max(spec[i][3], spec[j][3])
    idx = {sp[1]: sp for sp in spec}
    for col in columns:
        x = min(idx[l][2] for l in col); [idx[l].__setitem__(2, x) for l in col]
    for row in rows:
        y = max(idx[l][3] for l in row); [idx[l].__setitem__(3, y) for l in row]
    W, H = fig.get_figwidth(), fig.get_figheight(); out = {}
    for ax, letter, x, y in spec:
        fig.text(x / W, y / H, letter, fontweight="bold", fontsize=fontsize, ha="right", va="bottom")
        out[letter] = (x, y)
    return out
