"""`overlap()` makes the decision and `two_groups()` draws it on a figproof page. Overlap is measured, not guessed: with both axes scaled to
their joint range, a point is MIXED when its nearest neighbour belongs to the other group, and the groups overlap when more than
`threshold` of the smaller group's points are mixed. Two separated clusters score near 0; two samples of one distribution score near the
larger group's share of the points.

    import figproof as fp
    from figproof import points
    fig = fp.figure("internal", ...)
    out = points.two_groups(fig, (x, y, w, h), (xa, ya), (xb, yb), names=("PCR-cDNA", "direct RNA"), roles=("focal", "reference"),
                            xlabel="...", ylabel="...")
    out["decision"]   # "facets" or "overlay"; out["axes"] is the list of axes drawn
"""
from __future__ import annotations

import matplotlib.patheffects as pe
import numpy as np

from . import tokens as T

THRESHOLD = 0.20
MARKER_AREA = 16
HALO_PT = 2.2
FACET_GAP_IN = 0.18


def mixing(xa, ya, xb, yb):
    """Share of the smaller group's points whose nearest neighbour (axes scaled to their joint range) belongs to the other group."""
    a = np.column_stack([np.asarray(xa, float), np.asarray(ya, float)])
    b = np.column_stack([np.asarray(xb, float), np.asarray(yb, float)])
    if len(a) == 0 or len(b) == 0:
        return 0.0
    pts = np.vstack([a, b])
    span = pts.max(axis=0) - pts.min(axis=0)
    span[span == 0] = 1.0
    z = (pts - pts.min(axis=0)) / span
    lab = np.r_[np.zeros(len(a), int), np.ones(len(b), int)]
    d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = lab[d.argmin(axis=1)]
    small = 0 if len(a) <= len(b) else 1
    sel = lab == small
    return float((nn[sel] != small).mean())


def overlap(xa, ya, xb, yb, threshold=THRESHOLD):
    return mixing(xa, ya, xb, yb) > threshold


def _styles(roles):
    from . import panels as P
    ink = T.colour("neutral.ink")
    ca, cb = (T.colour(f"role.{r}") for r in roles)
    filled = dict(marker="D", c=ca, edgecolors=ink, linewidths=P.STROKE["hairline"])
    ring = dict(marker="o", facecolors="white", edgecolors=cb, linewidths=P.STROKE["secondary"],
                path_effects=[pe.withStroke(linewidth=HALO_PT, foreground="white")])
    return filled, ring


def _key(ax, entries, y=1.07):
    ""
    from . import panels as P
    ink = T.colour("neutral.ink")
    x = 0.03
    for name, kw in entries:
        kw = {k: v for k, v in kw.items() if k != "path_effects"}
        ax.scatter([x], [y], s=MARKER_AREA, transform=ax.transAxes, clip_on=False, **kw)
        ax.text(x + 0.05, y, name, transform=ax.transAxes, va="center", ha="left", fontsize=P.TYPE["annotation"], color=ink,
                clip_on=False)
        x += 0.46


def two_groups(fig, box, a, b, *, names=("A", "B"), roles=("focal", "reference"), xlabel=None, ylabel=None, threshold=THRESHOLD,
               label=None, decision=None):
    """Draw groups a = (x, y) and b = (x, y) inside `box` = (x, y, w, h) inches on a figproof page, faceted when they overlap and as
    filled against haloed open rings when they do not. `decision` forces "facets" or "overlay" (say why at the call site).
    Returns {"decision", "mixing", "axes"}."""
    (xa, ya), (xb, yb) = a, b
    mix = mixing(xa, ya, xb, yb)
    decision = decision or ("facets" if mix > threshold else "overlay")
    filled, ring = _styles(roles)
    ink = T.colour("neutral.ink")
    allx, ally = np.r_[np.asarray(xa, float), np.asarray(xb, float)], np.r_[np.asarray(ya, float), np.asarray(yb, float)]
    padx, pady = 0.06 * (np.ptp(allx) or 1.0), 0.08 * (np.ptp(ally) or 1.0)
    xlim, ylim = (allx.min() - padx, allx.max() + padx), (ally.min() - pady, ally.max() + pady)
    x0, y0, w, h = box
    if decision == "overlay":
        ax = fig.fp.axes(x0, y0, w, h, label=label)
        ax.scatter(xa, ya, s=MARKER_AREA, zorder=3, **filled)
        ax.scatter(xb, yb, s=MARKER_AREA, zorder=4, **ring)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        _key(ax, [(names[0], filled), (names[1], ring)])
        return {"decision": decision, "mixing": mix, "axes": [ax]}
    if decision != "facets":
        raise ValueError(f"decision must be 'facets' or 'overlay', not {decision!r}")
    fw = (w - FACET_GAP_IN) / 2.0
    axl = fig.fp.axes(x0, y0, fw, h, label=label)
    axr = fig.fp.axes(x0 + fw + FACET_GAP_IN, y0, fw, h, sharey=axl)
    for ax_, (xs, ys), kw, name in ((axl, a, filled, names[0]), (axr, b, ring, names[1])):
        ax_.scatter(xs, ys, s=MARKER_AREA, zorder=3, **kw)
        ax_.set_xlim(*xlim); ax_.set_ylim(*ylim)
        _key(ax_, [(name, kw)], y=1.08)
    axr.tick_params(labelleft=False)
    if ylabel:
        axl.set_ylabel(ylabel)
    if xlabel:
        axl.set_xlabel(xlabel)
        fig.draw_without_rendering()
        inv = axl.transAxes.inverted()
        lab = axl.xaxis.label.get_window_extent()
        mid_disp = (axl.get_window_extent().x0 + axr.get_window_extent().x1) / 2.0
        axl.xaxis.set_label_coords(*inv.transform((mid_disp, lab.y0 + lab.height / 2.0)))
    return {"decision": decision, "mixing": mix, "axes": [axl, axr]}
