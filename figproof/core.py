"""figproof.core — `figure()` and `finish()`, one entry and one exit per figure."""
from __future__ import annotations

import datetime
import io
import os
import shutil
import subprocess

import matplotlib

from . import _legacy
from . import gates as Gt
from . import tokens as T

MM = 25.4
WIDTHS_MM = {"third": 57.0, "single": 89.0, "onehalf": 120.0, "twothirds": 121.0, "double": 183.0, "full": 184.0}
SUPPLEMENTARY_WIDTHS_MM = {"third": 57.0, "single": 89.0, "onehalf": 120.0, "twothirds": 121.0, "double": 180.0}
WIDTH_TOL_MM = 1.0
LETTER_CASES = ("lower", "upper")
FOOT_IN, LEG_GAP_IN, PAD_IN = 0.36, 0.12, 0.08
SIDECAR_STUB = """# {name}

> STUB written by `figproof.finish()`. The concise legend on the figure is short by design; this file is the reviewer-grade record the
> figure standard requires. Fill in every TODO.

## Data source
TODO

## Processing
TODO

## Per panel
TODO

## Caveats
TODO

---
figure: `{png}` · mode: internal · script: `{script}` · rendered {ts}
"""


class GateError(RuntimeError):
    """finish() refused to write: one or more gates failed. `.problems` lists them."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("figproof.finish refused to write:\n  " + "\n  ".join(self.problems))


def exempt(artist, why: str = ""):
    """Mark one artist as a deliberate exception to the floors and the colour gates. Say why at the call site."""
    artist.set_gid("figproof-exempt")
    return artist


def layer(ax, name: str):
    """Tag an axes with the colour layer it speaks (identity, role, mark or neutral); the default is identity for an axis-off schematic
    and role otherwise. Tag every panel the default gets wrong, for example an alignment panel drawn with its axis on."""
    if name not in T.COLOUR_LAYERS + ("neutral",):
        raise ValueError(f"layer must be one of {T.COLOUR_LAYERS + ('neutral',)}, not {name!r}")
    ax._hf_layer = name
    return ax


def data_text(text):
    ""
    text.set_gid("figproof-data-text")
    return text


_OPEN = []


def discard(fig):
    """Close a figproof figure without saving it, restoring the type scale. Use it when drawing fails before finish()."""
    fp = getattr(fig, "fp", None)
    if fp is not None and fig in _OPEN:
        _OPEN.remove(fig)
        fp.scale.restore()


def no_letter(ax):
    """Keep an axes out of the automatic panel lettering (an inset, a key axes, a colour bar)."""
    ax._hf_no_letter = True
    return ax


class _Scale:
    """Applies a mode's type scale to rcParams and to the package modules' TYPE dicts; restore() puts everything back."""

    def __init__(self, kind, P, G, M, stroke_cap=None):
        self.saved_rc = matplotlib.rcParams.copy()
        self.P, self.G, self.M = P, G, M
        self.saved = [(d, dict(d)) for d in {id(x): x for x in (P.TYPE, G.TYPE, getattr(M, "TYPE", G.TYPE))}.values()]
        strokes = {id(x): x for x in (P.STROKE, G.STROKE, getattr(M, "STROKE", G.STROKE))}.values()
        self.saved += [(d, dict(d)) for d in strokes]
        self.saved_style = G.STYLE
        self.saved_effects = getattr(G, "EFFECTS", True)
        P.apply(matplotlib)
        if kind in ("journal", "abstract"):
            j = T.typography()[kind]
            matplotlib.rcParams.update({"font.size": j["in_figure"], "axes.labelsize": j["axis_label"], "axes.titlesize": j["axis_label"],
                                        "xtick.labelsize": j["tick_label"], "ytick.labelsize": j["tick_label"], "legend.fontsize": j["key"]})
            for d, _ in self.saved:
                for k in ("axis_label", "tick_label", "in_figure", "annotation", "panel_letter"):
                    if k in d and k in j:
                        d[k] = j[k]
        if stroke_cap is not None:
            for d in strokes:
                for k in d:
                    d[k] = min(d[k], stroke_cap)
            matplotlib.rcParams.update({k: min(matplotlib.rcParams[k], stroke_cap) for k in
                                        ("lines.linewidth", "axes.linewidth", "patch.linewidth", "xtick.major.width", "ytick.major.width")})

    def restore(self):
        matplotlib.rcParams.update(self.saved_rc)
        for d, orig in self.saved:
            d.clear(); d.update(orig)
        self.G.STYLE = self.saved_style
        self.G.EFFECTS = self.saved_effects


class HF:
    """What figure() hands to finish(), attached as `fig.fp`."""

    def __init__(self, fig, mode, spec, W, H, top, body_h, bottom, legend, script, provenance, title, scale, legend_pt, letter_case):
        self.fig, self.mode, self.spec = fig, mode, spec
        self.W, self.H, self.top, self.body_h, self.bottom = W, H, top, body_h, bottom
        self.legend, self.script, self.provenance, self.title = legend, script, provenance, title
        self.scale, self.legend_pt, self.letter_case = scale, legend_pt, letter_case

    def axes(self, x, y, w, h, *, label=None, **kw):
        """An axes placed in inches from the BODY's top-left corner (y grows downward, as on a page)."""
        W, H = self.W, self.H
        ax = self.fig.add_axes([x / W, 1 - (self.top + y + h) / H, w / W, h / H], **kw)
        if label:
            ax._hf_label = label
        return ax


def _width_in(width, mode):
    """A named column width, a string in millimetres ("57mm"), or a number in inches."""
    table = SUPPLEMENTARY_WIDTHS_MM if mode == "supplementary" else WIDTHS_MM
    if isinstance(width, str):
        w = width.strip().lower()
        if w.endswith("mm"):
            try:
                return float(w[:-2]) / MM
            except ValueError:
                pass
        if width not in table:
            raise ValueError(f"width {width!r}: use one of {sorted(table)}, a width in millimetres such as '57mm', or inches")
        return table[width] / MM
    return float(width)


def _letter_case(letters, spec, mode):
    case = letters or spec.get("letter_case", "lower")
    if case not in LETTER_CASES:
        raise ValueError(f"letters={letters!r}: use 'upper' (A, B, C) or 'lower' (a, b, c)")
    return case


def _letter_pt(typo, spec, case):
    if spec["letters"] == "journal":
        return typo["letters_pt"]["journal_" + case]
    return typo["letters_pt"]["standard"]


def figure(mode="internal", width="double", body_height=3.0, *, legend=None, script=None, provenance=None, title=None, top=0.30,
           stroke_cap=None, letters=None):
    """Open a figure for `mode`. `letters` sets the panel-letter case: "upper" (A, B, C; 10 pt bold in journal and supplementary
    modes, the default there) or "lower" (a, b, c; 8 pt bold in journal modes, and the default for internal and collab figures).
    """
    import matplotlib.pyplot as plt
    spec = T.mode_spec(mode)
    P, G, M, FS = _legacy.panels(), _legacy.glyphs(), _legacy.molecules(), _legacy.figstyle()
    W = _width_in(width, mode)
    if mode == "abstract":
        W, body_height, top = spec["page_in"][0], spec["page_in"][1], 0.0
    letter_case = _letter_case(letters, spec, mode)
    if mode == "journal":
        mm = W * MM
        if min(abs(mm - w) for w in spec["widths_mm"]) > WIDTH_TOL_MM:
            raise ValueError(f"journal width {mm:.1f} mm: the allowed widths are {spec['widths_mm']} mm")
    if mode == "slide" and not title:
        raise ValueError("slide mode needs title= (the one-sentence claim above the figure)")
    if spec["legend"] == "page" and not legend:
        raise ValueError(f"{mode} mode puts the legend on the page; pass legend=")
    if mode == "internal" and not provenance:
        raise ValueError("internal mode needs provenance= (data source and key processing; the figure standard)")
    if _OPEN:
        raise RuntimeError("another figproof figure is still open: finish() it, or figproof.discard(fig) it, before starting the next "
                           "(the type scale is process-global, so two open figures would restore it in the wrong order)")
    typo = T.typography()
    legend_pt = typo["legend_pt"]
    if stroke_cap is None:
        stroke_cap = spec.get("stroke_cap_pt")
    scale = _Scale(spec["type"], P, G, M, stroke_cap=stroke_cap)
    leg_h = 0.0
    if spec["legend"] == "page":
        probe = plt.figure(figsize=(W, 4.0), layout="none")
        leg_h = FS.concise_legend(probe, legend, script="", y0=0.0, fontsize=legend_pt) * probe.get_figheight()
        plt.close(probe)
    bottom = {"internal": FOOT_IN + leg_h + LEG_GAP_IN, "collab": PAD_IN + leg_h + LEG_GAP_IN, "abstract": 0.0}.get(mode, PAD_IN)
    H = top + body_height + bottom
    if mode == "journal" and H * MM > spec["max_height_mm"] + 1e-6:
        scale.restore()
        raise ValueError(f"journal page is {H * MM:.0f} mm tall; the maximum is {spec['max_height_mm']} mm")
    if mode == "supplementary" and H * MM > spec["max_mm"][1] + 1e-6:
        scale.restore()
        raise ValueError(f"supplementary page is {H * MM:.0f} mm tall; the maximum is {spec['max_mm'][1]} mm")
    if spec["type"] == "journal":
        G.STYLE = "flat"
        G.EFFECTS = False
    fig = plt.figure(figsize=(W, H), layout="none")
    fig.fp = HF(fig, mode, spec, W, H, top, body_height, bottom, legend, script, provenance, title, scale, legend_pt, letter_case)
    _OPEN.append(fig)
    return fig


def _pdf_checks(path, W, H, spec):
    bad = []
    wmm, hmm = W * MM, H * MM
    if min(abs(wmm - w) for w in spec["widths_mm"]) > WIDTH_TOL_MM:
        bad.append(f"journal PDF is {wmm:.1f} mm wide; the allowed widths are {spec['widths_mm']} mm")
    if hmm > spec["max_height_mm"] + 1e-6:
        bad.append(f"journal PDF is {hmm:.0f} mm tall; the maximum is {spec['max_height_mm']} mm")
    if shutil.which("pdffonts"):
        out = subprocess.run(["pdffonts", path], capture_output=True, text=True).stdout.splitlines()[2:]
        if not out:
            bad.append("the PDF lists no fonts: the text was outlined (journals require live, unoutlined text)")
        for ln in out:
            cols = ln.split()
            if "Type 3" in ln or ("TrueType" not in ln):
                bad.append(f"font not embedded as TrueType: {ln.strip()}")
            elif len(cols) >= 5 and "no" in cols[-5:-3]:
                bad.append(f"font not embedded: {ln.strip()}")
    return bad


def finish(fig, name, outdir, *, panels=None, rows=(), columns=(), strict=True, sidecar_text=None):
    """Legend -> letters -> footer and stamp -> draw -> gates -> save -> guarded sidecar. Returns {files, problems, warnings}.
    With strict=True (default) a failed gate raises GateError and nothing is written."""
    fp = fig.fp
    P, FS, FQ = _legacy.panels(), _legacy.figstyle(), _legacy.figqa()
    spec, mode, W, H = fp.spec, fp.mode, fp.W, fp.H
    typo = T.typography()
    ink = T.colour("neutral.ink")
    files, problems, warnings = [], [], []
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        if spec["legend"] == "page":
            y0 = (FOOT_IN if spec["footer"] else PAD_IN) / H
            script_label = (os.path.basename(fp.script) if fp.script else name) if spec["stamp"] else ""
            FS.concise_legend(fig, fp.legend, script=script_label, y0=y0, fontsize=fp.legend_pt, color=ink)
        fp.notes_roster = []
        if spec["letters"] == "notes":
            named = [a for a in fig.axes if a.get_visible() and not getattr(a, "_hf_schematic", False)]
            fp.notes_roster = [(ltr, getattr(a, "_hf_panel_name", "") or getattr(a, "_hf_label", "") or "")
                               for ltr, a in zip("abcdefghijklmnopqrstuvwxyz", named)]
            panels = []
        elif spec["letters"] == "none":
            panels = []
        elif panels is None:
            axs = [a for a in fig.axes if a.get_visible() and not getattr(a, "_hf_no_letter", False)]
            panels = list(zip(axs, "abcdefghijklmnopqrstuvwxyz")) if len(axs) > 1 else []
        case = fp.letter_case
        panels = [(ax, ltr.upper() if case == "upper" else ltr.lower()) for ax, ltr in (panels or [])]
        letter_pt = _letter_pt(typo, spec, case)
        before = len(fig.texts)
        if panels:
            FS.letters_2f(fig, panels, fontsize=letter_pt, rows=rows, columns=columns)
        letters = fig.texts[before:]
        if spec["footer"]:
            fig.text(0.055, 0.15 / H, fp.provenance, fontsize=5.5, color="0.45", ha="left", va="bottom")
        fig.canvas.draw()
        if spec["type"] == "journal":
            problems += Gt.type_band_journal(fig, letters, lo=typo["journal"]["min_pt"], hi=typo["journal"]["max_pt"],
                                             letter_pt=letter_pt, case=case, panels_mod=P)
            problems += Gt.no_minor_ticks_or_grid(fig)
            warnings += Gt.symbol_size(fig)
        elif spec["type"] == "abstract":
            problems += Gt.type_band_journal(fig, letters, lo=typo["abstract"]["min_pt"], hi=typo["abstract"]["max_pt"],
                                             letter_pt=typo["abstract"]["panel_letter"], panels_mod=P)
            problems += Gt.abstract_one_panel(fig, spec.get("panels", 1))
            warnings += Gt.abstract_text_budget(fig, spec.get("text_budget_words"))
        else:
            problems += Gt.floors_standard(fig, P)
        problems += Gt.overlaps(fig, FQ)
        problems += Gt.containment(fig, FQ)
        problems += Gt.cards(fig)
        problems += Gt.hex_meanings(fig)
        problems += Gt.coloured_text(fig)
        warnings += Gt.declarable_warnings(fig)
        if letters:
            warnings += Gt.legend_callout_case(fp.legend, case)
        if spec["type"] != "abstract":
            warnings += Gt.caption_annotations(fig)
            warnings += Gt.panel_headings(fig, slide=spec["letters"] == "notes")
        if spec["letters"] == "notes":
            problems += Gt.slide_panels(fig)
        if spec.get("stroke_cap_pt") is not None:
            problems += Gt.strokes_at_most(fig, spec["stroke_cap_pt"])
        warnings.append(f"{len(FQ.find_overlaps(fig, include_data=True))} overlaps including data (advisory)")
        if problems and strict:
            raise GateError(problems)
        os.makedirs(outdir, exist_ok=True)
        base = os.path.join(outdir, name)
        dpi = spec["dpi"]
        if mode in ("internal", "collab"):
            fig.savefig(base + ".png", dpi=dpi); files.append(base + ".png")
        elif mode == "journal":
            fig.savefig(base + ".pdf"); files.append(base + ".pdf")
            fig.savefig(base + ".png", dpi=dpi); files.append(base + ".png")
            with open(base + "_journal_legend.txt", "w") as fh:
                fh.write((fp.legend or "") + "\n")
            files.append(base + "_journal_legend.txt")
            pdf_bad = _pdf_checks(base + ".pdf", W, H, spec)
            if pdf_bad:
                problems += pdf_bad
                if strict:
                    for f in files:
                        os.remove(f)
                    raise GateError(pdf_bad)
        elif mode == "supplementary":
            fig.savefig(base + ".jpg", dpi=min(dpi, 300), pil_kwargs={"quality": 95}); files.append(base + ".jpg")
            with open(base + "_legend.txt", "w") as fh:
                fh.write((fp.legend or "") + "\n")
            files.append(base + "_legend.txt")
            size = os.path.getsize(base + ".jpg")
            wmm, hmm = W * MM, H * MM
            ext_bad = ([f"supplementary file is {size / 1e6:.1f} MB; the maximum is 10 MB"] if size > spec["max_bytes"] else []) + \
                      ([f"supplementary page {wmm:.0f} x {hmm:.0f} mm exceeds {spec['max_mm']} mm"]
                       if wmm > spec["max_mm"][0] + 0.5 or hmm > spec["max_mm"][1] + 0.5 else [])
            if ext_bad:
                problems += ext_bad
                if strict:
                    for f in files:
                        os.remove(f)
                    raise GateError(ext_bad)
        elif mode == "slide":
            files += _slide(fig, fp, base, dpi)
            warnings += getattr(fp, "slide_warnings", [])
        elif mode == "abstract":
            sw, sh = spec["page_in"]
            if abs(W - sw) > 1e-6 or abs(H - sh) > 1e-6:
                bad = [f"abstract page is {W:.3f} x {H:.3f} in; a graphical abstract is {sw} x {sh} in square"]
                problems += bad
                if strict:
                    raise GateError(bad)
            fig.savefig(base + ".png", dpi=dpi); files.append(base + ".png")
            fig.savefig(base + ".pdf"); files.append(base + ".pdf")
        if spec["sidecar"]:
            md = base + ".md"
            if os.path.exists(md):
                warnings.append(f"sidecar NOT written: {md} exists (refusing to overwrite)")
            else:
                with open(md, "w") as fh:
                    fh.write(sidecar_text or SIDECAR_STUB.format(name=name, png=os.path.basename(base + ".png"),
                                                                  script=fp.script or "TODO", ts=stamp))
                files.append(md)
        return {"files": files, "problems": problems, "warnings": warnings, "mode": mode, "page_in": [round(W, 4), round(H, 4)]}
    finally:
        if fig in _OPEN:
            _OPEN.remove(fig)
        fp.scale.restore()


def _slide_title(SW, SH, title):
    import matplotlib.pyplot as plt
    s = plt.figure(figsize=(SW, SH), layout="none")
    s.text(0.45 / SW, 1 - 0.42 / SH, title, fontsize=20, fontweight="bold", ha="left", va="top", color=T.colour("neutral.ink"))
    return s


def _slide(fig, fp, base, dpi):
    ""
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi); buf.seek(0)
    img = np.asarray(Image.open(buf).convert("RGB"))
    SW, SH = fp.spec["page_in"]
    s = _slide_title(SW, SH, fp.title)
    ih, iw = img.shape[:2]
    h_in = min(SH - 1.25, (SW - 0.9) * ih / iw); w_in = h_in * iw / ih
    ax = s.add_axes([(SW - w_in) / 2 / SW, 0.35 / SH, w_in / SW, h_in / SH]); ax.imshow(img); ax.set_axis_off()
    s.savefig(base + ".png", dpi=dpi); plt.close(s)
    svg = []
    try:
        _slide_svg(fig, fp, base + ".svg", ((SW - w_in) / 2, SH - 0.35 - h_in, w_in))
        svg = [base + ".svg"]
    except Exception as exc:
        fp.slide_warnings = [f"slide SVG (for Google Slides) NOT written: {type(exc).__name__}: {exc}"]
    with open(base + "_notes.txt", "w") as fh:
        fh.write((fp.legend or "") + "\n")
        roster = [r for r in getattr(fp, "notes_roster", []) if r[1]]
        if roster:
            fh.write("\nPanels, in reading order:\n")
            for ltr, name in roster:
                fh.write(f"  ({ltr}) {name}\n")
    return [base + ".png"] + svg + [base + "_notes.txt"]


def _slide_svg(fig, fp, path, box):
    """The slide as one SVG: matplotlib's own title page with the body figure's SVG grafted in at `box` = (x, y, w) inches from the
    slide's top-left, the same place the PNG puts the raster. Text stays live (svg.fonttype none). The title page's ids get a prefix,
    because both SVGs number their elements from 1 and a duplicate id would let a url(#...) reference resolve to the wrong element."""
    import xml.etree.ElementTree as ET
    import matplotlib.pyplot as plt
    NS, XL = "http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink"
    ET.register_namespace("", NS); ET.register_namespace("xlink", XL)
    SW, SH = fp.spec["page_in"]
    with matplotlib.rc_context({"svg.fonttype": "none", "svg.hashsalt": "figproof"}):
        b = io.BytesIO(); fig.savefig(b, format="svg")
        s = _slide_title(SW, SH, fp.title)
        t = io.BytesIO(); s.savefig(t, format="svg"); plt.close(s)
    body, page = ET.fromstring(b.getvalue()), ET.fromstring(t.getvalue())
    for el in page.iter():
        if el.get("id"):
            el.set("id", "slide_" + el.get("id"))
    vb = [float(v) for v in body.get("viewBox").split()]
    x_in, y_in, w_in = box
    g = ET.SubElement(page, f"{{{NS}}}g", {"id": "figproof_body",
                                            "transform": f"translate({x_in * 72:.3f} {y_in * 72:.3f}) scale({w_in * 72 / vb[2]:.6f})"})
    for child in list(body):
        g.append(child)
    ET.ElementTree(page).write(path, xml_declaration=True, encoding="utf-8")
