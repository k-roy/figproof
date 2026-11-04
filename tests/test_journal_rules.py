"""Planted-failure tests for the journal rules: the 6 to 7 pt type band, column widths, panel-letter case and size, the 1.5 pt stroke
cap, axes that reach 0, minor ticks and grid lines, symbol size, legend callout case, the supplementary mode, and canvas containment of panel letters.

Each test plants the failure and checks the MESSAGE, not just that something failed.

    pytest tests/                      # or: python tests/test_journal_rules.py
"""
import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import figproof as fp  # noqa: E402
from figproof import gates as Gt  # noqa: E402

OUT = tempfile.mkdtemp()
GREY = "#999999"
CAPS_LEGEND = "Figure 1 | Title. (A) One. (B) Two."
LOWER_LEGEND = "Figure 1 | Title. (a) One. (b) Two."


def _two_panels(fig, minor=False, grid=False, s=36, log=False):
    a = fig.fp.axes(0.66, 0.1, 0.9, 1.1, label="(A)")
    a.scatter([1, 2, 3], [1, 2, 3], s=s, color=GREY)
    a.set_xlabel("Dose (nM)")
    a.set_ylabel("Signal")
    a.set_ylim(0, 3.5)
    if log:
        a.set_xscale("log")
        if not minor:
            a.xaxis.set_minor_locator(NullLocator())
    else:
        a.set_xlim(0, 3.5)
    if grid:
        a.grid(True)
    b = fig.fp.axes(2.2, 0.1, 1.1, 1.1, label="(B)")
    b.plot([0, 1], [0, 1], lw=1.0, color=GREY)
    b.set_xlabel("Time (h)")
    b.set_ylabel("Signal")


def _letters(fig):
    return [(t.get_text(), t.get_fontsize(), t.get_fontweight()) for t in fig.texts if len(t.get_text()) == 1 and t.get_text().isalpha()]


def _run(mode="journal", width="single", letters=None, legend=CAPS_LEGEND, **kw):
    fig = fp.figure(mode, width=width, body_height=1.5, legend=legend, letters=letters)
    try:
        _two_panels(fig, **kw)
    except Exception:
        fp.discard(fig)
        raise
    r = fp.finish(fig, "t", OUT, strict=False)
    r["letters"] = _letters(fig)
    plt.close(fig)
    return r


def _p(r):
    return " | ".join(r["problems"])


def _w(r):
    return " | ".join(r["warnings"])


def test_clean_journal_figure_passes_with_capital_letters():
    r = _run(log=True)
    assert not r["problems"], _p(r)
    assert r["letters"] == [("A", 10.0, "bold"), ("B", 10.0, "bold")]
    assert "symbols as small" not in _w(r)


def test_minor_ticks_fail():
    r = _run(log=True, minor=True)
    assert "minor ticks on the x axis" in _p(r)


def test_grid_lines_fail():
    r = _run(grid=True)
    assert "grid lines" in _p(r)


def test_small_symbols_warn_but_do_not_block():
    r = _run(s=9)
    assert "symbols as small as 3.0 pt" in _w(r)
    assert not any("symbol" in p for p in r["problems"])


def test_lowercase_letters_on_request():
    r = _run(letters="lower", legend=LOWER_LEGEND)
    assert not r["problems"], _p(r)
    assert r["letters"] == [("a", 8.0, "bold"), ("b", 8.0, "bold")]


def test_bad_letter_case_rejected():
    with pytest.raises(ValueError, match="letters="):
        fp.figure("journal", width="single", body_height=1.0, legend="x", letters="title")


def test_letter_band_gate_checks_case():
    f = plt.figure(figsize=(3.5, 2))
    t = f.text(0.1, 0.9, "a", fontsize=10, fontweight="bold")
    assert any("10 pt bold capitals" in b for b in Gt.type_band_journal(f, [t], letter_pt=10.0, case="upper"))
    t.remove()
    t2 = f.text(0.5, 0.9, "A", fontsize=10, fontweight="bold")
    assert not Gt.type_band_journal(f, [t2], letter_pt=10.0, case="upper")
    plt.close(f)


def test_legend_callouts_in_the_other_case_warn():
    r = _run(legend=LOWER_LEGEND)
    assert "legend calls out (a)" in _w(r)


@pytest.mark.parametrize("width", ["third", "single", "onehalf", "twothirds", "double", "full", "57mm", "184mm"])
def test_journal_widths_accepted(width):
    fig = fp.figure("journal", width=width, body_height=1.0, legend="x")
    fp.discard(fig)
    plt.close(fig)


def test_off_grid_width_rejected():
    with pytest.raises(ValueError, match="allowed widths"):
        fp.figure("journal", width="60mm", body_height=1.0, legend="x")


def test_supplementary_mode_writes_jpeg():
    r = _run(mode="supplementary")
    assert not r["problems"], _p(r)
    assert any(f.endswith(".jpg") for f in r["files"])


def test_extended_mode_is_gone():
    with pytest.raises(KeyError, match="supplementary"):
        fp.figure("extended", width="single", body_height=1.0, legend="x")


def test_internal_mode_keeps_lowercase_and_its_footer_fits():
    legend = ("Figure 1 | Long legend that wraps over several lines to exercise the per-word page legend. (a) First panel, with "
              "enough words to reach the right margin and wrap at least once more. (b) Second panel, also described at length so the "
              "legend spans three or four lines across the full double-column width of the page.")
    fig = fp.figure("internal", width="double", body_height=1.5, legend=legend, provenance="synthetic data; footer test",
                    script="/some/long/path/to/NN_internal_figure_script.py")
    _two_panels(fig)
    r = fp.finish(fig, "t_internal", OUT, strict=False)
    assert not r["problems"], _p(r)
    assert [(a, b) for a, b, _ in _letters(fig)] == [("a", 11.0), ("b", 11.0)]
    assert "legend calls out" not in _w(r)
    plt.close(fig)


def test_linear_axis_that_misses_zero_fails_and_log_or_exempt_passes():
    fig = fp.figure("journal", width="single", body_height=1.5, legend=CAPS_LEGEND)
    _two_panels(fig)
    fig.axes[0].set_ylim(1, 3.5)
    r = fp.finish(fig, "t_zero", OUT, strict=False)
    assert "the y axis runs 1 to 3.5 and does not reach 0" in _p(r)
    plt.close(fig)
    fig = fp.figure("journal", width="single", body_height=1.5, legend=CAPS_LEGEND)
    _two_panels(fig)
    fig.axes[0].set_ylim(1, 3.5)
    fp.exempt(fig.axes[0], "a coordinate axis")
    r = fp.finish(fig, "t_zero_exempt", OUT, strict=False)
    assert "does not reach 0" not in _p(r)
    plt.close(fig)
    r = _run(log=True)                                   # the log x axis starts at 1, which is allowed
    assert "does not reach 0" not in _p(r)


def test_text_below_6_pt_fails():
    def small(fig):
        _two_panels(fig)
        fig.axes[1].text(0.1, 0.9, "n = 3", fontsize=5.5)
    fig = fp.figure("journal", width="single", body_height=1.5, legend=CAPS_LEGEND)
    small(fig)
    r = fp.finish(fig, "t_small", OUT, strict=False)
    assert "journal text must be 6 to 7 pt" in _p(r)
    plt.close(fig)


def test_stroke_over_1_5_pt_fails_and_png_is_600_dpi():
    from PIL import Image
    r = _run()
    png = [f for f in r["files"] if f.endswith(".png")][0]
    assert abs(Image.open(png).info["dpi"][0] - 600) < 1
    fig = fp.figure("journal", width="single", body_height=1.5, legend=CAPS_LEGEND)
    _two_panels(fig)
    fig.axes[1].plot([0, 1], [1, 0], lw=2.0, color=GREY)
    r = fp.finish(fig, "t_heavy", OUT, strict=False)
    assert "at most 1.5 pt" in _p(r)
    plt.close(fig)


def test_letter_past_the_left_edge_fails_containment():
    fig = fp.figure("journal", width="single", body_height=1.2, legend="x")
    a = fig.fp.axes(0.30, 0.1, 1.2, 1.0)
    a.set_ylabel("Signal")
    a.set_xlabel("Time (h)")
    b = fig.fp.axes(2.0, 0.1, 1.2, 1.0)
    b.set_xlabel("Time (h)")
    r = fp.finish(fig, "t_clip", OUT, strict=False)
    assert "extend beyond the canvas" in _p(r)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
