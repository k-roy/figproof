"""A single-column (89 mm) journal panel pair, built with figproof on synthetic data.

    python examples/journal_example.py          # writes examples/out/journal_example_fig.{pdf,png} and the legend as a .txt

Journal mode sets 6 to 7 pt type and refuses anything outside that band, draws 10 pt bold capital panel letters, and refuses minor
ticks and grid lines. The legend goes to its own file, as journals ask.
"""
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

import figproof as fp  # noqa: E402
from figproof import tokens as T  # noqa: E402

FOCAL, REFERENCE, NEUTRAL = T.colour("role.focal"), T.colour("role.reference"), T.colour("neutral.neutral")
LEGEND = ("Figure 2 | Dose response and time course for a synthetic reporter. (A) Reporter signal across six doses, mean of three "
          "replicates with each replicate as a point. (B) Signal over 24 h in treated and control cells.")

rng = np.random.default_rng(11)
fig = fp.figure("journal", width="single", body_height=1.55, legend=LEGEND)

ax_a = fig.fp.axes(0.62, 0.12, 0.92, 1.15, label="(A)")
dose = np.array([0, 1, 3, 10, 30, 100])
resp = 1 / (1 + (8 / np.maximum(dose, 0.1)) ** 1.2)
for d, r in zip(dose, resp):
    reps = r + rng.normal(0, 0.04, 3)
    ax_a.scatter(np.full(3, d + 1), reps, s=36, color=NEUTRAL, zorder=3)
ax_a.plot(dose + 1, resp, color=FOCAL, lw=1.0)
ax_a.set_xscale("log")
ax_a.xaxis.set_minor_locator(NullLocator())    # a log axis adds minor ticks; journal figures use major ticks only
ax_a.set_xlabel("Dose + 1 (nM)")
ax_a.set_ylabel("Reporter signal")

ax_b = fig.fp.axes(2.16, 0.12, 1.18, 1.15, label="(B)")
t = np.linspace(0, 24, 25)
ax_b.plot(t, 1 - np.exp(-t / 6) + rng.normal(0, 0.02, t.size), color=FOCAL, lw=1.0, label="Treated")
ax_b.plot(t, 0.15 + rng.normal(0, 0.02, t.size), color=REFERENCE, lw=1.0, label="Control")
ax_b.set_xlabel("Time (h)")
ax_b.set_ylabel("Reporter signal")
ax_b.set_ylim(-0.05, 1.45)
ax_b.legend(frameon=False, loc="upper left", ncols=2)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out, exist_ok=True)
print(fp.finish(fig, "journal_example_fig", out)["files"])
