"""A two-panel RNA figure in collaborator mode, built with figproof on synthetic data.

    python examples/isoform_example.py          # writes examples/out/isoform_example_fig.png

finish() runs every gate (type and stroke floors, overlaps, containment, colour meanings) and refuses to write a figure that fails.
"""
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

import figproof as fp  # noqa: E402
from figproof import tokens as T  # noqa: E402

FOCAL, REFERENCE, NEUTRAL = T.colour("role.focal"), T.colour("role.reference"), T.colour("neutral.neutral")
LEGEND = ("Figure 1 | Isoform usage and protein output for a synthetic gene set. (a) Share of reads from the dominant isoform in "
          "each condition, mean of four replicates with each replicate as a point. (b) RNA versus protein log2 fold change for 200 "
          "simulated genes; genes with an isoform switch are highlighted.")

rng = np.random.default_rng(7)
fig = fp.figure("collab", width="double", body_height=2.3, legend=LEGEND)

ax_a = fig.fp.axes(0.60, 0.30, 2.40, 1.70, label="(a)")
share = {"control": rng.normal(0.62, 0.04, 4), "treated": rng.normal(0.41, 0.05, 4)}
for i, (name, vals) in enumerate(share.items()):
    ax_a.bar(i, vals.mean(), width=0.6, color=REFERENCE if name == "control" else FOCAL)
    ax_a.scatter(np.full(4, i) + rng.uniform(-0.12, 0.12, 4), vals, s=10, color=NEUTRAL, zorder=3)
ax_a.set_xticks([0, 1])
ax_a.set_xticklabels(list(share))
ax_a.set_ylim(0, 0.8)
ax_a.set_ylabel("dominant isoform share")

ax_b = fig.fp.axes(4.10, 0.30, 2.40, 1.70, label="(b)")
rna = rng.normal(0, 0.8, 200)
protein = 0.35 * rna + rng.normal(0, 0.5, 200)
switch = rng.random(200) < 0.1
ax_b.scatter(rna[~switch], protein[~switch], s=6, color=NEUTRAL)
ax_b.scatter(rna[switch], protein[switch], s=10, color=FOCAL)
ax_b.axhline(0, color=NEUTRAL, lw=0.6, zorder=0)
ax_b.axvline(0, color=NEUTRAL, lw=0.6, zorder=0)
ax_b.set_xlabel("RNA log2 fold change")
ax_b.set_ylabel("protein log2 fold change")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out, exist_ok=True)
result = fp.finish(fig, "isoform_example_fig", out)
print(result["files"])
