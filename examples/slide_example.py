"""A talk slide built with figproof on synthetic data.

    python examples/slide_example.py            # writes examples/out/slide_example_fig.png, an .svg and the speaker notes

Slide mode builds a 13.333 by 7.5 in page with a one-sentence claim as its title. The legend goes to the speaker notes, so every panel
names itself on the slide and panel letters are refused.
"""
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

import figproof as fp  # noqa: E402
from figproof import tokens as T  # noqa: E402

FOCAL, REFERENCE, NEUTRAL = T.colour("role.focal"), T.colour("role.reference"), T.colour("neutral.neutral")
LEGEND = ("Synthetic data. Left, share of reads from the dominant isoform per replicate. Right, RNA versus protein fold change for "
          "200 simulated genes, with isoform-switch genes highlighted.")

rng = np.random.default_rng(3)
fig = fp.figure("slide", width="double", body_height=3.4, legend=LEGEND,
                title="Treated cells shift isoform usage, and protein follows RNA only loosely.")

ax_a = fig.fp.axes(0.60, 0.35, 2.40, 2.70)
share = {"control": rng.normal(0.62, 0.04, 4), "treated": rng.normal(0.41, 0.05, 4)}
for i, (name, vals) in enumerate(share.items()):
    ax_a.bar(i, vals.mean(), width=0.6, color=REFERENCE if name == "control" else FOCAL)
    ax_a.scatter(np.full(4, i) + rng.uniform(-0.12, 0.12, 4), vals, s=12, color=NEUTRAL, zorder=3)
ax_a.set_xticks([0, 1])
ax_a.set_xticklabels(list(share))
ax_a.set_ylim(0, 0.8)
ax_a.set_ylabel("isoform share")
ax_a.set_xlabel("condition")
ax_a.set_title("dominant isoform share")

ax_b = fig.fp.axes(4.10, 0.35, 2.40, 2.70)
rna = rng.normal(0, 0.8, 200)
protein = 0.35 * rna + rng.normal(0, 0.5, 200)
switch = rng.random(200) < 0.1
ax_b.scatter(rna[~switch], protein[~switch], s=6, color=NEUTRAL)
ax_b.scatter(rna[switch], protein[switch], s=12, color=FOCAL)
ax_b.set_xlabel("RNA log2 FC")
ax_b.set_ylabel("protein log2 FC")
ax_b.set_title("RNA vs protein")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out, exist_ok=True)
print(fp.finish(fig, "slide_example_fig", out)["files"])
