"""    import figproof as fp
    fig = fp.figure("internal", width="double", body_height=4.0, legend=LEGEND, script=__file__, provenance=PROV)
    ax = fig.fp.axes(0.6, 0.3, 2.6, 1.8)
    ...
    fp.finish(fig, "NN_slug_fig", outdir)
"""
from . import points, schematics, slides, tokens
from .core import GateError, data_text, discard, exempt, figure, finish, layer, no_letter

__all__ = ["figure", "finish", "discard", "layer", "exempt", "data_text", "no_letter", "GateError", "tokens", "points", "slides", "schematics", "MODES",
           "__version__"]
__version__ = "0.1.0"
MODES = ("internal", "collab", "journal", "slide", "supplementary")
