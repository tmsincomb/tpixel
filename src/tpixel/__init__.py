"""tpixel — Pixel-block alignment viewer for hundreds of sequences."""

from importlib.metadata import PackageNotFoundError, version

from tpixel.fasta import fasta_panel, read_fasta
from tpixel.hiv import hiv_panel
from tpixel.models import Marker, Panel, Region, SeqGroup
from tpixel.renderer import panel_figsize, plot_panel, render_panels, to_patchwork

try:
    __version__ = version("tpixel")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Marker",
    "Panel",
    "Region",
    "SeqGroup",
    "fasta_panel",
    "hiv_panel",
    "panel_figsize",
    "plot_panel",
    "read_fasta",
    "render_panels",
    "to_patchwork",
]
