"""Tests for patchworklib integration (plot_panel, panel_figsize, to_patchwork)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import patchworklib as pw  # noqa: E402

from tpixel.models import Marker, Panel, Region, SeqGroup
from tpixel.renderer import panel_figsize, plot_panel, to_patchwork


def _simple_panel() -> Panel:
    return Panel(
        label="ref",
        ref_row=list("ACGTACGT"),
        seq_rows=[("s1", list("ACCTACGT")), ("s2", list("ACGT-CGT"))],
        total_cols=8,
        col_labels=[(0, "1"), (4, "5")],
    )


def _roark_panel() -> Panel:
    ref_row = list("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT")
    aln_len = len(ref_row)
    groups = [
        SeqGroup("animal_1", [("s1", list("ACCTACGTACGTACGTACGTACGTACGTACGTACGTACGT"))]),
        SeqGroup("animal_2", [("s2", list("ACGTACGAACGTACGTACGTACGTACGTACGTACGTACGT"))]),
    ]
    regions = [
        Region("R1", 0, 15, "#BBDEFB"),
        Region("R2", 15, 30, "#EEEEEE"),
        Region("R3", 30, 40, "#F8BBD0"),
    ]
    markers = [Marker(col=5, label="M5"), Marker(col=25, label="M25")]
    return Panel(
        label="TestRef",
        ref_row=ref_row,
        seq_rows=[],
        total_cols=aln_len,
        col_labels=[(0, "1"), (20, "21")],
        regions=regions,
        markers=markers,
        groups=groups,
        title="Roark Panel",
    )


class TestPanelFigsize:
    def test_returns_tuple(self):
        w, h = panel_figsize(_simple_panel())
        assert isinstance(w, (int, float))
        assert isinstance(h, (int, float))

    def test_minimum_dimensions(self):
        w, h = panel_figsize(_simple_panel())
        assert w >= 6.0
        assert h >= 3.0

    def test_width_scales_with_columns(self):
        short = Panel("r", list("ACGT"), [("s", list("ACGT"))], 4, [])
        long = Panel("r", list("A") * 500, [("s", list("A") * 500)], 500, [])
        w_short, _ = panel_figsize(short)
        w_long, _ = panel_figsize(long)
        assert w_long > w_short


class TestPlotPanel:
    def test_returns_axes(self):
        ax = plot_panel(_simple_panel())
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_creates_figure_when_no_ax(self):
        ax = plot_panel(_simple_panel())
        assert ax.figure is not None
        plt.close(ax.figure)

    def test_draws_on_provided_axes(self):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        returned = plot_panel(_simple_panel(), ax=ax)
        assert returned is ax
        plt.close(fig)

    def test_roark_panel(self):
        ax = plot_panel(_roark_panel())
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)


class TestToPatchwork:
    def test_returns_brick(self):
        brick = to_patchwork(_simple_panel(), label="t1")
        assert isinstance(brick, pw.Brick)

    def test_unique_labels(self):
        b1 = to_patchwork(_simple_panel(), label="p1")
        b2 = to_patchwork(_simple_panel(), label="p2")
        assert b1._label != b2._label

    def test_horizontal_compose(self, output_dir):
        b1 = to_patchwork(_simple_panel(), label="h1")
        b2 = to_patchwork(_simple_panel(), label="h2")
        composed = b1 | b2
        out = output_dir / "patchwork_horizontal.png"
        composed.savefig(str(out), dpi=100)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_vertical_compose(self, output_dir):
        b1 = to_patchwork(_simple_panel(), label="v1")
        b2 = to_patchwork(_simple_panel(), label="v2")
        composed = b1 / b2
        out = output_dir / "patchwork_vertical.png"
        composed.savefig(str(out), dpi=100)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_roark_compose(self, output_dir):
        b1 = to_patchwork(_roark_panel(), label="r1")
        b2 = to_patchwork(_simple_panel(), label="r2")
        composed = b1 / b2
        out = output_dir / "patchwork_roark_composed.png"
        composed.savefig(str(out), dpi=100)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_panel_on_brick(self):
        brick = pw.Brick(figsize=(6, 3), label="ext")
        returned = plot_panel(_simple_panel(), ax=brick)
        assert returned is brick
