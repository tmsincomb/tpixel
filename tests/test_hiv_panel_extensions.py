"""Tests for hiv_panel extensions: region_palette, show_nt_ruler, V1/V2 merge.

Backs VAL-TPIXEL-001 (region_palette), VAL-TPIXEL-002 (nt_ruler),
VAL-TPIXEL-003 (v1/v2 merge) in the SHIV-Romy audit mission's validation
contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tpixel.hiv import hiv_panel
from tpixel.renderer import render_panels

# -- Fixtures -------------------------------------------------------------

# A 90-NT alignment covering HxB2 AA positions 1..30 (all SP region in canon
# boundaries, but we overlay custom V1/V2/V3 for the merge test via
# region_palette).  We then use amino-acid sequences for simplicity in the
# palette test by switching to AA seq_type.


def _aa_seqs() -> list[tuple[str, str]]:
    # 600-AA alignment spans into V3, V4, V5, and gp41 regions (gp41 starts at 512).
    hxb2 = "M" * 600
    ref = "M" * 600
    return [
        ("HxB2", hxb2),
        ("animal1_ref", ref),
        ("animal1_s1", "M" * 599 + "L"),
        ("animal2_s1", "L" + "M" * 599),
    ]


@pytest.fixture
def aa_hiv_fasta(tmp_path, write_fasta):
    return write_fasta(_aa_seqs(), name="aa_hiv.fasta")


# -- VAL-TPIXEL-001: region_palette override -----------------------------


class TestRegionPaletteOverride:
    def test_palette_overrides_single_region_color(self, aa_hiv_fasta):
        """Passing region_palette={'V3': '#00FF00'} colors V3 green."""
        panel = hiv_panel(
            str(aa_hiv_fasta),
            seq_type="AA",
            region_palette={"V3": "#00FF00"},
        )
        assert panel.regions is not None
        v3_regions = [r for r in panel.regions if "V3" in r.name]
        assert v3_regions, "V3 region must be present"
        for r in v3_regions:
            assert r.color.upper() == "#00FF00"

    def test_palette_default_unchanged(self, aa_hiv_fasta):
        """Not passing region_palette preserves default colors."""
        panel = hiv_panel(str(aa_hiv_fasta), seq_type="AA")
        assert panel.regions is not None
        # V3 default is "#BBDEFB"
        v3_regions = [r for r in panel.regions if "V3" in r.name]
        for r in v3_regions:
            assert r.color.upper() == "#BBDEFB"

    def test_palette_overrides_multiple_regions(self, aa_hiv_fasta):
        """Palette can override several regions at once."""
        panel = hiv_panel(
            str(aa_hiv_fasta),
            seq_type="AA",
            region_palette={
                "V3": "#11AA11",
                "V4": "#22BB22",
                "V5": "#33CC33",
                "gp41": "#FFEE88",
            },
        )
        colors = {r.name: r.color.upper() for r in panel.regions or []}
        assert colors.get("V3") == "#11AA11"
        assert colors.get("V4") == "#22BB22"
        assert colors.get("V5") == "#33CC33"
        assert colors.get("gp41") == "#FFEE88"

    def test_palette_unknown_region_silent(self, aa_hiv_fasta):
        """Palette entries for absent region names are ignored (no error)."""
        panel = hiv_panel(
            str(aa_hiv_fasta),
            seq_type="AA",
            region_palette={"NotARegion": "#FF00FF", "V3": "#00FF00"},
        )
        v3 = [r for r in panel.regions or [] if "V3" in r.name]
        assert v3[0].color.upper() == "#00FF00"


# -- VAL-TPIXEL-002: NT coordinate ruler ---------------------------------


class TestNTRuler:
    def test_show_nt_ruler_produces_nt_scale_labels(self, aa_hiv_fasta):
        """When show_nt_ruler=True, col_labels use NT coordinates."""
        panel = hiv_panel(
            str(aa_hiv_fasta),
            seq_type="AA",
            show_nt_ruler=True,
            nt_ruler_step=250,
        )
        # Label values are NT scale (multiples of 250): 250, 500, 750, ...
        label_values = [int(label) for _col, label in panel.col_labels]
        assert label_values, "NT ruler should produce tick labels"
        # Steps are multiples of nt_ruler_step
        for v in label_values:
            assert v % 250 == 0, f"NT label {v} is not a multiple of 250"
        # Span should reach at least 750 NT for a 600-AA (~1800 NT) alignment
        assert max(label_values) >= 750

    def test_show_nt_ruler_default_off(self, aa_hiv_fasta):
        """Default behavior: AA ticks, no NT ruler."""
        panel = hiv_panel(
            str(aa_hiv_fasta),
            seq_type="AA",
            tick_step=50,
        )
        label_values = [int(label) for _col, label in panel.col_labels]
        # AA tick values <= alignment length (600)
        assert max(label_values) <= 600

    def test_nt_ruler_renders_to_png(self, aa_hiv_fasta, output_dir):
        """End-to-end: NT ruler panel renders without error."""
        panel = hiv_panel(
            str(aa_hiv_fasta),
            seq_type="AA",
            show_nt_ruler=True,
            nt_ruler_step=250,
        )
        out = Path(output_dir) / "hiv_nt_ruler.png"
        render_panels([panel], str(out), dpi=100)
        assert out.exists()
        assert out.stat().st_size > 0
        # PNG is a valid image (Pillow can open it)
        with Image.open(out) as im:
            assert im.width > 0 and im.height > 0

    def test_show_nt_ruler_populates_top_ruler_field(self, aa_hiv_fasta):
        """``show_nt_ruler=True`` populates the dedicated ``nt_ruler_labels``
        Panel field that drives top-track rendering.

        Backs scrutiny-m2 blocker #2 (the renderer must use a field other
        than ``col_labels`` so it can render ABOVE the region bar without
        colliding with Layer 6's bottom x-axis).
        """
        panel = hiv_panel(
            str(aa_hiv_fasta),
            seq_type="AA",
            show_nt_ruler=True,
            nt_ruler_step=250,
        )
        assert panel.nt_ruler_labels is not None
        assert len(panel.nt_ruler_labels) >= 1
        # Values match the tick-value semantics (multiples of step).
        for _col, label in panel.nt_ruler_labels:
            assert int(label) % 250 == 0

    def test_show_nt_ruler_off_leaves_top_ruler_field_none(self, aa_hiv_fasta):
        """When ``show_nt_ruler=False`` the Panel has no top ruler band."""
        panel = hiv_panel(str(aa_hiv_fasta), seq_type="AA", tick_step=50)
        assert panel.nt_ruler_labels is None

    def test_nt_ruler_is_above_region_bar_in_rendered_png(
        self, aa_hiv_fasta, output_dir
    ):
        """Placement assertion: rendered PNG shows NT ruler pixels ABOVE the
        colored region bar, not below it.

        Strategy: render the panel, then sample a vertical column near the
        left edge of the image. Walk DOWN from y=0 and find:
          (1) the first coloured (non-white) pixel band — this is the NT
              ruler (black tick marks + dark numeric labels).
          (2) the first coloured pixel band that is distinctly a region
              color (one of the pastels from the palette / default V1 blue).

        The test asserts that band (1) appears at a smaller y-coordinate
        (i.e. HIGHER on the page) than band (2).
        """
        panel = hiv_panel(
            str(aa_hiv_fasta),
            seq_type="AA",
            show_nt_ruler=True,
            nt_ruler_step=250,
            # Use the SHIV palette so the region bar is clearly coloured.
            region_palette={
                "V3": "#B2D8B2",
                "V4": "#7FDBDA",
                "V5": "#C7A8E0",
                "gp41": "#FFE28A",
            },
        )
        out = Path(output_dir) / "hiv_nt_ruler_placement.png"
        render_panels([panel], str(out), dpi=150)

        with Image.open(out) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size

            # Probe several columns in the left 30% of the panel so we
            # have a good chance of crossing a tick mark AND a region-bar
            # swatch without being distracted by left-margin row labels.
            probe_xs = [
                int(width * 0.35),
                int(width * 0.40),
                int(width * 0.45),
                int(width * 0.50),
                int(width * 0.55),
            ]

            def _is_dark(px: tuple[int, int, int]) -> bool:
                # Dark = tick mark or numeric label glyph.
                return max(px) < 120

            def _is_region_color(px: tuple[int, int, int]) -> bool:
                # Coloured but NOT white and NOT near-black. The palette
                # hexes are all mid-value pastels (each channel is in the
                # ~120..255 range).
                r, g, b = px
                if r > 245 and g > 245 and b > 245:
                    return False  # white / background
                if max(r, g, b) < 140:
                    return False  # dark ink
                if abs(r - g) < 10 and abs(g - b) < 10:
                    return False  # grey (neutral / match color)
                return True

            ruler_ys: list[int] = []
            region_ys: list[int] = []

            for x in probe_xs:
                first_dark = None
                first_region = None
                for y in range(height):
                    px = rgb.getpixel((x, y))
                    if first_dark is None and _is_dark(px):
                        first_dark = y
                    if first_region is None and _is_region_color(px):
                        first_region = y
                    if first_dark is not None and first_region is not None:
                        break
                if first_dark is not None:
                    ruler_ys.append(first_dark)
                if first_region is not None:
                    region_ys.append(first_region)

            assert ruler_ys, (
                "Could not find any dark pixels (expected NT ruler tick / "
                "numeral ink) in the probed columns of the rendered PNG."
            )
            assert region_ys, (
                "Could not find any coloured region-bar pixels in the "
                "probed columns of the rendered PNG."
            )

            # The shallowest dark pixel (smallest y → highest on page)
            # must be strictly above the shallowest region-colour pixel.
            first_ruler_y = min(ruler_ys)
            first_region_y = min(region_ys)
            assert first_ruler_y < first_region_y, (
                f"NT ruler must render ABOVE the region bar. "
                f"first ruler-ink y={first_ruler_y}, "
                f"first region-colour y={first_region_y}."
            )


# -- VAL-TPIXEL-003: V1/V2 merge via shared palette color ----------------


class TestV1V2Merge:
    def test_v1_v2_same_color_merges_to_single_region(self, aa_hiv_fasta):
        """When palette pins V1 and V2 to the same color, they merge."""
        panel = hiv_panel(
            str(aa_hiv_fasta),
            seq_type="AA",
            region_palette={"V1": "#F8C8DC", "V2": "#F8C8DC"},
        )
        names = [r.name for r in panel.regions or []]
        # No separate V1 and V2 entries
        assert "V1" not in names, f"V1 should be merged, got regions: {names}"
        assert "V2" not in names, f"V2 should be merged, got regions: {names}"
        # Exactly one merged entry should appear
        merged = [r for r in panel.regions or [] if "V1" in r.name and "V2" in r.name]
        assert len(merged) == 1, (
            f"Expected exactly one merged V1/V2 region, got {[r.name for r in merged]}"
        )
        m = merged[0]
        assert m.color.upper() == "#F8C8DC"

    def test_v1_v2_merged_span_covers_both(self, aa_hiv_fasta):
        """Merged region span = union of V1 and V2 original spans."""
        # Reference: un-merged panel to capture the natural V1 and V2 spans.
        baseline = hiv_panel(str(aa_hiv_fasta), seq_type="AA")
        v1 = next(r for r in baseline.regions or [] if r.name == "V1")
        v2 = next(r for r in baseline.regions or [] if r.name == "V2")
        expected_start = min(v1.start, v2.start)
        expected_end = max(v1.end, v2.end)

        merged_panel = hiv_panel(
            str(aa_hiv_fasta),
            seq_type="AA",
            region_palette={"V1": "#F8C8DC", "V2": "#F8C8DC"},
        )
        merged = next(
            r for r in merged_panel.regions or []
            if "V1" in r.name and "V2" in r.name
        )
        assert merged.start == expected_start
        assert merged.end == expected_end

    def test_v1_v2_different_colors_not_merged(self, aa_hiv_fasta):
        """V1 and V2 with different colors remain separate regions."""
        panel = hiv_panel(
            str(aa_hiv_fasta),
            seq_type="AA",
            region_palette={"V1": "#F8C8DC", "V2": "#FF0000"},
        )
        names = [r.name for r in panel.regions or []]
        assert "V1" in names
        assert "V2" in names


# -- write_fasta fixture in conftest is tmp_path-scoped, but this test
# module's fixtures pass the path positionally, so we rely on the existing
# conftest write_fasta fixture.
