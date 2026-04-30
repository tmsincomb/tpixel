"""Integration tests: hiv_panel + render_panels with no-HxB2 alignments.

Exercises the full pipeline a user hits when they drop HxB2 from their
alignment and expect the region bar / NT ruler to keep rendering.

Fixtures live in ``tests/data/`` and are pre-stripped of HxB2 (only the
lineage ``_ref`` row plus samples). Dual-render visual comparisons
(`with HxB2` vs `without HxB2`) cannot be synthesised from these
fixtures alone — the bundled HxB2 has residues at columns where the
lineage has a gap, and those residues are lost when projecting back
into a no-HxB2 alignment, shifting AA numbering. For that reason the
tests below compare anchor mode against itself (CH505 vs SF162p3
fixtures) rather than reconstructing a synthetic HxB2 row.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tpixel.hiv import hiv_panel
from tpixel.renderer import render_panels

DATA_DIR = Path(__file__).resolve().parent / "data"
ANCHOR_FIXTURES = {
    "CH505": DATA_DIR / "CH505.aln.fasta",
    "SF162p3": DATA_DIR / "SF162p3.aln.fasta",
}


@pytest.mark.parametrize("lineage", ["CH505", "SF162p3"])
def test_hiv_panel_anchor_mode_produces_regions_and_ruler(lineage):
    """hiv_panel must produce non-empty regions + NT ruler without HxB2."""
    src = ANCHOR_FIXTURES[lineage]
    if not src.exists():
        pytest.skip(f"missing fixture {src}")

    panel = hiv_panel(str(src), ref_positions=[1], show_nt_ruler=True)

    assert panel.regions, "anchor mode must yield a non-empty region list"
    assert panel.nt_ruler_labels, "anchor mode must yield NT ruler labels"
    region_names = {r.name for r in panel.regions}
    assert {"SP", "C1", "V1", "V2", "C2", "V3", "C3", "V4", "C4", "V5", "C5", "gp41"} <= region_names


@pytest.mark.parametrize("lineage", ["CH505", "SF162p3"])
def test_anchor_mode_renders_to_png(output_dir, lineage):
    """End-to-end render: no-HxB2 fixture -> PNG file on disk."""
    src = ANCHOR_FIXTURES[lineage]
    if not src.exists():
        pytest.skip(f"missing fixture {src}")

    panel = hiv_panel(str(src), ref_positions=[1])
    out = output_dir / f"{lineage.lower()}_anchor_no_hxb2.png"
    if out.exists():
        out.unlink()
    render_panels([panel], str(out))
    assert out.exists() and out.stat().st_size > 0


def test_anchor_mode_variant_labels_emit():
    """--variant-labels in anchor mode must yield non-empty substitution labels."""
    src = ANCHOR_FIXTURES["CH505"]
    if not src.exists():
        pytest.skip(f"missing fixture {src}")

    panel = hiv_panel(str(src), ref_positions=[1], show_variant_labels=True)
    assert panel.extra_col_labels, "variant labels must be emitted in anchor mode"
    subs = [lbl for _, lbl in panel.extra_col_labels if not lbl.endswith("-")]
    assert subs, "anchor mode must emit substitution labels"


def test_anchor_mode_pngs_markers_present():
    """PNGS markers must persist in anchor mode."""
    src = ANCHOR_FIXTURES["CH505"]
    if not src.exists():
        pytest.skip(f"missing fixture {src}")

    panel = hiv_panel(str(src), ref_positions=[1])
    assert panel.markers, "PNGS markers must persist in anchor mode"


def test_dual_anchor_renders_for_visual_comparison(output_dir):
    """Render both tests/data fixtures side by side via the anchor pathway.

    Lands two PNGs in ``tests/output/`` so the user can confirm region
    bars and PNGS markers survive on both lineages without HxB2.
    """
    rendered = []
    for lineage, src in ANCHOR_FIXTURES.items():
        if not src.exists():
            continue
        panel = hiv_panel(str(src), ref_positions=[1])
        out = output_dir / f"dual_{lineage.lower()}_no_hxb2.png"
        if out.exists():
            out.unlink()
        render_panels([panel], str(out))
        assert out.exists() and out.stat().st_size > 0
        assert panel.regions, f"{lineage}: region header must persist"
        assert panel.markers, f"{lineage}: PNGS markers must persist"
        rendered.append(out)
    assert len(rendered) >= 1


def test_hiv_panel_explicit_anchor_kwargs():
    """anchor_id + anchor_lineage override the auto-detected default."""
    src = ANCHOR_FIXTURES["SF162p3"]
    if not src.exists():
        pytest.skip(f"missing fixture {src}")

    panel = hiv_panel(
        str(src),
        ref_positions=[1],
        anchor_id="SF162p3_ref",
        anchor_lineage="SF162p3",
    )
    assert panel.regions
