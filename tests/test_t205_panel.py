"""Tests for the T205-4 NT fixture and a stacked CH505-over-T205 render.

T205-4.fasta is a 10-record nucleotide alignment whose ``_ref`` row is
``T250-4_ref`` (a known anchor lineage) but has no HxB2 row. Because anchor
mode currently rejects NT alignments, the plain ``fasta_panel`` is the only
supported path — that's the contract these tests pin down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tpixel import fasta_panel, hiv_panel, render_panels

DATA_DIR = Path(__file__).resolve().parent / "data"
T205_FASTA = DATA_DIR / "T205-4.fasta"
CH505_FASTA = DATA_DIR / "CH505.aln.fasta"


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"missing fixture {path}")


def test_fasta_panel_t205_builds():
    """Plain fasta_panel must load the NT T205-4 fixture cleanly."""
    _require(T205_FASTA)

    panel = fasta_panel(str(T205_FASTA))

    assert panel.total_cols == 2619
    assert panel.total_seqs == 9  # 10 records minus the primary ref
    assert panel.label == "T205-4"
    assert panel.ref_row[0] == "A"  # NT alphabet sanity check


def test_hiv_panel_t205_nt_raises():
    """Anchor mode + NT is documented as unsupported — keep it that way."""
    _require(T205_FASTA)

    with pytest.raises(ValueError) as exc:
        hiv_panel(str(T205_FASTA), ref_positions=[1])

    msg = str(exc.value)
    assert "Anchor mode" in msg
    assert "AA" in msg


def test_t205_renders_to_png(output_dir):
    """End-to-end: T205-4 fasta_panel renders to a non-empty PNG."""
    _require(T205_FASTA)

    panel = fasta_panel(str(T205_FASTA))
    out = output_dir / "t205_panel.png"
    if out.exists():
        out.unlink()

    render_panels([panel], str(out))

    assert out.exists()
    assert out.stat().st_size > 0


def test_stack_ch505_over_t205(output_dir):
    """Stack CH505 (AA hiv_panel) above T205-4 (NT fasta_panel) in one render."""
    _require(CH505_FASTA)
    _require(T205_FASTA)

    p_ch505 = hiv_panel(str(CH505_FASTA), ref_positions=[1])
    p_t205 = fasta_panel(str(T205_FASTA))

    panels = [p_ch505, p_t205]
    assert len(panels) == 2

    out = output_dir / "stacked_ch505_over_t205.png"
    if out.exists():
        out.unlink()

    render_panels(panels, str(out))

    assert out.exists()
    assert out.stat().st_size > 0
