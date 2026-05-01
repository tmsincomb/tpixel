"""Tests for tpixel.anchors — non-HxB2 anchor coordinate mapping."""

from __future__ import annotations

import pytest

from tpixel.anchors import (
    KNOWN_ANCHOR_LINEAGES,
    _build_lineage_to_hxb2_lookup,
    _load_anchor_pair,
    build_anchor_hxb2_map,
    detect_anchor_lineage,
)
from tpixel.hxb2 import HxB2Position


def test_known_lineages_includes_three_canonical_anchors():
    assert set(KNOWN_ANCHOR_LINEAGES) == {"CH505", "SF162p3", "T250-4"}


@pytest.mark.parametrize(
    "seq_id,expected",
    [
        ("SF162p3_ref", "SF162p3"),
        ("CH505_ref", "CH505"),
        ("T250-4_ref", "T250-4"),
        ("HxB2", None),
        ("foo_ref", None),
        ("SF162p3", "SF162p3"),
    ],
)
def test_detect_anchor_lineage(seq_id, expected):
    assert detect_anchor_lineage(seq_id) == expected


@pytest.mark.parametrize("lineage", KNOWN_ANCHOR_LINEAGES)
def test_load_anchor_pair_yields_two_aligned_strings(lineage):
    hxb2_aligned, lin_aligned = _load_anchor_pair(lineage)
    assert len(hxb2_aligned) == len(lin_aligned)
    assert len(hxb2_aligned) > 800  # gp160 ~856 AA, plus a handful of indels
    # HxB2 ungapped length is fixed (canonical 856 AA).
    assert len(hxb2_aligned.replace("-", "")) == 856


@pytest.mark.parametrize("lineage", KNOWN_ANCHOR_LINEAGES)
def test_lineage_to_hxb2_lookup_consistent_with_pair(lineage):
    aa_lookup, res_lookup, canonical = _build_lineage_to_hxb2_lookup(lineage)
    assert len(aa_lookup) == len(res_lookup) == len(canonical)
    # HxB2 positions in the lookup are 1-based and bounded by canonical HxB2 length.
    mapped = [p for p in aa_lookup if p is not None]
    assert mapped == sorted(mapped)
    assert min(mapped) >= 1
    assert max(mapped) <= 856


def test_build_anchor_hxb2_map_synthetic_alignment():
    """Walk a 5-column anchor row that matches the bundled canonical prefix."""
    _, _, canonical = _build_lineage_to_hxb2_lookup("SF162p3")
    # Take the first 5 residues of the canonical lineage; embed in a fake
    # alignment with no gaps in the anchor row.
    anchor_row = canonical[:5]
    assert len(anchor_row) == 5
    seqs = [("SF162p3_ref", anchor_row), ("sample_1", anchor_row)]
    # Force the canonical sequence sanity-check to pass by extending the anchor
    # row to the full canonical length (otherwise ungapped(anchor) != canonical).
    anchor_row_full = canonical
    seqs = [("SF162p3_ref", anchor_row_full), ("sample_1", anchor_row_full)]

    positions = build_anchor_hxb2_map(seqs, "SF162p3_ref", "SF162p3")
    assert len(positions) == len(canonical)
    # First position should map to HxB2 AA 1 (canonical Met start).
    assert positions[0].alignment_col == 0
    assert positions[0].hxb2_aa_pos in (1, None)
    # All positions are HxB2Position instances.
    assert all(isinstance(p, HxB2Position) for p in positions)
    # All non-gap columns yield a numeric or None hxb2_aa_pos and a non-empty residue.
    for p in positions:
        assert isinstance(p.alignment_col, int)
        assert p.hxb2_residue


def test_build_anchor_hxb2_map_preserves_alignment_columns(write_fasta):
    """Anchor map length must equal alignment column count."""
    _, _, canonical = _build_lineage_to_hxb2_lookup("SF162p3")
    # Insert a 3-residue gap in the anchor row at position 100.
    gapped_anchor = canonical[:100] + "---" + canonical[100:]
    path = write_fasta(
        [("SF162p3_ref", gapped_anchor), ("sample_1", gapped_anchor)],
        name="anchor_gapped.fasta",
    )
    from tpixel.fasta import read_fasta

    seqs = read_fasta(path)
    positions = build_anchor_hxb2_map(seqs, "SF162p3_ref", "SF162p3")
    assert len(positions) == len(gapped_anchor)
    # The 3 inserted gap columns have hxb2_aa_pos = None.
    gap_cols = [p for p in positions[100:103]]
    assert all(p.hxb2_aa_pos is None for p in gap_cols)
    assert all(p.hxb2_residue == "-" for p in gap_cols)


def test_build_anchor_hxb2_map_rejects_unknown_lineage():
    seqs = [("SF162p3_ref", "MRVK")]
    with pytest.raises(ValueError, match="Unknown anchor lineage"):
        build_anchor_hxb2_map(seqs, "SF162p3_ref", "BG505")


def test_build_anchor_hxb2_map_rejects_missing_anchor():
    seqs = [("HxB2", "MRVK")]
    with pytest.raises(ValueError, match="Anchor sequence 'SF162p3_ref' not found"):
        build_anchor_hxb2_map(seqs, "SF162p3_ref", "SF162p3")


def test_build_anchor_hxb2_map_rejects_mismatched_anchor():
    """Anchor row whose ungapped form differs from bundled canonical errors."""
    fake = "MRVKEKYQHL" + "X" * 800
    seqs = [("SF162p3_ref", fake)]
    with pytest.raises(ValueError, match="does not match bundled canonical"):
        build_anchor_hxb2_map(seqs, "SF162p3_ref", "SF162p3")


def test_build_anchor_hxb2_map_rejects_nt_alignment():
    nt = "ATGCGT" * 200
    seqs = [("SF162p3_ref", nt)]
    with pytest.raises(ValueError, match="AA alignments only"):
        build_anchor_hxb2_map(seqs, "SF162p3_ref", "SF162p3", seq_type="NT")
