"""Tests for wildtype+pos+mutation variant labels vs HxB2."""

from __future__ import annotations

import pytest

from tpixel.hiv import hiv_panel
from tpixel.hxb2 import build_hxb2_map, hxb2_variant_labels


class TestHxb2VariantLabelsHelper:
    def test_single_aa_substitution(self):
        """Query differs by one AA → one label in K+pos+M form."""
        hxb2 = "MKRVK"
        query = "MKEVK"
        m = build_hxb2_map(
            [("HxB2", hxb2), ("lin_ref", query)], hxb2_id="HxB2", seq_type="AA"
        )
        out = hxb2_variant_labels(list(hxb2), list(query), m, seq_type="AA")
        assert out == [(2, "R3E")]

    def test_identical_rows_produce_no_labels(self):
        hxb2 = "MKRVK"
        m = build_hxb2_map(
            [("HxB2", hxb2), ("lin_ref", hxb2)], hxb2_id="HxB2", seq_type="AA"
        )
        assert hxb2_variant_labels(list(hxb2), list(hxb2), m, seq_type="AA") == []

    def test_deletion_emits_dash_mutation(self):
        """Query gap at a position where HxB2 has a residue → 'K3-'."""
        hxb2 = "MKRVK"
        query = "MK-VK"
        m = build_hxb2_map(
            [("HxB2", hxb2), ("lin_ref", query)], hxb2_id="HxB2", seq_type="AA"
        )
        out = hxb2_variant_labels(list(hxb2), list(query), m, seq_type="AA")
        assert out == [(2, "R3-")]

    def test_insertion_relative_to_hxb2_is_skipped(self):
        """HxB2 gap column has no stable position → no label emitted."""
        # HxB2 gap at col 2; query has residue there (an insertion).
        hxb2 = "MK-VK"
        query = "MKRVK"
        m = build_hxb2_map(
            [("HxB2", hxb2), ("lin_ref", query)], hxb2_id="HxB2", seq_type="AA"
        )
        out = hxb2_variant_labels(list(hxb2), list(query), m, seq_type="AA")
        assert out == []

    def test_nt_mode_uses_nucleotide_positions(self):
        """NT mode numbers positions by 1-based NT counter, not codon."""
        # 6 NT = 2 codons. Change col 4 (pos 5) from A→T.
        hxb2 = "ATGAAA"
        query = "ATGATA"
        m = build_hxb2_map(
            [("HxB2", hxb2), ("lin_ref", query)], hxb2_id="HxB2", seq_type="NT"
        )
        out = hxb2_variant_labels(list(hxb2), list(query), m, seq_type="NT")
        assert out == [(4, "A5T")]

    def test_multiple_variants_preserve_column_order(self):
        hxb2 = "MKRVKE"
        query = "MEEVKD"
        m = build_hxb2_map(
            [("HxB2", hxb2), ("lin_ref", query)], hxb2_id="HxB2", seq_type="AA"
        )
        out = hxb2_variant_labels(list(hxb2), list(query), m, seq_type="AA")
        assert out == [(1, "K2E"), (2, "R3E"), (5, "E6D")]


class TestHivPanelVariantLabelsWiring:
    def _fasta(self, tmp_path, write_fasta):
        return write_fasta(
            [
                ("HxB2",        "M" * 600),
                ("animal1_ref", "M" * 168 + "E" + "M" * 431),   # K169E equivalent: M vs E at pos 169
                ("animal1_s1",  "M" * 600),
            ],
            name="variant.fasta",
        )

    def test_flag_off_leaves_extra_col_labels_unset(self, tmp_path, write_fasta):
        panel = hiv_panel(
            str(self._fasta(tmp_path, write_fasta)),
            seq_type="AA",
            show_variant_labels=False,
        )
        assert panel.extra_col_labels is None

    def test_flag_on_populates_variant_labels(self, tmp_path, write_fasta):
        panel = hiv_panel(
            str(self._fasta(tmp_path, write_fasta)),
            seq_type="AA",
            show_variant_labels=True,
        )
        assert panel.extra_col_labels is not None
        # One variant: HxB2 M vs lineage E at column 168 (AA pos 169).
        assert panel.extra_col_labels == [(168, "M169E")]

    def test_renderer_consumes_variant_labels_without_error(
        self, tmp_path, write_fasta, output_dir
    ):
        """Full render path including Layer 6b must succeed."""
        from tpixel.renderer import render_panels

        panel = hiv_panel(
            str(self._fasta(tmp_path, write_fasta)),
            seq_type="AA",
            show_variant_labels=True,
        )
        out = output_dir / "variant_labels.png"
        render_panels([panel], out, dpi=72)
        assert out.exists() and out.stat().st_size > 0
