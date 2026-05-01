"""Tests for tpixel pixel-block rendering."""

from __future__ import annotations

import random

from tpixel.fasta import fasta_panel, read_fasta
from tpixel.models import Marker, Panel, Region, SeqGroup
from tpixel.renderer import render_panels


class TestReadFasta:
    def test_roundtrip(self, write_fasta):
        seqs = [("ref", "ACGTACGT"), ("s1", "ACCTACGT")]
        path = write_fasta(seqs)
        result = read_fasta(path)
        assert len(result) == 2
        assert result[0] == ("ref", "ACGTACGT")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.fasta"
        path.write_text("")
        assert read_fasta(str(path)) == []


class TestFastaPanel:
    def test_panel_shape(self, write_fasta):
        seqs = [("ref", "ACGTACGT"), ("s1", "ACCTACGT"), ("s2", "ACGTACGA")]
        path = write_fasta(seqs)
        panel = fasta_panel(path)
        assert panel.total_cols == 8
        assert len(panel.ref_row) == 8
        assert len(panel.seq_rows) == 2

    def test_column_slicing(self, write_fasta):
        seqs = [("ref", "ACGTACGTACGT"), ("s1", "ACCTACGTACGT")]
        path = write_fasta(seqs)
        panel = fasta_panel(path, col_start=2, col_end=5)
        assert panel.total_cols == 4
        assert panel.ref_row == ["C", "G", "T", "A"]


class TestRenderPanels:
    def test_basic_pixel_render(self, write_fasta, output_dir):
        seqs = [("ref", "ACGTACGT"), ("s1", "ACCTACGT"), ("s2", "ACGTACGA")]
        path = write_fasta(seqs)
        panel = fasta_panel(path)
        out = output_dir / "basic_pixel.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_large_alignment(self, write_fasta, output_dir):
        """Render 200 sequences to verify it scales."""
        random.seed(42)
        ref = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        lines = [("ref", ref)]
        for i in range(200):
            mutant = list(ref)
            for j in random.sample(range(len(ref)), k=8):
                mutant[j] = random.choice([b for b in "ACGT" if b != ref[j]])
            for j in random.sample(range(len(ref)), k=3):
                mutant[j] = "-"
            lines.append((f"seq_{i:03d}", "".join(mutant)))
        path = write_fasta(lines)
        panel = fasta_panel(path)
        out = output_dir / "large_200_seqs.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_stacked_panels(self, write_fasta, output_dir):
        f1 = write_fasta(
            [("ref", "ACGTACGT"), ("s1", "ACCTACGT")], "group1.fasta"
        )
        f2 = write_fasta(
            [("ref", "ACGTACGT"), ("s2", "ACGTACGA")], "group2.fasta"
        )
        p1 = fasta_panel(f1)
        p2 = fasta_panel(f2)
        out = output_dir / "stacked_pixel.png"
        render_panels([p1, p2], str(out), dpi=150)
        assert out.exists()

    def test_custom_cell_size(self, write_fasta, output_dir):
        seqs = [("ref", "ACGTACGT"), ("s1", "ACCTACGT")]
        path = write_fasta(seqs)
        panel = fasta_panel(path)
        out = output_dir / "custom_cell.png"
        render_panels([panel], str(out), dpi=150, cell=0.05)
        assert out.exists()

    def test_all_gaps(self, write_fasta, output_dir):
        seqs = [("ref", "ACGT"), ("s1", "----")]
        path = write_fasta(seqs)
        panel = fasta_panel(path)
        out = output_dir / "all_gaps_pixel.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()

    def test_all_matches(self, write_fasta, output_dir):
        seqs = [("ref", "ACGTACGT"), ("s1", "ACGTACGT"), ("s2", "ACGTACGT")]
        path = write_fasta(seqs)
        panel = fasta_panel(path)
        out = output_dir / "all_matches_pixel.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()

    def test_output_formats(self, write_fasta, output_dir):
        seqs = [("ref", "ACGT"), ("s1", "ACGA")]
        path = write_fasta(seqs)
        panel = fasta_panel(path)
        for ext in ["svg", "pdf"]:
            out = output_dir / f"pixel_format.{ext}"
            render_panels([panel], str(out), dpi=150)
            assert out.exists()
            assert out.stat().st_size > 0


class TestRoarkLayout:
    """Test the full 7-layer Roark layout with regions, markers, groups."""

    def _make_roark_panel(self) -> Panel:
        """Build a panel that exercises all 7 layers."""
        random.seed(99)
        ref = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        ref_row = list(ref)
        aln_len = len(ref)

        groups = []
        for g in range(3):
            seqs = []
            for i in range(10):
                mutant = list(ref)
                for j in random.sample(range(aln_len), k=5):
                    mutant[j] = random.choice([b for b in "ACGT-" if b != ref[j]])
                seqs.append((f"g{g}_s{i}", mutant))
            groups.append(SeqGroup(name=f"animal_{g}", seqs=seqs))

        regions = [
            Region("R1", 0, 10, "#BBDEFB"),
            Region("R2", 10, 25, "#EEEEEE"),
            Region("R3", 25, 40, "#F8BBD0"),
        ]

        markers = [
            Marker(col=5, label="M5"),
            Marker(col=15, label="M15"),
            Marker(col=30, label="M30"),
        ]

        col_labels = [(i, str(i + 1)) for i in range(0, aln_len, 10)]

        return Panel(
            label="TestRef",
            ref_row=ref_row,
            seq_rows=[],
            total_cols=aln_len,
            col_labels=col_labels,
            regions=regions,
            markers=markers,
            marker_color="#4CAF50",
            groups=groups,
        )

    def test_full_roark_png(self, output_dir):
        panel = self._make_roark_panel()
        out = output_dir / "roark_full.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_full_roark_pdf(self, output_dir):
        panel = self._make_roark_panel()
        out = output_dir / "roark_full.pdf"
        render_panels([panel], str(out), dpi=300)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_regions_only(self, output_dir):
        """Panel with regions but no markers or groups."""
        ref_row = list("ACGTACGT")
        regions = [
            Region("A", 0, 4, "#BBDEFB"),
            Region("B", 4, 8, "#F8BBD0"),
        ]
        panel = Panel(
            label="ref",
            ref_row=ref_row,
            seq_rows=[("s1", list("ACCTACGT"))],
            total_cols=8,
            col_labels=[(0, "1")],
            regions=regions,
        )
        out = output_dir / "regions_only.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()

    def test_markers_only(self, output_dir):
        """Panel with markers but no regions or groups."""
        ref_row = list("ACGTACGT")
        markers = [Marker(col=2, label="X2"), Marker(col=5, label="X5")]
        panel = Panel(
            label="ref",
            ref_row=ref_row,
            seq_rows=[("s1", list("ACCTACGT"))],
            total_cols=8,
            col_labels=[(0, "1")],
            markers=markers,
        )
        out = output_dir / "markers_only.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()

    def test_grouped_sequences(self, output_dir):
        """Panel with grouped sequences but no regions or markers."""
        ref_row = list("ACGTACGT")
        groups = [
            SeqGroup("group_A", [("s1", list("ACCTACGT")), ("s2", list("ACGTACGA"))]),
            SeqGroup("group_B", [("s3", list("A-GTACGT"))]),
        ]
        panel = Panel(
            label="ref",
            ref_row=ref_row,
            seq_rows=[],
            total_cols=8,
            col_labels=[(0, "1")],
            groups=groups,
        )
        out = output_dir / "grouped_seqs.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()

    def test_large_roark(self, output_dir):
        """Stress test: 500-col alignment, 5 groups, 200 seqs total."""
        random.seed(77)
        ref = "".join(random.choices("ACGT", k=500))
        ref_row = list(ref)

        groups = []
        for g in range(5):
            seqs = []
            for i in range(40):
                mutant = list(ref)
                for j in random.sample(range(500), k=30):
                    mutant[j] = random.choice([b for b in "ACGT-" if b != ref[j]])
                seqs.append((f"g{g}_s{i}", mutant))
            groups.append(SeqGroup(name=f"animal_{g}", seqs=seqs))

        regions = [
            Region("V1", 0, 100, "#BBDEFB"),
            Region("C1", 100, 200, "#EEEEEE"),
            Region("V2", 200, 300, "#BBDEFB"),
            Region("C2", 300, 400, "#EEEEEE"),
            Region("V3", 400, 500, "#F8BBD0"),
        ]

        markers = [Marker(col=i, label=f"N{i}") for i in range(10, 500, 40)]

        col_labels = [(i, str(i + 1)) for i in range(0, 500, 50)]

        panel = Panel(
            label="StressRef",
            ref_row=ref_row,
            seq_rows=[],
            total_cols=500,
            col_labels=col_labels,
            regions=regions,
            markers=markers,
            groups=groups,
        )
        out = output_dir / "roark_stress.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()
        assert out.stat().st_size > 0


class TestTitleAndStats:
    """Test that title only appears when explicitly set, and stats appear in legend."""

    def test_no_title_by_default(self, output_dir):
        """Panel renders without title when title is not set."""
        ref_row = list("ACGTACGT")
        groups = [
            SeqGroup("sample_A", [("s1", list("ACCTACGT")), ("s2", list("ACGTACGA"))]),
            SeqGroup("sample_B", [("s3", list("A-GTACGT"))]),
        ]
        panel = Panel(
            label="ref",
            ref_row=ref_row,
            seq_rows=[],
            total_cols=8,
            col_labels=[(0, "1")],
            groups=groups,
        )
        assert panel.title is None
        out = output_dir / "no_title.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_title_when_set(self, output_dir):
        """Panel renders with title when explicitly provided."""
        ref_row = list("ACGTACGT")
        groups = [
            SeqGroup("sample_A", [("s1", list("ACCTACGT"))]),
            SeqGroup("sample_B", [("s2", list("A-GTACGT"))]),
        ]
        panel = Panel(
            label="ref",
            ref_row=ref_row,
            seq_rows=[],
            total_cols=8,
            col_labels=[(0, "1")],
            groups=groups,
            title="My Custom Title",
        )
        assert panel.title == "My Custom Title"
        out = output_dir / "with_title.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_hiv_panel_no_auto_title(self, tmp_path, output_dir):
        """HIV panel should not generate an auto title."""
        from tpixel.hiv import hiv_panel

        # Build a minimal HIV-like FASTA
        hxb2 = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        ref = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        s1 = "ACCTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        s2 = "ACGTACGAACGTACGTACGTACGTACGTACGTACGTACGT"

        fasta_path = tmp_path / "hiv_test.fasta"
        lines = []
        for name, seq in [("HxB2", hxb2), ("animal1_ref", ref), ("animal1_s1", s1), ("animal2_s1", s2)]:
            lines.append(f">{name}\n{seq}\n")
        fasta_path.write_text("".join(lines))

        panel = hiv_panel(str(fasta_path))
        assert panel.title is None

        # Explicitly set title to simulate --title
        panel.title = "Custom HIV Title"
        out = output_dir / "hiv_custom_title.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()
        assert out.stat().st_size > 0


class TestPanel:
    def test_defaults(self):
        p = Panel("test", ["A", "C"], [], 2, [], ins_columns=None)
        assert p.ins_columns == set()
        assert p.label == "test"

    def test_total_seqs_flat(self):
        p = Panel("t", ["A"], [("s1", ["A"]), ("s2", ["A"])], 1, [])
        assert p.total_seqs == 2

    def test_total_seqs_grouped(self):
        g = [SeqGroup("g1", [("s1", ["A"])]), SeqGroup("g2", [("s2", ["A"]), ("s3", ["A"])])]
        p = Panel("t", ["A"], [], 1, [], groups=g)
        assert p.total_seqs == 3

    def test_effective_groups_from_flat(self):
        p = Panel("t", ["A"], [("s1", ["A"])], 1, [])
        eg = p.effective_groups
        assert len(eg) == 1
        assert eg[0].name == ""
        assert len(eg[0].seqs) == 1

    def test_seq_type_nt(self):
        p = Panel("t", list("ACGTACGT"), [], 8, [])
        assert p.seq_type == "NT"

    def test_seq_type_aa(self):
        p = Panel("t", list("MWLKFHRD"), [], 8, [])
        assert p.seq_type == "AA"

    def test_seq_type_nt_with_gaps(self):
        p = Panel("t", list("ACG-T.NU"), [], 8, [])
        assert p.seq_type == "NT"


class TestHxB2NTMapping:
    """Test nucleotide-aware HxB2 coordinate mapping."""

    def test_nt_codon_aa_positions(self):
        """9-NT HxB2 seq → AA positions [1,1,1,2,2,2,3,3,3]."""
        from tpixel.hxb2 import build_hxb2_map

        seqs = [("HxB2", "ACGACGACG")]
        hxb2_map = build_hxb2_map(seqs, seq_type="NT")
        aa_positions = [p.hxb2_aa_pos for p in hxb2_map]
        assert aa_positions == [1, 1, 1, 2, 2, 2, 3, 3, 3]

    def test_nt_gaps_skipped(self):
        """Gaps in the HxB2 NT sequence get None for aa_pos."""
        from tpixel.hxb2 import build_hxb2_map

        seqs = [("HxB2", "ACG---ACG")]
        hxb2_map = build_hxb2_map(seqs, seq_type="NT")
        aa_positions = [p.hxb2_aa_pos for p in hxb2_map]
        assert aa_positions == [1, 1, 1, None, None, None, 2, 2, 2]

    def test_nt_autodetect(self):
        """Auto-detect NT from HxB2 sequence characters."""
        from tpixel.hxb2 import build_hxb2_map

        # 9 NTs → 3 full codons, aa_pos = (nt_counter-1)//3 + 1
        seqs = [("HxB2", "ACGTACGTA")]
        hxb2_map = build_hxb2_map(seqs, seq_type=None)
        # Should auto-detect as NT and assign codon-based positions
        aa_positions = [p.hxb2_aa_pos for p in hxb2_map]
        assert aa_positions == [1, 1, 1, 2, 2, 2, 3, 3, 3]

    def test_aa_mode_unchanged(self):
        """AA mode still counts one residue = one AA position."""
        from tpixel.hxb2 import build_hxb2_map

        seqs = [("HxB2", "MWLK")]
        hxb2_map = build_hxb2_map(seqs, seq_type="AA")
        aa_positions = [p.hxb2_aa_pos for p in hxb2_map]
        assert aa_positions == [1, 2, 3, 4]

    def test_region_boundaries_at_codon_edges(self):
        """Regions assigned via AA position from codon-level counting."""
        from tpixel.hxb2 import build_hxb2_map

        # Build a 90-NT HxB2 sequence → 30 AA positions → all in SP region (1-30)
        nt_seq = "ACG" * 30  # 30 codons = AA positions 1-30
        seqs = [("HxB2", nt_seq)]
        hxb2_map = build_hxb2_map(seqs, seq_type="NT")
        # Last three NTs map to AA pos 30 → SP region
        for p in hxb2_map[-3:]:
            assert p.region == "SP"


class TestPNGSFromNT:
    """Test PNGS detection from nucleotide sequences."""

    def test_translate_basic(self):
        from tpixel.pngs import translate

        # ATG=M, AAT=N, GCT=A, TCT=S
        assert translate("ATGAATGCTTCT") == "MNAS"

    def test_translate_with_ambiguity(self):
        from tpixel.pngs import translate

        # NNN is ambiguous → X
        assert translate("ATGNNN") == "MX"

    def test_find_pngs_markers_nt(self):
        """NXS motif: AAT(N) GCT(A) TCT(S) → PNGS at AA pos 0."""
        from tpixel.pngs import find_pngs_markers_nt

        # Protein: N A S → PNGS at position 0
        nt_ref = "AATGCTTCT"
        markers = find_pngs_markers_nt(nt_ref)
        assert len(markers) == 1
        assert markers[0].col == 0  # first NT of the N codon
        assert markers[0].label == "N1"  # 0-based AA pos + 1

    def test_find_pngs_markers_nt_with_gaps(self):
        """Gaps in the NT alignment should not affect PNGS detection."""
        from tpixel.pngs import find_pngs_markers_nt

        # Insert gaps before the NAS codons
        nt_ref = "---AATGCTTCT"
        markers = find_pngs_markers_nt(nt_ref)
        assert len(markers) == 1
        assert markers[0].col == 3  # column 3 (after 3 gap columns)

    def test_find_pngs_markers_nt_with_hxb2_labels(self):
        """PNGS markers get HxB2 AA position labels when map is provided."""
        from tpixel.hxb2 import build_hxb2_map
        from tpixel.pngs import find_pngs_markers_nt

        # HxB2 and ref are the same NT sequence: N A S
        nt_seq = "AATGCTTCT"
        seqs = [("HxB2", nt_seq)]
        hxb2_map = build_hxb2_map(seqs, seq_type="NT")
        markers = find_pngs_markers_nt(nt_seq, hxb2_map)
        assert len(markers) == 1
        # Column 0 maps to HxB2 AA pos 1
        assert markers[0].label == "N1"

    def test_no_pngs_when_x_is_proline(self):
        """NPS motif should NOT be detected as PNGS."""
        from tpixel.pngs import find_pngs_markers_nt

        # AAT=N, CCT=P, TCT=S → NPS, not a valid PNGS
        nt_ref = "AATCCTTCT"
        markers = find_pngs_markers_nt(nt_ref)
        assert len(markers) == 0


class TestNTRendering:
    """Test that NT HIV panels build and render successfully."""

    def test_nt_hiv_panel_builds(self, write_fasta):
        """NT HIV panel builds without errors."""
        from tpixel.hiv import hiv_panel

        # Build minimal NT HIV alignment (30 codons = 90 NT per sequence)
        hxb2_nt = "ACGTACGTAC" * 9  # 90 NT
        ref_nt = "ACGTACGTAC" * 9
        s1_nt = list(ref_nt)
        s1_nt[3] = "A"  # substitution
        s1_nt = "".join(s1_nt)

        seqs = [
            ("HxB2", hxb2_nt),
            ("animal1_ref", ref_nt),
            ("animal1_s1", s1_nt),
        ]
        path = write_fasta(seqs, "nt_hiv.fasta")
        panel = hiv_panel(str(path))
        assert panel.total_cols == 90

    def test_nt_hiv_panel_renders_png(self, write_fasta, output_dir):
        """NT HIV panel renders to PNG without errors."""
        from tpixel.hiv import hiv_panel
        from tpixel.renderer import render_panels

        hxb2_nt = "ACGTACGTAC" * 9
        ref_nt = "ACGTACGTAC" * 9
        s1_nt = list(ref_nt)
        s1_nt[3] = "A"
        s1_nt = "".join(s1_nt)
        s2_nt = list(ref_nt)
        s2_nt[6] = "A"
        s2_nt = "".join(s2_nt)

        seqs = [
            ("HxB2", hxb2_nt),
            ("animal1_ref", ref_nt),
            ("animal1_s1", s1_nt),
            ("animal2_s1", s2_nt),
        ]
        path = write_fasta(seqs, "nt_hiv_render.fasta")
        panel = hiv_panel(str(path))
        out = output_dir / "nt_hiv_panel.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_autodetect_nt_fasta(self, write_fasta):
        """Auto-detect correctly identifies NT FASTA as NT."""
        from tpixel.hiv import hiv_panel

        hxb2_nt = "ACGTACGTAC" * 9
        ref_nt = "ACGTACGTAC" * 9
        s1_nt = "ACCTACGTAC" * 9

        seqs = [
            ("HxB2", hxb2_nt),
            ("animal1_ref", ref_nt),
            ("animal1_s1", s1_nt),
        ]
        path = write_fasta(seqs, "autodetect_nt.fasta")
        panel = hiv_panel(str(path))
        assert panel.seq_type == "NT"

    def test_autodetect_aa_fasta(self, write_fasta):
        """Auto-detect correctly identifies AA FASTA as AA."""
        from tpixel.hiv import hiv_panel

        # Use amino acid characters that aren't ACGTU
        hxb2_aa = "MWLKFHRD" * 5
        ref_aa = "MWLKFHRD" * 5
        s1_aa = "MWLKFHRE" * 5

        seqs = [
            ("HxB2", hxb2_aa),
            ("animal1_ref", ref_aa),
            ("animal1_s1", s1_aa),
        ]
        path = write_fasta(seqs, "autodetect_aa.fasta")
        panel = hiv_panel(str(path))
        assert panel.seq_type == "AA"

    def test_forced_nt_mode(self, write_fasta):
        """Explicit seq_type='NT' overrides auto-detection."""
        from tpixel.hiv import hiv_panel

        hxb2_nt = "ACGTACGTAC" * 9
        ref_nt = "ACGTACGTAC" * 9
        s1_nt = "ACCTACGTAC" * 9

        seqs = [
            ("HxB2", hxb2_nt),
            ("animal1_ref", ref_nt),
            ("animal1_s1", s1_nt),
        ]
        path = write_fasta(seqs, "forced_nt.fasta")
        panel = hiv_panel(str(path), seq_type="NT")
        assert panel.seq_type == "NT"


class TestHeterologousColoring:
    """Three-way comparison: match, pure mismatch, heterologous (matches secondary ref)."""

    def _panel(self, ref: str, secondary: str | None, rows: list[tuple[str, str]]) -> Panel:
        return Panel(
            label="primary",
            ref_row=list(ref),
            seq_rows=[(name, list(bases)) for name, bases in rows],
            total_cols=len(ref),
            col_labels=[],
            secondary_ref_row=list(secondary) if secondary is not None else None,
        )

    def test_no_secondary_ref_falls_back_to_mismatch(self, output_dir):
        panel = self._panel("ACGTACGT", None, [("s1", "ATCTACGT")])
        out = output_dir / "no_secondary.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()
        assert panel.secondary_ref_row is None

    def test_heterologous_color_default(self):
        panel = self._panel("AAAA", "TTTT", [("s1", "TTTT")])
        assert panel.heterologous_color == "#FF6F00"
        assert panel.heterologous_color != "#D32F2F"

    def test_three_way_classification(self, output_dir):
        panel = self._panel(
            ref="AAAA",
            secondary="TGCT",
            rows=[
                ("heterologous_at_123", "AGCT"),
                ("pure_mismatch_at_1", "ACAA"),
            ],
        )
        out = output_dir / "three_way_coloring.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()

    def test_secondary_ref_shorter_than_primary(self, output_dir):
        # If secondary_ref_row is shorter than ref_row, the renderer must
        # not index out of bounds. Falls back to MISMATCH for cols beyond.
        panel = self._panel("AAAAAAAA", "GGGG", [("s1", "GGGGCCCC")])
        out = output_dir / "secondary_short.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()

    def test_legend_includes_heterologous(self, output_dir):
        panel = self._panel("AAAA", "GGGG", [("s1", "GGAA")])
        out = output_dir / "legend_with_heterologous.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()

    def test_hiv_panel_with_secondary_ref(self, write_fasta, output_dir):
        # End-to-end: hiv_panel loads a secondary ref FASTA aligned to the
        # panel's coordinate space and renders correctly.
        from tpixel.hiv import hiv_panel

        hxb2_aa = "MWLKFHRD" * 5
        ref_aa = "MWLKFHRD" * 5
        sec_aa = "MWLKFHRE" * 5  # differs from primary at every 8th pos
        s1_aa = "MWLKFHRE" * 5  # matches secondary at the diverging positions

        main_path = write_fasta(
            [("HxB2", hxb2_aa), ("animal1_ref", ref_aa), ("animal1_s1", s1_aa)],
            "main.fasta",
        )
        sec_path = write_fasta([("animal2_ref", sec_aa)], "secondary.fasta")

        panel = hiv_panel(str(main_path), secondary_ref_path=str(sec_path))
        assert panel.secondary_ref_row is not None
        assert len(panel.secondary_ref_row) == panel.total_cols

        out = output_dir / "hiv_with_secondary.png"
        render_panels([panel], str(out), dpi=150)
        assert out.exists()

    def test_hiv_panel_secondary_ref_length_mismatch_raises(self, write_fasta):
        # The secondary ref must already be aligned to the panel's columns;
        # if its length differs, hiv_panel must reject it with a clear error.
        from tpixel.hiv import hiv_panel

        main = write_fasta(
            [("HxB2", "ACGTACGT"), ("animal1_ref", "ACGTACGT"), ("animal1_s1", "ACGTACGT")],
            "main.fasta",
        )
        sec = write_fasta([("other_ref", "ACGT")], "wrong_length.fasta")  # too short

        import pytest
        with pytest.raises(ValueError, match="Secondary ref length"):
            hiv_panel(str(main), secondary_ref_path=str(sec))
