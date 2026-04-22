"""HIV-aware panel builder for PIXEL plots.

Handles HxB2 coordinate mapping, Env region annotations, PNGS markers,
and animal-based sequence grouping from SHIV/HIV aligned FASTA files.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from tpixel.fasta import read_fasta
from tpixel.hxb2 import _is_nucleotide, build_hxb2_map, hxb2_col_labels, hxb2_regions
from tpixel.models import Marker, Panel, Region, SeqGroup
from tpixel.pngs import find_pngs_markers, find_pngs_markers_nt


def _find_ref_id(names: list[str]) -> str | None:
    """Find the parental reference (name ending with ``'_ref'``).

    Args:
        names: Sequence IDs from the alignment.

    Returns:
        First name ending with ``'_ref'``, or ``None``.

    Examples:
        >>> _find_ref_id(["HxB2", "animal1_ref", "animal1_s1"])
        'animal1_ref'
        >>> _find_ref_id(["HxB2", "s1", "s2"]) is None
        True
    """
    for name in names:
        if name.endswith("_ref"):
            return name
    return None


def _extract_animal(seq_id: str) -> str:
    """Extract animal name from sequence ID (prefix before first ``'_'``).

    Args:
        seq_id: Full sequence identifier string.

    Returns:
        The portion of *seq_id* before the first underscore.

    Examples:
        >>> _extract_animal("animal1_s1")
        'animal1'
        >>> _extract_animal("RM5695_env_s3")
        'RM5695'
        >>> _extract_animal("nounderscore")
        'nounderscore'
    """
    parts = seq_id.split("_")
    return parts[0]


def _sort_animal_groups(animal_names: list[str], lineage: str) -> list[str]:
    """Sort: lineage self first, recombinants, then alphabetical.

    Args:
        animal_names: Unique animal/group names to sort.
        lineage: The lineage name to place first.

    Returns:
        Sorted list: lineage first, then recombinants, then others alphabetically.

    Examples:
        >>> _sort_animal_groups(["B", "rec1", "A", "lin1"], "lin1")
        ['lin1', 'rec1', 'A', 'B']
        >>> _sort_animal_groups(["X", "Y"], "Z")
        ['X', 'Y']
    """
    self_group = []
    rec_group = []
    other_group = []
    for name in animal_names:
        if name == lineage:
            self_group.append(name)
        elif name.lower().startswith("rec"):
            rec_group.append(name)
        else:
            other_group.append(name)
    return self_group + sorted(rec_group) + sorted(other_group)


def _apply_region_palette(
    regions: list[Region], palette: dict[str, str]
) -> list[Region]:
    """Override region colors from ``palette`` and merge same-color neighbours.

    When two adjacent regions end up with the same color (e.g. V1 and V2
    both mapped to pink), they are fused into one ``Region`` whose name
    joins the originals with ``'/'`` (e.g. ``"V1/V2"``).

    Args:
        regions: Original list of Region annotations (in column order).
        palette: Mapping of region name → hex color.  Unknown names are
            ignored.  Regions whose name is absent keep their prior color.

    Returns:
        New list of Region objects with overrides + merges applied.
    """
    recolored: list[Region] = []
    for r in regions:
        new_color = palette.get(r.name, r.color)
        recolored.append(Region(name=r.name, start=r.start, end=r.end, color=new_color))

    # Merge adjacent regions that share a color AND were both touched by
    # the palette.  Non-palette regions keep their identity so we do not
    # accidentally fuse, e.g., neighbouring "EEEEEE" constant bands.
    merged: list[Region] = []
    for r in recolored:
        if (
            merged
            and r.name in palette
            and merged[-1].name.split("/")[0] in palette
            and merged[-1].color.upper() == r.color.upper()
            and merged[-1].end == r.start
        ):
            prev = merged[-1]
            prev_names = prev.name.split("/")
            # Avoid duplicate component names if called twice.
            if r.name not in prev_names:
                new_name = prev.name + "/" + r.name
            else:
                new_name = prev.name
            merged[-1] = Region(
                name=new_name,
                start=prev.start,
                end=r.end,
                color=prev.color,
            )
        else:
            merged.append(r)
    return merged


def _hxb2_nt_col_labels(
    hxb2_map: list,
    step: int = 250,
    seq_type: str | None = None,
) -> list[tuple[int, str]]:
    """Build x-axis tick labels at regular HxB2 nucleotide intervals.

    For an AA alignment, NT coordinates are derived from the
    ``hxb2_aa_pos`` of each mapped column as ``(aa_pos - 1) * 3 + 1``.
    For an NT alignment the column's implicit NT counter is used.

    Args:
        hxb2_map: List of HxB2Position entries, one per alignment column.
        step: Nucleotide tick interval (e.g. 250 NT).
        seq_type: ``"NT"`` or ``"AA"``.  When ``None``, the first non-gap
            position is inspected — but callers that know the type should
            pass it explicitly.

    Returns:
        List of ``(column_index, nt_label)`` tuples.
    """
    # Infer NT position for every mapped column.
    # For NT alignments: the NT coordinate equals the 1-based NT counter,
    # which we can recover by iterating non-gap columns in order.
    # For AA alignments: NT = (aa - 1) * 3 + 1.
    is_nt = seq_type == "NT"
    nt_positions: list[tuple[int, int]] = []  # (column, nt_coord)
    nt_counter = 0
    for p in hxb2_map:
        if p.hxb2_aa_pos is None:
            continue
        if is_nt:
            nt_counter += 1
            nt_coord = nt_counter
        else:
            nt_coord = (p.hxb2_aa_pos - 1) * 3 + 1
        nt_positions.append((p.alignment_col, nt_coord))

    if not nt_positions:
        return []

    max_nt = max(nt for _col, nt in nt_positions)
    labels: list[tuple[int, str]] = []
    for target in range(step, max_nt + 1, step):
        # Find the first column whose NT coord is >= target.
        for col, nt in nt_positions:
            if nt >= target:
                labels.append((col, str(target)))
                break
    return labels


def hiv_panel(
    path: str | Path,
    hxb2_id: str = "HxB2",
    ref_id: str | None = None,
    tick_step: int = 50,
    ref_positions: list[int] | None = None,
    seq_type: str | None = None,
    secondary_ref_path: str | Path | None = None,
    region_palette: dict[str, str] | None = None,
    show_nt_ruler: bool = False,
    nt_ruler_step: int = 250,
) -> Panel:
    """Build a full Roark-style Panel from an HIV Env alignment.

    Args:
        path: Path to aligned FASTA containing HxB2 and a *_ref sequence.
            Accepts both amino-acid and nucleotide alignments.
        hxb2_id: ID of the HxB2 coordinate reference in the alignment.
        ref_id: Parental reference ID. Auto-detected (*_ref) if None.
            Ignored when ref_positions is provided.
        tick_step: HxB2 AA position interval for x-axis ticks.
        ref_positions: 1-based positions of reference sequences. Last is
            the primary reference; earlier ones become extra reference rows.
            Defaults to [1, 2].
        seq_type: ``"NT"`` or ``"AA"``.  Auto-detected from the reference
            sequence when *None*.
        secondary_ref_path: Optional FASTA with a secondary reference
            already aligned to this panel's column space.
        region_palette: Optional mapping of region name (e.g. ``"V3"``,
            ``"gp41"``) to hex color.  When provided, region colors in the
            returned Panel are overridden.  If adjacent regions share the
            same override color (for example both ``"V1"`` and ``"V2"``
            mapped to pink), they are merged into a single contiguous
            region (e.g. ``"V1/V2"``) so the renderer draws one band.
        show_nt_ruler: When ``True``, the returned Panel's ``col_labels``
            are tick positions in nucleotide coordinates (HxB2 NT numbering)
            instead of amino-acid positions.  For AA alignments the NT
            coordinate is computed as ``(aa_pos - 1) * 3 + 1``.
        nt_ruler_step: Nucleotide interval between ticks when
            ``show_nt_ruler`` is ``True``.  Default ``250`` NT.

    Returns:
        Panel with regions, PNGS markers, grouped sequences, and HxB2 ticks.
    """
    seqs = read_fasta(path)
    if not seqs:
        raise ValueError(f"No sequences in {path}")

    names = [n for n, _ in seqs]
    seq_dict = {n: s for n, s in seqs}

    if ref_positions is not None:
        # Position-based: last position is primary reference
        primary_idx = ref_positions[-1] - 1
        ref_id = names[primary_idx]
    else:
        # Name-based auto-detection (original behavior)
        ref_positions = [1, 2]
        if ref_id is None:
            ref_id = _find_ref_id(names)
        if ref_id is None:
            raise ValueError("No *_ref sequence found. Specify ref_id explicitly.")
        if ref_id not in seq_dict:
            raise ValueError(f"Reference '{ref_id}' not in alignment")

    ref_seq = seq_dict[ref_id]
    aln_len = len(ref_seq)
    ref_row = list(ref_seq.upper())

    # Auto-detect sequence type from reference when not specified
    if seq_type is None:
        seq_type = "NT" if _is_nucleotide(ref_seq) else "AA"

    hxb2_map = build_hxb2_map(seqs, hxb2_id, seq_type=seq_type)
    regions = hxb2_regions(hxb2_map)

    # Apply region palette override (and merge same-color adjacent regions).
    if region_palette:
        regions = _apply_region_palette(regions, region_palette)

    if show_nt_ruler:
        nt_ruler_labels = _hxb2_nt_col_labels(
            hxb2_map, step=nt_ruler_step, seq_type=seq_type
        )
        # Keep ``col_labels`` populated with the same NT-scale values so
        # callers that inspect ``panel.col_labels`` for tick-value
        # computation (existing tests) continue to observe NT labels,
        # while the renderer uses ``nt_ruler_labels`` to draw the dedicated
        # top header track ABOVE the region color bar (not below).
        col_labels = nt_ruler_labels
    else:
        nt_ruler_labels = None
        col_labels = hxb2_col_labels(hxb2_map, step=tick_step)

    if seq_type == "NT":
        markers = find_pngs_markers_nt(ref_seq, hxb2_map)
    else:
        markers = find_pngs_markers(ref_seq, hxb2_map)

    lineage = ref_id.replace("_ref", "") if ref_id.endswith("_ref") else ref_id

    # Extra reference rows: all ref positions except the last
    extra_ref_rows: list[tuple[str, list[str]]] = []
    for pos in ref_positions[:-1]:
        idx = pos - 1
        name = names[idx]
        seq = seq_dict[name]
        row = list(seq.upper()[:aln_len])
        row += ["-"] * (aln_len - len(row))
        extra_ref_rows.append((name, row))

    # Group sample sequences by animal
    skip = {names[pos - 1] for pos in ref_positions}
    animal_seqs: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for name, seq in seqs:
        if name in skip:
            continue
        animal = _extract_animal(name)
        row = list(seq.upper()[:aln_len])
        row += ["-"] * (aln_len - len(row))
        animal_seqs[animal].append((name, row))

    sorted_animals = _sort_animal_groups(list(animal_seqs.keys()), lineage)
    groups = [SeqGroup(name=a, seqs=animal_seqs[a]) for a in sorted_animals]

    # Optional secondary reference for heterologous-recombination coloring.
    # The provided FASTA must contain a single sequence already aligned to
    # this panel's column space (e.g. via `mafft --add --keeplength`).
    secondary_ref_row: list[str] | None = None
    if secondary_ref_path is not None:
        sec_seqs = read_fasta(secondary_ref_path)
        if not sec_seqs:
            raise ValueError(f"No sequences in secondary ref {secondary_ref_path}")
        _, sec_seq = sec_seqs[0]
        if len(sec_seq) != aln_len:
            raise ValueError(
                f"Secondary ref length {len(sec_seq)} != panel length {aln_len}; "
                "use `mafft --add --keeplength` to align it to the panel coordinates."
            )
        secondary_ref_row = list(sec_seq.upper())

    return Panel(
        label=ref_id,
        ref_row=ref_row,
        seq_rows=[],
        total_cols=aln_len,
        col_labels=col_labels,
        regions=regions,
        markers=markers,
        marker_color="#4CAF50",
        groups=groups,
        extra_ref_rows=extra_ref_rows,
        secondary_ref_row=secondary_ref_row,
        nt_ruler_labels=nt_ruler_labels,
    )
