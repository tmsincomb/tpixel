"""HIV-aware panel builder for PIXEL plots.

Handles HxB2 coordinate mapping, Env region annotations, PNGS markers,
and animal-based sequence grouping from SHIV/HIV aligned FASTA files.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from tpixel.fasta import read_fasta
from tpixel.hxb2 import _is_nucleotide, build_hxb2_map, hxb2_col_labels, hxb2_regions
from tpixel.models import Marker, Panel, SeqGroup
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


def hiv_panel(
    path: str | Path,
    hxb2_id: str = "HxB2",
    ref_id: str | None = None,
    tick_step: int = 50,
    ref_positions: list[int] | None = None,
    seq_type: str | None = None,
    secondary_ref_path: str | Path | None = None,
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
    )
