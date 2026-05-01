"""HxB2 coordinate mapping for HIV Env gp160 protein alignments.

Maps alignment columns to HxB2 amino acid positions and Env structural
regions using the LANL convention boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from tpixel.models import Region

ENV_REGIONS: list[tuple[str, int, int]] = [
    ("SP",    1,  30),
    ("C1",   31, 130),
    ("V1",  131, 157),
    ("V2",  158, 196),
    ("C2",  197, 295),
    ("V3",  296, 331),
    ("C3",  332, 384),
    ("V4",  385, 418),
    ("C4",  419, 459),
    ("V5",  460, 469),
    ("C5",  470, 511),
    ("gp41", 512, 856),
]

REGION_COLORS: dict[str, str] = {
    "SP":   "#FFF9C4",
    "C1":   "#EEEEEE",
    "V1":   "#BBDEFB",
    "V2":   "#BBDEFB",
    "C2":   "#EEEEEE",
    "V3":   "#BBDEFB",
    "C3":   "#EEEEEE",
    "V4":   "#BBDEFB",
    "C4":   "#EEEEEE",
    "V5":   "#BBDEFB",
    "C5":   "#EEEEEE",
    "gp41": "#F8BBD0",
}

_REGION_LOOKUP: dict[int, str] = {}
for _name, _start, _end in ENV_REGIONS:
    for _pos in range(_start, _end + 1):
        _REGION_LOOKUP[_pos] = _name


def get_env_region(hxb2_aa_pos: int) -> str | None:
    """Return the Env region name for an HxB2 amino acid position.

    Args:
        hxb2_aa_pos: 1-based HxB2 amino acid position.

    Returns:
        Region name (e.g. ``'V3'``) or ``None`` if outside known boundaries.

    Examples:
        >>> get_env_region(1)
        'SP'
        >>> get_env_region(131)
        'V1'
        >>> get_env_region(296)
        'V3'
        >>> get_env_region(900) is None
        True
    """
    return _REGION_LOOKUP.get(hxb2_aa_pos)


@dataclass
class HxB2Position:
    """A single alignment column mapped to HxB2 coordinates.

    Attributes:
        alignment_col: 0-based alignment column index.
        hxb2_aa_pos: 1-based HxB2 amino acid position, or ``None`` for gaps.
        region: Env region name (e.g. ``'V3'``), or ``None``.
        hxb2_residue: The residue character at this column in the HxB2 sequence.
    """

    alignment_col: int
    hxb2_aa_pos: int | None
    region: str | None
    hxb2_residue: str


def _is_nucleotide(seq: str) -> bool:
    """Return True if *seq* looks like a nucleotide sequence.

    Examples:
        >>> _is_nucleotide("ACGTACGT")
        True
        >>> _is_nucleotide("MWLK")
        False
        >>> _is_nucleotide("ACG-T.NU")
        True
        >>> _is_nucleotide("")
        True
    """
    nt_chars = set("ACGTUNacgtun-.")
    return all(c in nt_chars for c in seq)


def build_hxb2_map(
    aligned_seqs: list[tuple[str, str]],
    hxb2_id: str = "HxB2",
    seq_type: str | None = None,
) -> list[HxB2Position]:
    """Walk the HxB2 row and map every alignment column to HxB2 coordinates.

    Args:
        aligned_seqs: List of (name, sequence) from read_fasta.
        hxb2_id: Sequence ID of HxB2 in the alignment.
        seq_type: ``"NT"`` or ``"AA"``.  Auto-detected from the HxB2
            sequence when *None*.

    Returns:
        One HxB2Position per alignment column.
    """
    hxb2_seq = None
    for name, seq in aligned_seqs:
        if name == hxb2_id or name.split()[0] == hxb2_id:
            hxb2_seq = seq
            break

    if hxb2_seq is None:
        raise ValueError(f"HxB2 sequence '{hxb2_id}' not found in alignment")

    if seq_type is None:
        seq_type = "NT" if _is_nucleotide(hxb2_seq) else "AA"

    is_nt = seq_type == "NT"

    positions: list[HxB2Position] = []
    nt_counter = 0
    aa_counter = 0

    for col_idx, residue in enumerate(hxb2_seq):
        if residue in ("-", "."):
            positions.append(HxB2Position(col_idx, None, None, residue))
        else:
            if is_nt:
                nt_counter += 1
                aa_pos = (nt_counter - 1) // 3 + 1
            else:
                aa_counter += 1
                aa_pos = aa_counter
            positions.append(HxB2Position(col_idx, aa_pos, get_env_region(aa_pos), residue))

    return positions


def hxb2_col_labels(hxb2_map: list[HxB2Position], step: int = 50) -> list[tuple[int, str]]:
    """Build x-axis tick labels at regular HxB2 AA intervals.

    Always includes the first HxB2-mapped column as a tick so that
    panels starting at a non-standard AA position (e.g. a windowed
    alignment that begins mid-sequence) expose their starting
    coordinate. Duplicates with the step-interval ticks are filtered
    so a panel whose first AA lands exactly on a multiple of ``step``
    doesn't render two overlapping tick labels.
    """
    mapped = [p for p in hxb2_map if p.hxb2_aa_pos is not None]
    if not mapped:
        return []
    first = mapped[0]
    max_pos = mapped[-1].hxb2_aa_pos
    labels: list[tuple[int, str]] = [(first.alignment_col, str(first.hxb2_aa_pos))]
    for target in range(step, max_pos + 1, step):
        for p in mapped:
            if p.hxb2_aa_pos == target:
                if p.alignment_col != labels[0][0]:
                    labels.append((p.alignment_col, str(target)))
                break
    return labels


def hxb2_variant_labels(
    hxb2_row: list[str],
    query_row: list[str],
    hxb2_map: list[HxB2Position],
    *,
    seq_type: str = "AA",
) -> list[tuple[int, str]]:
    """Return (column_index, ``"wildtype+pos+mutation"``) labels for every
    column where *query_row* differs from *hxb2_row*.

    Label format follows the standard mutation shorthand:

    * Substitution → ``"K169E"`` (HxB2 residue + HxB2 position + query residue).
    * Deletion (query gap) → ``"K169-"``.
    * Insertion relative to HxB2 (HxB2 gap) → skipped (those columns have
      no stable HxB2 position to print).
    * Matching positions → skipped.

    For AA alignments the position is the 1-based HxB2 amino-acid position
    (e.g. ``169``). For NT alignments it is the 1-based HxB2 nucleotide
    position (e.g. ``507``).

    Args:
        hxb2_row: Aligned HxB2 row (single-character strings).
        query_row: Aligned query row to compare against HxB2 (same length).
        hxb2_map: Output of :func:`build_hxb2_map` (one entry per column).
        seq_type: ``"AA"`` or ``"NT"``. Controls whether positions are
            reported in amino-acid or nucleotide coordinates.

    Returns:
        Ordered list of ``(col_idx, label)`` pairs, one per variant column.

    Examples:
        >>> from tpixel.hxb2 import build_hxb2_map, hxb2_variant_labels
        >>> seqs = [("HxB2", "MKRVK"), ("lin_ref", "MKEVK")]
        >>> m = build_hxb2_map(seqs, hxb2_id="HxB2", seq_type="AA")
        >>> hxb2_variant_labels(list("MKRVK"), list("MKEVK"), m, seq_type="AA")
        [(2, 'R3E')]
    """
    GAPS = {"-", ".", " "}
    is_nt = seq_type == "NT"

    out: list[tuple[int, str]] = []
    nt_counter = 0
    for i, p in enumerate(hxb2_map):
        hxb2_res = hxb2_row[i] if i < len(hxb2_row) else "-"
        query_res = query_row[i] if i < len(query_row) else "-"

        if hxb2_res in GAPS:
            continue

        if is_nt:
            nt_counter += 1
            pos_label = str(nt_counter)
        else:
            if p.hxb2_aa_pos is None:
                continue
            pos_label = str(p.hxb2_aa_pos)

        if query_res == hxb2_res:
            continue

        mut = "-" if query_res in GAPS else query_res.upper()
        out.append((i, f"{hxb2_res.upper()}{pos_label}{mut}"))

    return out


def hxb2_regions(hxb2_map: list[HxB2Position]) -> list[Region]:
    """Build Region annotations from HxB2 position map.

    Columns whose ``region`` is ``None`` (HxB2 row gap, or lineage
    insertion vs HxB2 in anchor mode) extend whichever region they fall
    within rather than fragmenting the bar — small indels do not break
    visual continuity. A region ends only when the position map either
    transitions to a different named region or runs out of columns.
    """
    region_spans: list[tuple[str, int, int]] = []
    current: str | None = None
    span_start = 0

    for p in hxb2_map:
        r = p.region
        if r is None:
            # Stay inside the current span — gap column gets absorbed.
            continue
        if r == current:
            continue
        if current is not None:
            region_spans.append((current, span_start, p.alignment_col))
        current = r
        span_start = p.alignment_col

    if current is not None:
        region_spans.append((current, span_start, len(hxb2_map)))

    return [
        Region(name, start, end, REGION_COLORS.get(name, "#EEEEEE"))
        for name, start, end in region_spans
        if name is not None
    ]
