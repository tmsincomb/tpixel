"""Non-HxB2 anchor coordinate mapping for HIV Env alignments.

When the user's alignment lacks the literal ``HxB2`` row (e.g. they want to
plot a single-reference panel against ``SF162p3_ref`` only), region bands and
the NT ruler still need HxB2 amino-acid coordinates. This module ships
pre-aligned ``(HxB2, lineage)`` AA pairs for known lineages and produces a
:class:`tpixel.hxb2.HxB2Position` list that the renderer consumes identically
to one built from a real HxB2 row.

Bundled anchor lineages (``data/anchor_refs/{lineage}.fasta``):

* ``CH505`` — CH505 T/F clone (from ``top_env_protein_aligned.fasta``).
* ``SF162p3`` — SF162p3 reference (from ``mid_env_protein_aligned.fasta``).
* ``T250-4`` — T250-4 reference (from ``bot_env_protein_aligned.fasta``).

Each bundled FASTA has two records of equal column count: HxB2 first, lineage
second, both gapped against each other (``-``). Provenance is the project's
own MAFFT-derived panel alignments.
"""

from __future__ import annotations

import functools
from importlib import resources

from tpixel.hxb2 import HxB2Position, _is_nucleotide, get_env_region

KNOWN_ANCHOR_LINEAGES: tuple[str, ...] = ("CH505", "SF162p3", "T250-4")


def detect_anchor_lineage(seq_id: str) -> str | None:
    """Return canonical lineage name if ``seq_id`` matches a known anchor.

    Args:
        seq_id: Sequence ID, typically ``"<lineage>_ref"``.

    Returns:
        One of :data:`KNOWN_ANCHOR_LINEAGES` or ``None``.

    Examples:
        >>> detect_anchor_lineage("SF162p3_ref")
        'SF162p3'
        >>> detect_anchor_lineage("CH505_ref")
        'CH505'
        >>> detect_anchor_lineage("T250-4_ref")
        'T250-4'
        >>> detect_anchor_lineage("foo_ref") is None
        True
        >>> detect_anchor_lineage("HxB2") is None
        True
    """
    base = seq_id.removesuffix("_ref")
    return base if base in KNOWN_ANCHOR_LINEAGES else None


@functools.cache
def _load_anchor_pair(lineage: str) -> tuple[str, str]:
    """Read the bundled ``(HxB2_aligned, lineage_aligned)`` AA pair.

    Args:
        lineage: One of :data:`KNOWN_ANCHOR_LINEAGES`.

    Returns:
        Tuple of two equal-length AA strings (HxB2, lineage), gapped with ``-``.

    Raises:
        ValueError: If lineage unknown or bundle malformed.
    """
    if lineage not in KNOWN_ANCHOR_LINEAGES:
        raise ValueError(
            f"Unknown anchor lineage '{lineage}'. Known: {KNOWN_ANCHOR_LINEAGES}"
        )
    fasta_path = resources.files("tpixel") / "data" / "anchor_refs" / f"{lineage}.fasta"
    text = fasta_path.read_text()

    records: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = []
            records.append((line[1:].split()[0], current))
        else:
            assert current is not None
            current.append(line.upper())

    if len(records) != 2:
        raise ValueError(
            f"Anchor bundle {lineage} must have exactly 2 records, got {len(records)}"
        )
    hxb2_id, hxb2_chunks = records[0]
    lin_id, lin_chunks = records[1]
    if hxb2_id != "HxB2":
        raise ValueError(f"Anchor bundle {lineage} first record must be HxB2, got {hxb2_id}")
    if lin_id != lineage:
        raise ValueError(
            f"Anchor bundle {lineage} second record must be {lineage}, got {lin_id}"
        )
    hxb2_aligned = "".join(hxb2_chunks)
    lin_aligned = "".join(lin_chunks)
    if len(hxb2_aligned) != len(lin_aligned):
        raise ValueError(
            f"Anchor bundle {lineage} record lengths differ: "
            f"HxB2={len(hxb2_aligned)} {lineage}={len(lin_aligned)}"
        )
    return hxb2_aligned, lin_aligned


@functools.cache
def _build_lineage_to_hxb2_lookup(
    lineage: str,
) -> tuple[list[int | None], list[str], str]:
    """Build per-lineage-residue lookup of HxB2 AA position and residue.

    Args:
        lineage: One of :data:`KNOWN_ANCHOR_LINEAGES`.

    Returns:
        Tuple of:
            * ``lineage_to_hxb2_aa`` — list indexed by 0-based lineage AA
              position. Entry is the 1-based HxB2 AA position, or ``None``
              when the lineage residue is an insertion vs HxB2.
            * ``lineage_to_hxb2_residue`` — list of HxB2 residue characters
              (or ``"-"`` for insertions vs HxB2) at each lineage position.
            * ``lineage_canonical_aa`` — ungapped lineage AA sequence,
              for sanity-checking against the user's anchor row.
    """
    hxb2_aligned, lin_aligned = _load_anchor_pair(lineage)
    lineage_to_hxb2_aa: list[int | None] = []
    lineage_to_hxb2_residue: list[str] = []
    canonical_chars: list[str] = []
    hxb2_pos = 0
    for h_res, l_res in zip(hxb2_aligned, lin_aligned):
        h_gap = h_res in ("-", ".")
        l_gap = l_res in ("-", ".")
        if not h_gap:
            hxb2_pos += 1
        if not l_gap:
            canonical_chars.append(l_res)
            if h_gap:
                lineage_to_hxb2_aa.append(None)
                lineage_to_hxb2_residue.append("-")
            else:
                lineage_to_hxb2_aa.append(hxb2_pos)
                lineage_to_hxb2_residue.append(h_res)
    return lineage_to_hxb2_aa, lineage_to_hxb2_residue, "".join(canonical_chars)


def build_anchor_hxb2_map(
    aligned_seqs: list[tuple[str, str]],
    anchor_id: str,
    lineage: str,
    seq_type: str | None = None,
) -> list[HxB2Position]:
    """Walk an anchor row to build an HxB2-coordinate map without an HxB2 row.

    The bundled lineage->HxB2 lookup supplies the HxB2 AA position for every
    non-gap residue of the canonical lineage AA sequence. This function pairs
    that lookup with the user's anchor row in their alignment, producing one
    :class:`HxB2Position` per alignment column.

    Args:
        aligned_seqs: List of ``(name, sequence)`` from
            :func:`tpixel.fasta.read_fasta`.
        anchor_id: Sequence ID of the anchor row in ``aligned_seqs``
            (typically ``"<lineage>_ref"``).
        lineage: Anchor lineage. Must be in :data:`KNOWN_ANCHOR_LINEAGES`.
        seq_type: ``"NT"`` or ``"AA"``. Auto-detected from the anchor row
            when ``None``.

    Returns:
        One :class:`HxB2Position` per alignment column.

    Raises:
        ValueError: If ``anchor_id`` not found, lineage unknown, NT mode
            requested (not yet supported), or the user's anchor row's
            ungapped sequence does not match the bundled canonical.
    """
    if lineage not in KNOWN_ANCHOR_LINEAGES:
        raise ValueError(
            f"Unknown anchor lineage '{lineage}'. Known: {KNOWN_ANCHOR_LINEAGES}"
        )

    anchor_seq: str | None = None
    for name, seq in aligned_seqs:
        if name == anchor_id or name.split()[0] == anchor_id:
            anchor_seq = seq
            break
    if anchor_seq is None:
        raise ValueError(f"Anchor sequence '{anchor_id}' not found in alignment")

    if seq_type is None:
        seq_type = "NT" if _is_nucleotide(anchor_seq) else "AA"
    if seq_type == "NT":
        raise ValueError(
            "Anchor mode currently supports AA alignments only; "
            "include the HxB2 row for NT alignments, or run with --aa."
        )

    lin_to_hxb2_aa, lin_to_hxb2_res, canonical = _build_lineage_to_hxb2_lookup(lineage)
    ungapped = "".join(c for c in anchor_seq.upper() if c not in ("-", "."))
    if ungapped != canonical:
        raise ValueError(
            f"Anchor row '{anchor_id}' (ungapped len={len(ungapped)}) does not match "
            f"bundled canonical {lineage} (len={len(canonical)}). The bundled mapping "
            f"is built from the project's panel alignments; sequence variants are not "
            f"supported yet."
        )

    positions: list[HxB2Position] = []
    lineage_pos = 0
    for col_idx, residue in enumerate(anchor_seq.upper()):
        if residue in ("-", "."):
            positions.append(HxB2Position(col_idx, None, None, residue))
        else:
            hxb2_aa = lin_to_hxb2_aa[lineage_pos]
            hxb2_res = lin_to_hxb2_res[lineage_pos]
            lineage_pos += 1
            region = get_env_region(hxb2_aa) if hxb2_aa is not None else None
            positions.append(HxB2Position(col_idx, hxb2_aa, region, hxb2_res))
    return positions
