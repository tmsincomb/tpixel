"""Data structures for tpixel alignment panels."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Region:
    """A named region annotation spanning a range of columns.

    Attributes:
        name: Display label for the region.
        start: 0-based start column index (inclusive).
        end: 0-based end column index (exclusive).
        color: Fill color (hex string).
    """

    name: str
    start: int
    end: int
    color: str = "#EEEEEE"


@dataclass
class Marker:
    """A single marker annotation at a specific column.

    Attributes:
        col: 0-based alignment column index.
        label: Display label (e.g. "N332").
    """

    col: int
    label: str


@dataclass
class SeqGroup:
    """A named group of sequences (e.g. one animal/sample).

    Attributes:
        name: Group display label.
        seqs: List of (seq_id, bases) tuples.
    """

    name: str
    seqs: list[tuple[str, list[str]]]


@dataclass
class Panel:
    """One horizontal alignment block: a reference row + sequence rows.

    Attributes:
        label: Display name for the panel (e.g. filename stem or lineage).
        ref_row: Reference sequence as a list of single-character strings.
                 Used as the comparison base for match/mismatch coloring.
        seq_rows: Flat read/sequence rows as (name, bases) tuples.
        total_cols: Total number of display columns.
        col_labels: Tick positions and labels for the x-axis as (column_index, label) pairs.
        ins_columns: Column indices that represent insertion positions.
        regions: Optional region annotations for the header band.
        markers: Optional marker annotations (e.g. PNGS sites).
        marker_color: Color for marker dots and labels.
        groups: Optional grouped sequences (overrides seq_rows for rendering).
        title: Optional title displayed above the panel.
        extra_ref_rows: Additional reference-style rows rendered above
                        the primary ref_row as (label, bases) tuples.
    """

    label: str
    ref_row: list[str]
    seq_rows: list[tuple[str, list[str]]]
    total_cols: int
    col_labels: list[tuple[int, str]]
    ins_columns: set[int] | None = None
    regions: list[Region] | None = None
    markers: list[Marker] | None = None
    marker_color: str = "#4CAF50"
    groups: list[SeqGroup] | None = None
    title: str | None = None
    extra_ref_rows: list[tuple[str, list[str]]] | None = None
    extra_col_labels: list[tuple[int, str]] | None = None
    secondary_ref_row: list[str] | None = None
    heterologous_color: str = "#FF6F00"

    def __post_init__(self) -> None:
        if self.ins_columns is None:
            self.ins_columns = set()

    @property
    def total_seqs(self) -> int:
        """Total number of sequences across all groups or flat rows.

        Examples:
            >>> Panel("t", ["A"], [("s1", ["A"]), ("s2", ["A"])], 1, []).total_seqs
            2
            >>> Panel("t", ["A"], [], 1, [], groups=[SeqGroup("g", [("s1", ["A"])])]).total_seqs
            1
        """
        if self.groups:
            return sum(len(g.seqs) for g in self.groups)
        return len(self.seq_rows)

    @property
    def seq_type(self) -> str:
        """Detect sequence type from the reference row: 'AA' or 'NT'.

        Examples:
            >>> Panel("t", list("ACGTACGT"), [], 8, []).seq_type
            'NT'
            >>> Panel("t", list("MWLKFHRD"), [], 8, []).seq_type
            'AA'
            >>> Panel("t", list("ACG-T.NU"), [], 8, []).seq_type
            'NT'
        """
        nt_chars = set("ACGTUNacgtun-.")
        for base in self.ref_row:
            if base not in nt_chars:
                return "AA"
        return "NT"

    @property
    def effective_groups(self) -> list[SeqGroup]:
        """Return groups for rendering; wraps flat seq_rows if no groups set.

        Examples:
            >>> p = Panel("t", ["A"], [("s1", ["A"])], 1, [])
            >>> len(p.effective_groups)
            1
            >>> p.effective_groups[0].name
            ''
            >>> Panel("t", ["A"], [], 1, []).effective_groups
            []
        """
        if self.groups:
            return self.groups
        if self.seq_rows:
            return [SeqGroup(name="", seqs=self.seq_rows)]
        return []
