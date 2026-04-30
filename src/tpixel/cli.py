"""Click CLI for tpixel."""

import sys

import click

from tpixel.anchors import KNOWN_ANCHOR_LINEAGES, detect_anchor_lineage
from tpixel.fasta import fasta_panel, read_fasta
from tpixel.renderer import render_panels


def _expand_stdin(paths: list[str]) -> list[str]:
    """If paths is ``['-']``, read file paths from stdin (one per line).

    Args:
        paths: List of file path strings. A single ``'-'`` triggers stdin reading.

    Returns:
        Expanded list of file paths.

    Examples:
        >>> _expand_stdin(["file1.fasta", "file2.fasta"])
        ['file1.fasta', 'file2.fasta']
        >>> _expand_stdin([])
        []
    """
    if paths and len(paths) == 1 and paths[0] == "-":
        return [line.strip() for line in sys.stdin if line.strip()]
    return list(paths)


def _auto_detect_hiv(fasta_path: str) -> bool:
    """Check if alignment qualifies for HIV mode.

    HIV mode is triggered when either:

    * the alignment contains an ``HxB2`` row plus a ``*_ref`` row
      (classic dual-reference layout), or
    * the alignment lacks ``HxB2`` but contains a ``*_ref`` row whose
      lineage prefix matches a known anchor (e.g. ``SF162p3_ref``,
      ``CH505_ref``, ``T250-4_ref``) so the renderer can still place the
      region bar via the bundled lineage→HxB2 mapping.

    Args:
        fasta_path: Path to the aligned FASTA file.

    Returns:
        ``True`` if HIV mode applies.
    """
    seqs = read_fasta(fasta_path)
    names = {n.split()[0] for n, _ in seqs}
    has_hxb2 = "HxB2" in names
    refs = [n for n in names if n.endswith("_ref")]
    if has_hxb2 and refs:
        return True
    return any(detect_anchor_lineage(n) is not None for n in refs)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog="Positional args and --fasta are combined. Use '-' for stdin:\n\n"
    "  tpixel alignment.fasta -o out.png\n"
    "  find . -name '*.fasta' | tpixel - -o out.png",
)
@click.argument("fasta_args", nargs=-1)
@click.option(
    "--fasta",
    multiple=True,
    help="Aligned FASTA file(s) — each becomes a panel. Use '-' for stdin.",
)
@click.option(
    "--columns", help="Column range for FASTA, 1-based inclusive (e.g. 1-120)."
)
@click.option(
    "-o",
    "--output",
    default="pixel.png",
    show_default=True,
    help="Output image path.",
)
@click.option(
    "--dpi", type=int, default=300, show_default=True, help="Image resolution."
)
@click.option(
    "--cell", type=float, default=None, help="Cell size in inches (default: 0.03)."
)
@click.option(
    "--hiv/--no-hiv",
    default=None,
    help="Force HIV mode (HxB2 regions, PNGS, animal grouping). Auto-detected if omitted.",
)
@click.option(
    "--nt/--aa",
    default=None,
    help="Force nucleotide or amino-acid mode. Auto-detected if omitted.",
)
@click.option(
    "--ref-pos",
    default="1,2",
    show_default=True,
    help="Comma-separated 1-based positions of reference sequences. "
    "Last position is the primary reference; earlier ones are extra reference rows.",
)
@click.option(
    "--title",
    default=None,
    help="Title displayed above the plot.",
)
@click.option(
    "--variant-labels/--no-variant-labels",
    "variant_labels",
    default=False,
    show_default=True,
    help="Draw 'wildtype+pos+mutation' labels (e.g. K169E) under the x-axis "
    "for every column where the lineage _ref differs from HxB2. HIV mode only. "
    "Works in anchor mode too — labels are computed against the bundled HxB2 "
    "residues carried by the lineage→HxB2 mapping.",
)
@click.option(
    "--anchor-id",
    default=None,
    help="Sequence ID to use as the header coordinate anchor when HxB2 is "
    "absent from the alignment. Defaults to the primary _ref row.",
)
@click.option(
    "--anchor-lineage",
    default=None,
    type=click.Choice(list(KNOWN_ANCHOR_LINEAGES)),
    help="Anchor lineage. Auto-detected from the anchor-id prefix if omitted "
    "(e.g. SF162p3_ref → SF162p3). Ignored when HxB2 is in the alignment.",
)
@click.option(
    "--markers/--no-markers",
    "markers",
    default=True,
    show_default=True,
    help="Show annotation markers above the reference row (currently PNGS "
    "green dots in HIV mode). Use --no-markers to suppress them.",
)
def main(
    fasta_args,
    fasta,
    columns,
    output,
    dpi,
    cell,
    hiv,
    nt,
    ref_pos,
    title,
    variant_labels,
    markers,
    anchor_id,
    anchor_lineage,
):
    """Pixel-block alignment viewer for hundreds of sequences.

    Renders Roark-style PIXEL plots: grey=match, red=substitution, black=gap.
    Each sequence is a thin row of colored blocks — no text in cells.

    HIV mode is auto-detected when the alignment contains HxB2 and a *_ref
    sequence. Force with --hiv or --no-hiv.
    """
    fasta_paths = _expand_stdin(list(fasta_args) + list(fasta))

    if not fasta_paths:
        raise click.UsageError("Provide at least one FASTA file")

    ref_positions = [int(x) for x in ref_pos.split(",")]

    panels = []
    col_start, col_end = None, None
    if columns:
        parts = columns.replace(",", "").split("-")
        col_start = int(parts[0])
        col_end = int(parts[1]) if len(parts) > 1 else None

    for fasta_path in fasta_paths:
        use_hiv = hiv if hiv is not None else _auto_detect_hiv(fasta_path)

        if use_hiv:
            from tpixel.hiv import hiv_panel

            seq_type = None
            if nt is True:
                seq_type = "NT"
            elif nt is False:
                seq_type = "AA"
            panel = hiv_panel(
                fasta_path,
                ref_positions=ref_positions,
                seq_type=seq_type,
                show_variant_labels=variant_labels,
                show_markers=markers,
                anchor_id=anchor_id,
                anchor_lineage=anchor_lineage,
            )
        else:
            if variant_labels:
                raise click.UsageError(
                    "--variant-labels requires HIV mode"
                )
            panel = fasta_panel(fasta_path, col_start, col_end, ref_positions=ref_positions)

        if title:
            panel.title = title
        panels.append(panel)

    render_panels(panels, output, dpi=dpi, cell=cell)
