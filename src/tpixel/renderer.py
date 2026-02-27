"""Pixel-block rendering engine for alignment panels.

Renders Roark-style PIXEL plots with up to 7 layers:
  1. Title
  2. Region header (colored bands)
  3. Marker annotation row (dots + labels, staggered)
  4. Reference row (dark grey bar, white for gaps)
  5. Sequence group blocks (thin bars: grey=match, red=sub, black=gap)
  6. X-axis with tick labels
  7. Legend
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

from tpixel.models import Panel

# -- Roark 3-color scheme --------------------------------------------------
MATCH_COLOR = "#BDBDBD"
MISMATCH_COLOR = "#D32F2F"
GAP_COLOR = "#212121"
REF_COLOR = "#616161"

# -- Layout constants (data units, y-axis) ----------------------------------
REGION_HEADER_HEIGHT = 1.0
HEADER_MARKER_PAD = 0.8
MARKER_ZONE_HEIGHT = 2.0
MARKER_REF_PAD = 0.3
REF_ROW_HEIGHT = 0.5
REF_SEQ_PAD = 0.3
SEQ_DATA_ROW = 0.35
GROUP_DATA_GAP = 1.0


def panel_figsize(panel: Panel) -> tuple[float, float]:
    """Calculate the recommended figure size for a panel in inches.

    Args:
        panel: Panel object to calculate size for.

    Returns:
        Tuple of (width, height) in inches.
    """
    aln_len = panel.total_cols
    total_seqs = panel.total_seqs
    n_groups = len(panel.effective_groups)

    has_regions = bool(panel.regions)
    has_markers = bool(panel.markers)
    has_title = bool(panel.title)

    fig_width = max(6, aln_len / 100 + 2)

    title_h = 0.5 if has_title else 0.0
    region_h = 0.4 if has_regions else 0.0
    marker_h = 0.6 if has_markers else 0.0
    ref_h = 0.15
    axis_h = 0.5
    legend_h = 0.4

    seq_row_h = 0.02
    group_gap_h = 0.06
    seq_zone_h = total_seqs * seq_row_h + max(0, n_groups - 1) * group_gap_h

    total_h = max(
        3.0,
        title_h + region_h + marker_h + ref_h + axis_h + legend_h + seq_zone_h,
    )

    return fig_width, total_h


def plot_panel(panel: Panel, ax: Axes | None = None) -> Axes:
    """Render a single panel onto a matplotlib Axes.

    If *ax* is ``None`` a new figure is created with dimensions from
    :func:`panel_figsize`.  Works with any ``matplotlib.axes.Axes``
    subclass, including ``patchworklib.Brick``.

    Args:
        panel: Panel object to render.
        ax: Optional matplotlib Axes (or patchworklib Brick) to draw on.
            When ``None``, a new figure and axes are created.

    Returns:
        The Axes that was drawn on.
    """
    if ax is None:
        w, h = panel_figsize(panel)
        _fig, ax = plt.subplots(1, 1, figsize=(w, h))

    _draw_panel(panel, ax)
    return ax


def to_patchwork(panel: Panel, label: str = "tpixel") -> "pw.Brick":
    """Create a patchworklib Brick containing the rendered panel.

    Args:
        panel: Panel object to render.
        label: Unique label for the Brick (must differ between Bricks
            when composing with ``|`` or ``/``).

    Returns:
        A ``patchworklib.Brick`` ready for composition.
    """
    import patchworklib as pw

    w, h = panel_figsize(panel)
    brick = pw.Brick(figsize=(w, h), label=label)
    _draw_panel(panel, brick)
    return brick


def render_panels(
    panels: list[Panel],
    out_path: str | Path = "pixel.png",
    dpi: int = 300,
    cell: float | None = None,
) -> None:
    """Render alignment panels as Roark-style pixel-block plots.

    Supports the full 7-layer layout when panels provide regions,
    markers, and grouped sequences. Falls back to a simpler view
    for basic panels with only ref_row + seq_rows.

    Args:
        panels: List of Panel objects to render vertically.
        out_path: Output image path (format inferred from extension).
        dpi: Output resolution in dots per inch.
        cell: Cell size in inches (unused in Roark layout, kept for API compat).
    """
    out_path = Path(out_path)

    for panel in panels:
        _render_single_panel(panel, out_path, dpi)

    if len(panels) > 1:
        print(f"Saved: {out_path} ({dpi} dpi, {len(panels)} panel(s))")
    else:
        print(
            f"Saved: {out_path} ({dpi} dpi, {panels[0].total_cols} cols, "
            f"{panels[0].total_seqs} seqs)"
        )


def _render_single_panel(panel: Panel, out_path: Path, dpi: int) -> None:
    """Render one panel to a file, using the 7-layer Roark layout."""
    ax = plot_panel(panel)
    fig = ax.figure

    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix == ".pdf":
        with PdfPages(str(out_path)) as pdf:
            pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
    else:
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)


def _draw_panel(panel: Panel, ax: Axes) -> None:
    """Draw all 7 layers of a Roark-style panel onto *ax*."""
    aln_len = panel.total_cols
    groups = panel.effective_groups
    total_seqs = panel.total_seqs
    n_groups = len(groups)

    has_regions = bool(panel.regions)
    has_markers = bool(panel.markers)
    has_title = bool(panel.title)

    # -- Y coordinate system (data units, top=0 downward) --------------------
    y_cursor = 0.0

    # Marker zone (ABOVE region header so labels are readable)
    if has_markers:
        y_marker_top = y_cursor
        y_marker_bot = y_cursor + MARKER_ZONE_HEIGHT
        y_cursor = y_marker_bot + HEADER_MARKER_PAD
    else:
        y_marker_top = y_marker_bot = y_cursor

    # Region header
    y_region_top = y_cursor
    if has_regions:
        y_region_bot = y_cursor + REGION_HEADER_HEIGHT
        y_cursor = y_region_bot
    else:
        y_region_bot = y_cursor

    # Reference rows (extra refs like HxB2 rendered above the primary ref)
    n_extra_refs = len(panel.extra_ref_rows) if panel.extra_ref_rows else 0
    y_cursor += MARKER_REF_PAD if (has_markers or has_regions) else 0.2
    y_extra_ref_top = y_cursor
    if n_extra_refs:
        y_cursor += n_extra_refs * REF_ROW_HEIGHT + REF_SEQ_PAD
    y_ref_top = y_cursor
    y_ref_bot = y_cursor + REF_ROW_HEIGHT
    y_cursor = y_ref_bot

    # Sequence zone
    y_cursor += REF_SEQ_PAD
    y_seq_start = y_cursor

    seq_data_total = (
        total_seqs * SEQ_DATA_ROW + max(0, n_groups - 1) * GROUP_DATA_GAP
    )

    y_axis_pos = y_seq_start + seq_data_total + 0.5
    y_max = y_axis_pos + 2.0

    ax.set_xlim(-aln_len * 0.08, aln_len * 1.02)
    ax.set_ylim(y_max, -0.5 if has_title else -0.1)
    ax.set_axis_off()

    # -- Layer 1: Title -------------------------------------------------------
    if has_title:
        ax.text(
            aln_len / 2,
            -0.3,
            panel.title,
            fontsize=8,
            ha="center",
            va="bottom",
            fontweight="bold",
            color="#212121",
        )

    # -- Layer 2: Region header -----------------------------------------------
    if has_regions:
        for region in panel.regions:
            width = region.end - region.start
            ax.add_patch(
                Rectangle(
                    (region.start, y_region_top),
                    width,
                    y_region_bot - y_region_top,
                    facecolor=region.color,
                    edgecolor="#9E9E9E",
                    linewidth=0.3,
                )
            )
            if width > aln_len * 0.015:
                ax.text(
                    region.start + width / 2,
                    (y_region_top + y_region_bot) / 2,
                    region.name,
                    fontsize=5,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="#424242",
                )

    # -- Layer 3: Marker annotation row (above region header) -----------------
    if has_markers:
        y_dot_row = y_marker_bot - 0.3
        y_label_above = y_dot_row - 0.5
        y_label_below = y_dot_row + 0.5

        for idx, marker in enumerate(panel.markers):
            col = marker.col
            # Dashed vertical guide from dot down through region header into seq zone
            ax.plot(
                [col + 0.5, col + 0.5],
                [y_dot_row, y_seq_start + seq_data_total],
                color=panel.marker_color,
                linewidth=0.3,
                linestyle=":",
                alpha=0.4,
                zorder=0,
            )
            # Green dot
            ax.plot(
                col + 0.5,
                y_dot_row,
                marker="o",
                markersize=2.5,
                color=panel.marker_color,
                markeredgecolor=panel.marker_color,
                zorder=3,
            )
            # Horizontal label, alternating above/below dot
            if idx % 2 == 0:
                y_text = y_label_above
                va = "bottom"
            else:
                y_text = y_label_below
                va = "top"
            ax.text(
                col + 0.5,
                y_text,
                marker.label,
                fontsize=2.5,
                ha="center",
                va=va,
                rotation=0,
                color=panel.marker_color,
                fontweight="bold",
                clip_on=True,
            )

    # -- Layer 4: Reference rows -----------------------------------------------
    # Extra reference rows (e.g. HxB2) above the primary
    if panel.extra_ref_rows:
        y_eref = y_extra_ref_top
        for eref_label, eref_row in panel.extra_ref_rows:
            eref_bot = y_eref + REF_ROW_HEIGHT
            ax.add_patch(
                Rectangle(
                    (0, y_eref),
                    aln_len,
                    eref_bot - y_eref,
                    facecolor=REF_COLOR,
                    edgecolor="none",
                )
            )
            for i, base in enumerate(eref_row):
                if base == "-":
                    ax.add_patch(
                        Rectangle(
                            (i, y_eref),
                            1,
                            eref_bot - y_eref,
                            facecolor="white",
                            edgecolor="none",
                        )
                    )
            ax.text(
                -aln_len * 0.005,
                (y_eref + eref_bot) / 2,
                eref_label,
                fontsize=5,
                ha="right",
                va="center",
                fontweight="bold",
                color="#212121",
            )
            y_eref = eref_bot

    # Primary reference row (comparison base for sample sequences)
    ax.add_patch(
        Rectangle(
            (0, y_ref_top),
            aln_len,
            y_ref_bot - y_ref_top,
            facecolor=REF_COLOR,
            edgecolor="none",
        )
    )
    for i, base in enumerate(panel.ref_row):
        if base == "-":
            ax.add_patch(
                Rectangle(
                    (i, y_ref_top),
                    1,
                    y_ref_bot - y_ref_top,
                    facecolor="white",
                    edgecolor="none",
                )
            )

    ax.text(
        -aln_len * 0.005,
        (y_ref_top + y_ref_bot) / 2,
        panel.label,
        fontsize=5,
        ha="right",
        va="center",
        fontweight="bold",
        color="#212121",
    )

    # -- Layer 5: Sequence group blocks ----------------------------------------
    y_cursor = y_seq_start
    label_positions: list[tuple[float, str, int]] = []

    for group_idx, group in enumerate(groups):
        group_y_start = y_cursor

        for _seq_id, row in group.seqs:
            row_y = y_cursor

            # Grey background for entire row
            ax.add_patch(
                Rectangle(
                    (0, row_y),
                    aln_len,
                    SEQ_DATA_ROW * 0.85,
                    facecolor=MATCH_COLOR,
                    edgecolor="none",
                )
            )

            # Overdraw mutations and gaps
            for i, base in enumerate(row):
                if base == " ":
                    continue
                ref_base = panel.ref_row[i] if i < len(panel.ref_row) else "-"
                if base == "-":
                    color = GAP_COLOR
                elif base == ref_base:
                    continue
                else:
                    color = MISMATCH_COLOR
                ax.add_patch(
                    Rectangle(
                        (i, row_y),
                        1,
                        SEQ_DATA_ROW * 0.85,
                        facecolor=color,
                        edgecolor="none",
                    )
                )

            y_cursor += SEQ_DATA_ROW

        group_y_center = (group_y_start + y_cursor) / 2
        if group.name:
            label_positions.append((group_y_center, group.name, len(group.seqs)))

        if group_idx < n_groups - 1:
            y_cursor += GROUP_DATA_GAP

    for y_center, name, count in label_positions:
        label = f"{name} ({count})"
        ax.text(
            -aln_len * 0.005,
            y_center,
            label,
            fontsize=4,
            ha="right",
            va="center",
            color="#424242",
        )

    # -- Layer 6: X-axis ticks -------------------------------------------------
    for col_idx, label in panel.col_labels:
        ax.plot(
            [col_idx + 0.5, col_idx + 0.5],
            [y_axis_pos - 0.2, y_axis_pos + 0.1],
            color="#424242",
            linewidth=0.5,
        )
        ax.text(
            col_idx + 0.5,
            y_axis_pos + 0.2,
            label,
            fontsize=4,
            ha="center",
            va="top",
            color="#424242",
        )

    # -- Layer 7: Legend -------------------------------------------------------
    legend_y = y_axis_pos + 1.2
    legend_items = [
        ("Match", MATCH_COLOR),
        ("Substitution", MISMATCH_COLOR),
        ("Gap/Indel", GAP_COLOR),
    ]
    if has_markers:
        legend_items.append(("Marker", panel.marker_color))

    legend_x_start = aln_len * 0.25
    legend_spacing = aln_len * 0.15

    for idx, (label, color) in enumerate(legend_items):
        x = legend_x_start + idx * legend_spacing
        ax.add_patch(
            Rectangle(
                (x, legend_y),
                aln_len * 0.015,
                0.4,
                facecolor=color,
                edgecolor="#9E9E9E",
                linewidth=0.3,
            )
        )
        ax.text(
            x + aln_len * 0.02,
            legend_y + 0.2,
            label,
            fontsize=5,
            ha="left",
            va="center",
            color="#424242",
        )

    # Stats summary on the right side of the legend
    stats = (
        f"{total_seqs} sequences, "
        f"{n_groups} samples, "
        f"{aln_len} positions, {panel.seq_type}"
    )
    ax.text(
        aln_len * 1.0,
        legend_y + 0.2,
        stats,
        fontsize=5,
        ha="right",
        va="center",
        color="#757575",
    )
