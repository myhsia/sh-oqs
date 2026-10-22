import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from typing import TypeAlias
matplotlib.rcParams.update({
    "font.size": 15, "text.usetex": True,
    "text.latex.preamble": r'''\usepackage{physics3}
                               \usephxmodu{ab.braket, operator}
                               \usepackage[mono = false]{libertine}
                               \usepackage{libertinust1math}''',
    "lines.solid_capstyle":  "round",
    "lines.dash_capstyle":   "round",
    "axes.formatter.use_mathtext": True,
})
PlotLine: TypeAlias = (
    tuple[np.ndarray, np.ndarray, str, str, float, float, str]
  | tuple[np.ndarray, np.ndarray, str, str, float, float, list[float], str]
)
LogLine: TypeAlias = (
    tuple[np.ndarray, np.ndarray, str, str, float, str, float, str]
)


def parse_xlims(raw_xlims, reorder: bool = True):
    """Normalize YAML ``xlims`` into ``[[x_lo, x_hi], x_step, ...]``.

    When ``reorder`` is true and two windows are given, the two groups are
    swapped so that ``xlims[0]`` is the larger (later) window, i.e.
    ``xlims[0][1]`` is the maximum time (used for the calculation range).
    """
    xlims = [tuple(x) if isinstance(x, list) else x for x in raw_xlims]
    if reorder and len(xlims) == 4:
        xlims = [xlims[2], xlims[3], xlims[0], xlims[1]]
    return xlims


class GridPlt:
    # Side labels are pinned by their baseline this many points from the spine.
    # Baseline pinning is glyph-independent, so a descender (e.g. the tail of a
    # subscript ``y``) cannot shift the label the way extent-based alignment
    # does; matplotlib's automatic label placement is bypassed entirely.
    LABEL_BASE_PT = 10.3
    # Figure width/heights (in): 16:1 legend bar, 16:9 legend-free bodies,
    # 16:10 (8:5) kernel page (which carries its own legend).
    W        = 7.2
    H_BAR    = W / 16
    H_BODY   = 9  * W / 16
    H_LEGEND = 10 * W / 16
    # Shared style for the small rounded tags (Re/Im and table frames).
    TAG_BBOX = dict(boxstyle = "round, pad = .20",
                    facecolor = "gray", alpha = .25)

    @staticmethod
    def round_box(ax, xy, width, height, rounding_size: float = 0.015,
                  transform = None, **kwargs):
        """Rounded box whose corners are circles, not ellipses.

        ``rounding_size`` is measured in x-units of ``transform`` (the axes
        fractions by default); ``mutation_aspect`` corrects the y corners for
        the display aspect ratio, so the radius is the same in both directions
        at any position of the axes (data or axes coordinates).
        """
        t = transform if transform is not None else ax.transAxes
        o = t.transform((0.0, 0.0))
        sx = float(np.hypot(*(t.transform((1.0, 0.0)) - o)))
        sy = float(np.hypot(*(t.transform((0.0, 1.0)) - o)))
        patch = FancyBboxPatch(
            xy, width, height,
            boxstyle = f"round,pad=0,rounding_size={rounding_size}",
            mutation_aspect = sx / sy,
            transform = t, **kwargs)
        ax.add_patch(patch)
        return patch

    @staticmethod
    def _new_grid(gridspec_kw: dict, figsize = None, nrow: int = 2):
        return plt.subplots(nrow, 2, sharey = 'row', sharex = 'col',
                            figsize  = figsize,
                            gridspec_kw  = gridspec_kw)

    @staticmethod
    def _style_axes(ax) -> None:
        ax.ticklabel_format(style = 'sci', scilimits   = (-2, 2),
                            axis  = 'y',   useMathText = True)
        ax.axhline(0, color = 'k', ls = '--', lw = 1)

    @staticmethod
    def _plot_lines(ax, lines: list[PlotLine], take_real: bool) -> None:
        for line in lines:
            x, y, color, ls, lw, alpha = line[:6]
            label = line[7] if len(line) > 7 else line[6]
            data_part = y.real if take_real else y.imag
            ax.plot(x,  data_part, ls = ls, label = label,
                    color = color, lw = lw, alpha = alpha)

    @staticmethod
    def _group_ylim(lines: list[PlotLine]) -> list[float] | None:
        for line in lines:
            if len(line) > 7:
                return line[6]
        return None

    @staticmethod
    def _cut_windows(xlims) -> tuple:
        """The two display windows (earlier left, later right) and their
        combined xticks, from a 4-entry calculation-order ``xlims``."""
        x_lo, s_lo, x_hi, s_hi = xlims[2], xlims[3], xlims[0], xlims[1]
        windows = [x_lo, x_hi]
        xticks = np.concatenate((
            np.arange( x_lo[0],  int(x_lo[1] + 1),
                      (x_lo[1] - x_lo[0]) / s_lo),
           -np.arange(-x_hi[1], -int(x_hi[0] - 1),
                      (x_hi[1] - x_hi[0]) / s_hi)))
        return windows, xticks

    @staticmethod
    def _normalize(ax, ylim: list[float] | None = None) -> None:
        if ylim is None:
            return
        lo, hi, n = ylim
        step = (hi - lo) / n
        pad  = 0.025 * (hi - lo)
        ax.set_yticks(np.arange(lo, hi + step, step))
        ax.set_ylim(lo - pad, hi + pad)

    @staticmethod
    def _align_corner_badges(fig, badges) -> None:
        if not badges:
            return
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for txt, ax, corner in badges:
            patch = txt.get_bbox_patch()
            if patch is not None:
                p_ext = patch.get_window_extent(renderer)
                a_ext = ax.get_window_extent()
                mx, my = 0.025 * a_ext.width, 0.025 * a_ext.height
                if   corner == 'SE':
                    dx, dy = (a_ext.x1 - mx) - p_ext.x1, (a_ext.y0 + my) - p_ext.y0
                elif corner == 'SW':
                    dx, dy = (a_ext.x0 + mx) - p_ext.x0, (a_ext.y0 + my) - p_ext.y0
                elif corner == 'NE':
                    dx, dy = (a_ext.x1 - mx) - p_ext.x1, (a_ext.y1 - my) - p_ext.y1
                else:  # NW
                    dx, dy = (a_ext.x0 + mx) - p_ext.x0, (a_ext.y1 - my) - p_ext.y1
                x, y = txt.get_position()
                txt.set_position((x + dx / a_ext.width, y + dy / a_ext.height))

    @staticmethod
    def _anchor_end_xticks(ax, x_min: float, x_max: float) -> None:
        """Anchor the first x-label's west edge and the last x-label's east
        edge to their tick markers (interior labels stay centred), so a
        trailing '8' and the next column's leading '0' do not merge."""
        for v, tl in zip(ax.get_xticks(), ax.get_xticklabels()):
            if np.isclose(v, x_min):
                tl.set_horizontalalignment("left")
            elif np.isclose(v, x_max):
                tl.set_horizontalalignment("right")

    @staticmethod
    def _reimgrid_panels(axs, biylabel: list[str],
                         plot_data: list[list[PlotLine]],
                         x_min: float, x_max: float,
                         xticks: np.ndarray) -> list:
        """Draw the Re/Im grid (one quantity per row) onto ``axs`` and return
        the badge list; ``axs`` has shape ``(len(plot_data), 2)``."""
        badges = []
        for row in range(len(plot_data)):
            for col in range(2):
                ax = axs[row, col]
                # Parametric Title Positioning
                title_text = r'$\Re$'        if col == 0 else r'$\Im$'
                x_pos, ha = (0.96, 'right')  if col == 0 else (0.04, 'left')
                y_pos, va = (0.00, 'bottom') if row == 0 else (0.88, 'top')
                corner = ('SE' if col == 0 else 'SW') if row == 0 \
                    else ('NE' if col == 0 else 'NW')
                ax.set_title(title_text, x = x_pos, ha = ha,
                                         y = y_pos, va = va,
                             bbox = GridPlt.TAG_BBOX)
                badges.append((ax.title, ax, corner))
                # Axis Limits & Formatting
                ax.set_xlim(x_min, x_max)
                ax.set_xticks(xticks)
                ax.xaxis.set_tick_params(labelbottom = True)
                GridPlt._anchor_end_xticks(ax, x_min, x_max)
                GridPlt._style_axes(ax)
                # Plot lines for the current row
                GridPlt._plot_lines(ax, plot_data[row], col == 0)
                # Left vs Right column formatting
                if col != 0:
                    ax.set_ylabel(biylabel[row],
                                  rotation = 270, labelpad = 15)
                    ax.yaxis.set_label_position("right")
                    ax.tick_params(left = False, labelleft = False)
            # Normalize
            GridPlt._normalize(ax, GridPlt._group_ylim(plot_data[row]))
        return badges

    @staticmethod
    def _baseline_frac(ax, side: str, offset_pt: float) -> float:
        """Axes-fraction x for a baseline pinned ``offset_pt`` points from the
        ``side`` spine of ``ax``."""
        fig = ax.figure
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        ext = ax.get_window_extent(r)
        sp = ext.x0 if side == "left" else ext.x1
        dx = offset_pt * fig.dpi / 72
        x_disp = sp - dx if side == "left" else sp + dx
        return ax.transAxes.inverted().transform((x_disp, 0))[0]

    @staticmethod
    def _pin_ylabel(ax, side: str | None = None,
                    offset_pt: float | None = None) -> None:
        """Pin ``ax``'s y-label baseline ``offset_pt`` points from its spine."""
        lbl = ax.yaxis.get_label()
        lbl.set_rotation_mode("anchor")
        lbl.set_va("baseline")
        lbl.set_ha("center")
        side = side or ax.yaxis.get_label_position()
        ax.yaxis.set_label_coords(
            GridPlt._baseline_frac(ax, side,
                                   offset_pt or GridPlt.LABEL_BASE_PT), 0.5)
        ax.figure.canvas.draw()

    @staticmethod
    def _pin_text(ax, txt, side: str = "right",
                  offset_pt: float | None = None) -> None:
        """Pin a side ``ax.text`` baseline ``offset_pt`` points from the spine."""
        txt.set_rotation_mode("anchor")
        txt.set_va("baseline")
        txt.set_ha("center")
        x = GridPlt._baseline_frac(ax, side,
                                   offset_pt or GridPlt.LABEL_BASE_PT)
        txt.set_position((x, txt.get_position()[1]))
        ax.figure.canvas.draw()

    @staticmethod
    def _symmetric_side_labels(fig, ax_pop, right_axs,
                               offset_pt: float | None = None) -> None:
        """Pin the population label (left) and the right y-labels by their
        baselines at ``GridPlt.LABEL_BASE_PT`` points from their spines."""
        GridPlt._pin_ylabel(ax_pop, "left", offset_pt)
        for ax in right_axs:
            GridPlt._pin_ylabel(ax, "right", offset_pt)

    @staticmethod
    def _pop_panel(ax, lines: list[PlotLine], xlim, xticks,
                   anchor: bool = True) -> None:
        """Draw a population panel: 0..1 y-range with the given x-range/ticks."""
        GridPlt._plot_lines(ax, lines, take_real = True)
        ax.set_xticks(xticks)
        ax.set_xlim(*xlim)
        if anchor:
            GridPlt._anchor_end_xticks(ax, *xlim)
        ax.set_ylim(-.05, 1.05)
        ax.set_yticks(np.arange(0.0, 1.5, .5))

    @staticmethod
    def _pop_split_labels(ax, side: str = "left", rotation: int = 90) -> None:
        """Two population part-labels, each centred between ticks (proj1 in the
        1.0-0.5 gap, proj0 in the 0.5-0.0 gap), baseline-pinned on ``side``."""
        lo, hi = ax.get_ylim()
        yf = lambda v: (v - lo) / (hi - lo)
        for v, lbl in ((0.75, r"$\rho_{11}(t)$"),
                       (0.25, r"$\rho_{00}(t)$")):
            txt = ax.text(0.0, yf(v), lbl, transform = ax.transAxes,
                          rotation = rotation, va = "baseline", ha = "center")
            GridPlt._pin_text(ax, txt, side)

    @staticmethod
    def _save_bar_and_body(fig, span_axs, handle_ax, pdf,
                           supxlabel: str, close_fig: bool = True,
                           emit_bar: bool = True) -> None:
        """Save the legend bar (from ``handle_ax``'s handles and the ``span_axs``
        x-span) as the first page, then the legend-free body.

        ``emit_bar`` can be turned off for a body whose legend is emitted
        separately."""
        fig.supxlabel(supxlabel, y = 0)
        if emit_bar:
            handles, labels = handle_ax.get_legend_handles_labels()
            x0, x1 = GridPlt._legend_xy(span_axs)
            GridPlt._legend_bar(handles, labels, x0, x1, pdf)
        if pdf is not None:
            pdf.savefig(fig)
        if close_fig:
            plt.close(fig)

    @staticmethod
    def _legend_xy(span_axs) -> tuple:
        """Figure-fraction x-span of the axes the top legend should cover."""
        xs0 = [a.get_position().x0 for a in span_axs]
        xs1 = [a.get_position().x1 for a in span_axs]
        return min(xs0), max(xs1)

    @staticmethod
    def _legend_bar(handles, labels, x0: float, x1: float,
                    pdf: PdfPages | None = None, close_fig: bool = True,
                    legend_kwargs: dict | None = None) -> None:
        """Save a 7.2x0.45 (16:1) legend-only page spanning ``x0..x1`` (figure
        fraction), so it lines up with the 7.2-in-wide body when included."""
        fig = plt.figure(figsize = (GridPlt.W, GridPlt.H_BAR))
        fig.legend(handles, labels, loc = 'center', mode = 'expand',
                   borderaxespad = 0, ncol = len(labels),
                   bbox_to_anchor = (x0, 0.0, x1 - x0, 1.0),
                   **(legend_kwargs or {}))
        if pdf is not None:
            pdf.savefig(fig)
        if close_fig:
            plt.close(fig)

    @staticmethod
    def semilog_page(pdf, lines: list[LogLine], xlabel: str, ylabel: str,
                     x_ticks = None, legend_kwargs: dict | None = None,
                     decorate = None, close_fig: bool = True) -> None:
        """One-page semilog (log y) figure on the standard 16:10 canvas.

        ``lines`` is a list of ``(x, y, color, ls, lw, marker, ms, label)``
        specs; the x-limits follow the first line's x-range and ``x_ticks``
        optionally overrides the ticks.  ``legend_kwargs`` configures the top
        legend (``None`` omits it).  ``decorate(ax)`` is invoked just before
        saving for caller-specific annotations."""
        fig = plt.figure(figsize = (GridPlt.W, GridPlt.H_LEGEND))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left = 0.08, right = 0.92)
        for x, y, color, ls, lw, marker, ms, label in lines:
            ax.semilogy(x, y, color = color, ls = ls, lw = lw,
                        marker = marker, ms = ms, label = label)
        ax.set_xlabel(xlabel)
        xs = lines[0][0]
        ax.set_xlim(xs[0], xs[-1])
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{np.log10(v):.0f}"))
        ax.set_ylabel(ylabel, rotation = 270, labelpad = 15)
        ax.yaxis.set_label_position("right")
        if legend_kwargs is not None:
            plt.legend(mode = "expand", bbox_to_anchor = (0, 1.02, 1, 0.2),
                       loc = "lower left", borderaxespad = 0,
                       **legend_kwargs)
        if decorate is not None:
            decorate(ax)
        GridPlt._pin_ylabel(ax, "right")
        if pdf is not None:
            pdf.savefig(fig)
        if close_fig:
            plt.close(fig)

    @staticmethod
    def _legend(fig, span_axs, handle_ax, supxlabel: str) -> tuple:
        """Add a full-width top legend spanning ``span_axs``; returns the
        legend's handles/labels and its x-span (for building the legend bar)."""
        fig.supxlabel(supxlabel, y = 0)
        handles, labels = handle_ax.get_legend_handles_labels()
        x0, x1 = GridPlt._legend_xy(span_axs)
        y1 = max(a.get_position().y1 for a in span_axs)
        y_off = 0.02 * (plt.rcParams["figure.subplot.top"] -
                        plt.rcParams["figure.subplot.bottom"])
        fig.legend(handles, labels, loc = 'lower left',
                    mode = 'expand', borderaxespad = 0,
                    ncol = len(labels),
                    bbox_to_anchor = (x0, y1 + y_off, x1 - x0, 0.2))
        return handles, labels, x0, x1

    @staticmethod
    def _finalize(fig, axs, supxlabel: str,
                  pdf: PdfPages | None, close_fig: bool,
                  legend: bool = True, pin_axes = None) -> None:
        if legend:
            GridPlt._legend(fig, [axs[0, 0], axs[0, 1]], axs[0, 0], supxlabel)
        else:
            fig.supxlabel(supxlabel, y = 0)
        # Pin the right labels by their baseline (no extent-based auto-adjust)
        for ax in (pin_axes or []):
            GridPlt._pin_ylabel(ax, "right")
        if pdf is not None:
            pdf.savefig(fig)
        if close_fig:
            plt.close(fig)

    @staticmethod
    def reim4gridplt(
        biylabel: list[str],
        supxlabel: str,
        xlims: list[tuple[float, float] | float],
        plot_data: list[list[PlotLine]],
        pop_lines: list[PlotLine] | None = None,
        pdf: PdfPages | None = None,
        close_fig: bool = True,
        emit_bar: bool = True,
    ) -> tuple:
        """Plot the real/imaginary parts for one or more quantities.

        The figure layout is selected by the shape of ``xlims``:
          - 2 entries ``[[x_min, x_max], x_step]``: one single-window figure
            with a 2x2 grid (rows = Re/Im) for all quantities; returns
            ``(fig, axs)``. When ``pop_lines`` is given, the page instead holds
            a population panel on the left and the 2x2 grid on the right,
            sharing one figure-level legend.
          - 4 entries ``[[x_lo, x_hi], step_lo, [x_lo2, x_hi2], step_hi2]``,
            given in calculation order (larger window first): one two-window
            figure per quantity in ``plot_data`` (rows = Re/Im, cols = the two
            time windows, displayed earlier-left/later-right); returns
            ``(figs, axss)``.
        """
        if len(xlims) not in (2, 4):
            raise ValueError(
                f"xlims must have 2 (single-window) or 4 (two-window) entries, "
                f"got {len(xlims)}")
        two_windows = (len(xlims) == 4)
        if not two_windows:
            # 1. Unpack xlims cleanly
            (x_min, x_max), x_step = xlims
            xticks = np.arange(x_min, x_max + (x_max - x_min) / x_step,
                              (x_max - x_min) / x_step)
            if pop_lines is None:
                nrow = len(plot_data)
                fig, axs = GridPlt._new_grid(
                    {'wspace': 0.1, 'hspace': 0.2, 'width_ratios': [1, 1],
                     'left': 0.08, 'right': 0.92, 'top': 0.96, 'bottom': 0.12},
                    figsize = (GridPlt.W, GridPlt.H_BODY),
                    nrow = nrow
                )
                axs = np.atleast_2d(axs)
                badges = GridPlt._reimgrid_panels(
                    axs, biylabel, plot_data, x_min, x_max, xticks)
                if nrow > 1:
                    for col in range(2):
                        axs[0, col].xaxis.set_tick_params(labelbottom = False,
                                                          bottom = False)
                GridPlt._align_corner_badges(fig, badges)
                GridPlt._finalize(fig, axs, supxlabel, pdf, close_fig,
                                  legend = False,
                                  pin_axes = [axs[0, 1], axs[-1, 1]])
                return fig, axs
            # Combined page: population (left) | 2x2 Re/Im grid (right)
            fig = plt.figure(figsize = (GridPlt.W, GridPlt.H_BODY))
            gs = fig.add_gridspec(2, 3, width_ratios = [1, 1, 1],
                                  wspace = 0.1, hspace = 0.1,
                                  left = 0.08, right = 0.92,
                                  top = 0.96, bottom = 0.12)
            ax_pop = fig.add_subplot(gs[:, 0])
            axs = np.empty((2, 2), dtype = object)
            for row in range(2):
                for col in range(2):
                    kw = {}
                    if col > 0:
                        kw['sharey'] = axs[row, 0]
                    if row > 0:
                        kw['sharex'] = axs[0, col]
                    axs[row, col] = fig.add_subplot(gs[row, 1 + col], **kw)
            badges = GridPlt._reimgrid_panels(
                axs, biylabel, plot_data, x_min, x_max, xticks)
            for col in range(2):
                axs[0, col].xaxis.set_tick_params(labelbottom = False,
                                                  bottom = False)
            # Population panel (left)
            GridPlt._pop_panel(ax_pop, pop_lines, (x_min, x_max), xticks)
            GridPlt._pop_split_labels(ax_pop, "left", 90)
            for ax in (axs[0, 1], axs[1, 1]):
                GridPlt._pin_ylabel(ax, "right")
            # Move the grid y-ticks from the 2nd column's left to the 3rd
            # column's right (-1 and 1 only), at the default pad so the numbers
            # hug the y-label as on the 1st column.
            for row in range(2):
                axs[row, 0].tick_params(left = False, labelleft = False)
                axl = axs[row, 1]
                axl.set_yticks([-1, 1])
                axl.yaxis.set_major_formatter(
                    FuncFormatter(lambda v, _: f"${v:+g}$"))
                axl.tick_params(left = False, labelleft = False,
                                right = True, labelright = True)
            GridPlt._align_corner_badges(fig, badges)
            GridPlt._save_bar_and_body(fig, [ax_pop, axs[0, 1]], axs[0, 0],
                                       pdf, supxlabel, close_fig,
                                       emit_bar = emit_bar)
            return fig, axs
        # ``xlims`` arrives in calculation order (larger window first); swap
        # the two windows back so the earlier one is displayed on the left.
        windows, xticks = GridPlt._cut_windows(xlims)
        # Optional population page (1x2, one panel per window), saved with its
        # legend bar before the correlation figures.
        if pop_lines is not None:
            figp, axsp = plt.subplots(
                1, 2, sharey = True, figsize = (GridPlt.W, GridPlt.H_BODY),
                gridspec_kw = {'left': 0.08, 'right': 0.92, 'top': 0.96,
                               'bottom': 0.12, 'wspace': 0.05,
                               'width_ratios': [3, 2]})
            for col in range(2):
                GridPlt._pop_panel(axsp[col], pop_lines, windows[col], xticks,
                                   anchor = False)
            axsp[1].tick_params(left = False)
            GridPlt._pop_split_labels(axsp[1], "right", 270)
            GridPlt._save_bar_and_body(figp, list(axsp), axsp[0], pdf, supxlabel,
                                       emit_bar = emit_bar)
        figs = []
        axss = []
        # Create one two-window 2x2 figure per quantity (one pdf page each)
        for q, lines in enumerate(plot_data):
            ylim = GridPlt._group_ylim(lines)
            fig, axs = GridPlt._new_grid(
                {'wspace': .05, 'hspace': 0, 'width_ratios': [3, 2],
                 'left': 0.08, 'right': 0.92, 'top': 0.96, 'bottom': 0.12},
                figsize = (GridPlt.W, GridPlt.H_BODY)
            )
            # Strip the outer "$...$" wrapper to rebuild Re/Im labels
            body = biylabel[q][1:-1] \
                if biylabel[q].startswith('$') else biylabel[q]
            # 2. Iterate directly over rows and columns
            for row in range(2):
                for col in range(2):
                    ax = axs[row, col]
                    # Axis Limits & Formatting
                    GridPlt._style_axes(ax)
                    # Plot lines for the current row
                    GridPlt._plot_lines(ax, lines, row == 0)
                    ax.set_xticks(xticks)
                    ax.set_xlim(windows[col])
                    # Left vs Right column formatting
                    if col == 0:
                        sign = r'\Re' if row == 0 else r'\Im'
                    else:
                        ax.set_ylabel('$' + sign + '[' + body + ']$',
                                      rotation = 270, labelpad = 18)
                        ax.yaxis.set_label_position("right")
                        ax.tick_params(left = False)
                if ylim is not None:
                    GridPlt._normalize(ax, ylim)
            # Normalize
            if ylim is None:
                GridPlt._normalize(ax)
            # Hide the top row's x-ticks (shared with the lower row)
            for col in range(2):
                axs[0, col].xaxis.set_tick_params(labelbottom = False,
                                                  bottom = False)
            # 3. Figure-level adjustments, legend, save, and close
            GridPlt._finalize(fig, axs, supxlabel, pdf, close_fig,
                              legend = False,
                              pin_axes = [axs[0, 1], axs[1, 1]])
            figs.append(fig)
            axss.append(axs)
        return figs, axss
