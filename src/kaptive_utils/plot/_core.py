from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from enum import Enum, IntEnum, auto
from itertools import cycle
from re import compile as regex
from typing import Optional, Self, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from kaptive.database import Database
from kaptive.typing import TypingResult


# Enums ----------------------------------------------------------------------------------------------------------------
class GeneStatus(Enum):
    """
    Represents the structural status of a gene within an assembly.

    Attributes:
        NORMAL: Gene is complete and intact.
        TRUNCATED: Gene has internal frameshifts or stop codons.
        PARTIAL: Gene is cut off by the end of a contig.
    """
    NORMAL = auto()
    TRUNCATED = auto()  # Internal frameshift/null; gets dotted border
    PARTIAL = auto()  # Touches piece boundary; gets rectangle shape


class GenePresence(Enum):
    """
    Indicates if a gene was expected in the locus or is an unexpected insertion.

    Attributes:
        EXPECTED: Gene is part of the reference locus.
        UNEXPECTED: Gene is an insertion not present in the reference.
    """
    EXPECTED = auto()
    UNEXPECTED = auto()


class Strand(IntEnum):
    """
    Enumeration for genomic strands.

    Attributes:
        FORWARD: Positive strand (+).
        REVERSE: Negative strand (-).
        UNSTRANDED: Unknown or irrelevant strand (.).
    """
    FORWARD = 1
    REVERSE = -1
    UNSTRANDED = 0

    @classmethod
    def from_symbol(cls, symbol: str) -> Self:
        """
        Converts a string symbol to a Strand enum member.

        Args:
            symbol: The strand symbol ('+', '-', or '.').

        Returns:
            The corresponding Strand member.
        """
        if symbol == '+': return cls.FORWARD
        if symbol == '-': return cls.REVERSE
        return cls.UNSTRANDED

    def __str__(self) -> str:
        if self == Strand.FORWARD: return '+'
        if self == Strand.REVERSE: return '-'
        return '.'


# Classes --------------------------------------------------------------------------------------------------------------
class GeneColourMap:
    """
    Manages colour assignments for genes based on names and anchors.

    Assigns consistent colours to known 'anchor' genes and cycles through a
    colourblind-friendly palette for others.
    """
    _COLOURBLIND_FRIENDLY = ['#8dd3c7', '#ffffb3', '#bebada', '#80b1d3','#fdb462', '#b3de69', '#fccde5', '#bc80bd',
                             '#ccebc5', '#ffed6f']
    _GENE_REGEX = regex(r'[a-zA-Z]+')
    __slots__ = ('_map', '_anchors', '_null', '_cycler')
    def __init__(self, anchors: dict[str, str] = None, null: str = '#d3d3d3'):
        self._map = {}
        self._anchors = anchors or {}
        self._null = null
        self._cycler: Iterator[str] = cycle(self._COLOURBLIND_FRIENDLY)

    def __len__(self): return len(self._map)
    def __iter__(self): return iter(self._map.items())
    def __setitem__(self, key, value): self._map[key] = value
    def __getitem__(self, item) -> str:
        """
        Fetches the assigned colour for a gene, or assigns a new one dynamically.

        Args:
            item: The gene name.

        Returns:
            Hex colour string.
        """
        if (base_name := self._gene_to_anchor(item)) is None: return self._null
        # Return from map if already assigned
        if colour := self._map.get(base_name, None): return colour
        # Check if it's an anchor gene
        if (colour := self._anchors.get(base_name, None)) is None: colour = next(self._cycler)
        self.__setitem__(base_name, colour)
        return colour

    @classmethod
    def _gene_to_anchor(cls, gene_name: str) -> Union[str, None]:
        """
        Converts a gene name to its anchor which is usually the "cluster" name: wzyKL1
        """
        if not gene_name: return None
        if len(parts := gene_name.split('_', 2)) < 3: return None
        if m := cls._GENE_REGEX.match(parts[2]): return m.group(0)
        return None


class WzyColourMap(GeneColourMap):
    """
    Colour map specialized for Klebsiella K-locus (Capsule) genes.
    """
    def __init__(self, null: str = '#d3d3d3'):
        super().__init__(
            anchors={
                'galF': '#e6ab02', # Dark Yellow
                'gnd': '#66a61e',  # Green
                'wzi': '#ff7f00',  # Orange
                'wza': '#1b9e77',  # Teal
                'wzb': '#a6cee3',  # Light Blue
                'wzc': '#fb9a99',  # Salmon
                'wzx': '#cab2d6',  # Light Purple (Flippase)
                'wzy': '#6a3d9a',  # Dark Purple (Polymerase)
                'und': '#C92D20',
                'gtr': '#54B741',
                'itr': '#CBD6F8',
                'atr': '#9973D2',
            },
            null=null
        )

class WzmWztColourMap(GeneColourMap):
    """
    Colour map specialized for O-locus (LPS) genes.
    """
    def __init__(self, null: str = '#d3d3d3'):
        super().__init__(
            anchors={
                'wzm': '#e6ab02', # Dark Yellow
                'wzt': '#66a61e',  # Green
            },
            null=null
        )


class Plotter(ABC):
    """Abstract base class for all plotting components."""
    @abstractmethod
    def plot(self):
        """Render the component using Matplotlib."""
        ...
    @abstractmethod
    def plotly(self):
        """Render the component using Plotly."""
        ...


class GenePlotter(Plotter):
    """
    Handles the visualization of a single gene arrow.

    Attributes:
        name: Gene name.
        start: Start coordinate.
        end: End coordinate.
        strand: Genomic strand.
        colour: Fill colour.
        status: Structural status (Normal, Truncated, Partial).
        presence: Expected or Unexpected.
    """
    __slots__ = ('name', 'start', 'end', 'strand', 'colour', 'status', 'presence', 'edge_style', 'line_weight')
    def __init__(self, name: str, start: int, end: int, strand: Strand, colour: str,
                 status: GeneStatus = GeneStatus.NORMAL, presence: GenePresence = GenePresence.EXPECTED):
        """
        Initializes the GenePlotter.

        Args:
            name: Name of the gene.
            start: Start position.
            end: End position.
            strand: Strand enum or castable value.
            colour: Hex colour string.
            status: GeneStatus enum.
            presence: GenePresence enum.
        """
        self.name: str = name if isinstance(name, str) else str(name)
        self.start: int = start if isinstance(start, int) else int(start)
        self.end : int = end if isinstance(end, int) else int(end)
        self.strand: Strand = Strand(strand)
        self.colour: str = colour
        self.status: GeneStatus = GeneStatus(status)
        self.presence: GenePresence = GenePresence(presence)
        self.edge_style: str = ':' if self.status == GeneStatus.TRUNCATED else '-'
        self.line_weight: float = 1.5 if self.status == GeneStatus.TRUNCATED else 1

    def _get_vertices(self, global_start, global_end, y_offset, height):
        """Helper to extract 5-point geometry so it can be shared by Matplotlib and Plotly"""
        width = global_end - global_start

        # Draw a flat rectangle for contig boundaries or unstranded genes
        if self.status == GeneStatus.PARTIAL or self.strand == Strand.UNSTRANDED:
            return [
                (global_start, y_offset - height / 2),
                (global_end, y_offset - height / 2),
                (global_end, y_offset + height / 2),
                (global_start, y_offset + height / 2)
            ]

        # Dynamic head length: 30% of gene width, capped at 400bp max
        hl = min(width * 0.3, 400)

        if self.strand == Strand.FORWARD:
            return [
                (global_start, y_offset - height / 2),               # Bottom-left
                (global_end - hl, y_offset - height / 2),            # Bottom-right of body
                (global_end, y_offset),                              # Arrow tip
                (global_end - hl, y_offset + height / 2),            # Top-right of body
                (global_start, y_offset + height / 2)                # Top-left
            ]
        else: # Strand.REVERSE
            return [
                (global_end, y_offset - height / 2),                 # Bottom-right
                (global_start + hl, y_offset - height / 2),          # Bottom-left of body
                (global_start, y_offset),                            # Arrow tip
                (global_start + hl, y_offset + height / 2),          # Top-left of body
                (global_end, y_offset + height / 2)                  # Top-right
            ]

    def plot(self, ax: plt.Axes, global_x_offset: int = 0, y_offset: int = 0, height=0.4):
        """
        Draws the gene arrow on a Matplotlib Axes.

        Args:
            ax: The target Matplotlib Axes.
            global_x_offset: X-axis offset for the track.
            y_offset: Y-axis position for the track.
            height: Height of the gene arrow.
        """
        global_start = self.start + global_x_offset
        global_end = self.end + global_x_offset
        poly = patches.Polygon(
            self._get_vertices(global_start, global_end, y_offset, height),
            facecolor=self.colour,
            edgecolor='black',
            linestyle=self.edge_style,
            linewidth=self.line_weight,
            zorder=3
        )
        ax.add_patch(poly)
        ax.text(
            x=global_start + ((self.end - self.start) / 2),  # FIXED: added parentheses
            y=y_offset + (height / 2) + 0.1,
            s=self.name,
            ha='left',
            va='bottom',
            rotation=45,
            rotation_mode='anchor',
            fontsize=9,
            fontstyle='italic',
            zorder=4
        )

    def plotly(self, fig: go.Figure, global_x_offset: int = 0, y_offset: int = 0, height=0.4):
        """
        Adds the gene arrow trace to a Plotly Figure.

        Args:
            fig: The target Plotly Figure.
            global_x_offset: X-axis offset for the track.
            y_offset: Y-axis position for the track.
            height: Height of the gene arrow.
        """
        global_start = self.start + global_x_offset
        global_end = self.end + global_x_offset
        # 1. Get the shape vertices
        verts = self._get_vertices(global_start, global_end, y_offset, height)
        # 2. Close the loop for Plotly
        verts.append(verts[0])
        # 3. Split into X and Y lists
        x_coords = [v[0] for v in verts]
        y_coords = [v[1] for v in verts]
        # 4. Handle Styling
        dash_style = 'dot' if self.status == GeneStatus.TRUNCATED else 'solid'
        line_width = 1.5 if self.status == GeneStatus.TRUNCATED else 1

        # 5. Add the Polygon Trace
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            fill='toself',
            fillcolor=self.colour,
            mode='lines',
            line=dict(color='black', width=line_width, dash=dash_style),
            name=self.name,
            text=f"<b>{self.name}</b><br>Start: {self.start}<br>End: {self.end}<br>Strand: {self.strand}",
            hoverinfo='text',
            showlegend=False
        ))


class LocusPlotter(Plotter):
    """
    Visualizes a genomic track (backbone line + genes).

    Used for both the reference locus and the assembly pieces.
    """
    __slots__ = ('name', 'length', 'features', 'global_offset', 'line_colour', 'text_colour')
    def __init__(self, name: str, length: int, features: Iterable[GenePlotter] = None, global_offset: int = 0):
        """
        Initializes the LocusPlotter.

        Args:
            name: Name of the locus/contig.
            length: Total length in bp.
            features: List of GenePlotter objects to draw on this track.
            global_offset: X-position where this track starts.
        """
        self.name: str = name if isinstance(name, str) else str(name)
        self.length: int = length if isinstance(length, int) else int(length)
        self.features: list[GenePlotter] = list(features) if features else []
        self.global_offset: int = global_offset if isinstance(global_offset, int) else int(global_offset)
        self.line_colour: str = 'gray'
        self.text_colour: str = 'dimgray'

    def plot(self, ax: plt.Axes, y_offset: int):
        """
        Draws the track backbone and all features on Matplotlib.

        Args:
            ax: Target Axes.
            y_offset: Y-position for the track.
        """
        # Draw the backbone line
        ax.plot([self.global_offset, self.global_offset + self.length],
                [y_offset, y_offset], color=self.line_colour, linewidth=2, zorder=2)

        # Label the track
        ax.text(self.global_offset + self.length / 2, y_offset - 0.3,
                self.name, ha='center', va='top', fontsize=9, color=self.text_colour)

        for feat in self.features:
            feat.plot(ax, self.global_offset, y_offset)

    def plotly(self, fig: go.Figure, y_offset: int):
        """
        Draws the track backbone and all features on Plotly.

        Args:
            fig: Target Figure.
            y_offset: Y-position for the track.
        """
        # Draw the backbone line
        fig.add_trace(go.Scatter(
            x=[self.global_offset, self.global_offset + self.length],
            y=[y_offset, y_offset],
            mode='lines',
            line=dict(color=self.line_colour, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Label the track
        fig.add_annotation(
            x=self.global_offset + self.length / 2,
            y=y_offset - 0.3,
            text=self.name,
            showarrow=False,
            yanchor='top',
            font=dict(size=12, color=self.text_colour)
        )

        labels_x, labels_y, labels_text = [], [], []
        for feat in self.features:
            feat.plotly(fig, self.global_offset, y_offset)
            # Collect label info for batch rendering
            labels_x.append(self.global_offset + feat.start + (feat.end - feat.start) / 2)
            labels_y.append(y_offset + 0.3)  # height(0.4)/2 + 0.1
            labels_text.append(f"<i>{feat.name}</i>")

        if labels_x:
            for x, y, txt in zip(labels_x, labels_y, labels_text):
                fig.add_annotation(
                    x=x, y=y,
                    text=txt,
                    showarrow=False,
                    textangle=-45,
                    font=dict(size=11, color="black"),
                    xanchor="left",
                    yanchor="bottom"
                )


class AlignmentPlotter(Plotter):
    """
    Visualizes the shaded region connecting homologous genes between tracks.
    """
    __slots__ = ('track1', 'start1', 'end1', 'track2', 'start2', 'end2', 'percent_id', 'colour', 'alpha')

    def __init__(self, track1: LocusPlotter, start1: int, end1: int, track2: LocusPlotter, start2: int, end2: int,
                 percent_id: float = 100.0, colour: str = 'lightgrey'):
        """
        Initializes the AlignmentPlotter.

        Args:
            track1: The top track (usually reference).
            start1: Start coord on track1.
            end1: End coord on track1.
            track2: The bottom track (usually assembly piece).
            start2: Start coord on track2.
            end2: End coord on track2.
            percent_id: Percent identity of the match (affects opacity).
            colour: Fill colour.
        """
        self.track1: LocusPlotter = track1
        self.start1: int = start1
        self.end1: int = end1
        self.track2: LocusPlotter = track2
        self.start2: int = start2
        self.end2: int = end2
        self.percent_id: float = float(percent_id)
        self.colour: str = colour
        self.alpha: float = self._calculate_alpha()

    def _calculate_alpha(self, min_id=70.0, min_alpha=0.1, max_alpha=0.6):
        pid = max(self.percent_id, min_id)
        pid = min(pid, 100.0)
        scale = (pid - min_id) / (100.0 - min_id)
        return min_alpha + (scale * (max_alpha - min_alpha))

    def _get_vertices(self, y1: int, y2: int, gene_height: float):
        top_y = y1 - (gene_height / 2)
        x1_top = self.start1 + self.track1.global_offset
        x2_top = self.end1 + self.track1.global_offset

        bot_y = y2 + (gene_height / 2)
        x1_bot = self.start2 + self.track2.global_offset
        x2_bot = self.end2 + self.track2.global_offset

        return [
            (x1_top, top_y),
            (x2_top, top_y),
            (x2_bot, bot_y),
            (x1_bot, bot_y)
        ]

    def plot(self, ax: plt.Axes, y1: int, y2: int, gene_height: float = 0.4):
        """
        Draws the alignment polygon on Matplotlib.

        Args:
            ax: Target Axes.
            y1: Y-position of the top track.
            y2: Y-position of the bottom track.
            gene_height: Height of genes (used to attach polygon to gene edges).
        """
        ax.add_patch(patches.Polygon(self._get_vertices(y1, y2, gene_height), facecolor=self.colour, alpha=self.alpha,
                                     edgecolor='none', zorder=1))

    def plotly(self, fig: go.Figure, y1: int, y2: int, gene_height: float = 0.4):
        """
        Draws the alignment polygon on Plotly.

        Args:
            fig: Target Figure.
            y1: Y-position of the top track.
            y2: Y-position of the bottom track.
            gene_height: Height of genes.
        """
        verts = self._get_vertices(y1, y2, gene_height)
        verts.append(verts[0])
        x_coords = [v[0] for v in verts]
        y_coords = [v[1] for v in verts]

        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            fill='toself',
            fillcolor=self.colour,
            opacity=self.alpha,
            mode='none',
            showlegend=False,
            hoverinfo='skip'
        ))


class ResultPlotter(Plotter):
    """
    Orchestrates the plotting of a full Kaptive result (Reference + Assembly Pieces + Alignments).
    """
    __slots__ = ('sample_name', 'reference', 'pieces', 'title', 'padding', 'alignments')
    def __init__(self, sample_name: str, pieces: Iterable[LocusPlotter], alignments: Iterable[AlignmentPlotter] = None,
                 reference: LocusPlotter = None, title: str = "Kaptive Result", padding: int = 1500):
        """
        Initializes the ResultPlotter.

        Args:
            sample_name: Name of the sample.
            pieces: List of LocusPlotters representing assembly contigs.
            alignments: List of AlignmentPlotters connecting reference and pieces.
            reference: LocusPlotter for the reference gene cluster.
            title: Plot title.
            padding: Visual padding between assembly pieces in bp.
        """
        self.sample_name: str = sample_name
        self.pieces: list[LocusPlotter] = list(pieces) if pieces else []
        self.alignments: list[AlignmentPlotter] = list(alignments) if alignments else []
        self.reference: Optional[LocusPlotter] = reference
        self.title: str = title
        self.padding: int = padding

    def plot(self, ax: plt.Axes):
        """
        Renders the complete figure on a Matplotlib Axes.

        Args:
            ax: Target Axes.
        """
        ref_y = 2
        piece_y = 0

        # 1. Render Reference
        if self.reference:
            self.reference.plot(ax, y_offset=ref_y)

        # 2. Render Split pieces
        current_x = 0
        custom_ticks = []
        custom_labels = []

        for piece in self.pieces:
            piece.global_offset = current_x
            piece.plot(ax, y_offset=piece_y)

            # Setup ticks mapping to local coordinates
            custom_ticks.extend([current_x, current_x + piece.length])
            custom_labels.extend(['0', f'{piece.length}'])

            current_x += piece.length + self.padding

        # 3. Render Alignments (BLAST hits)
        for aln in self.alignments:
            aln.plot(ax, y1=ref_y, y2=piece_y)

        # 4. Axes Management
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels(custom_labels, fontsize=8)
        ax.set_yticks([piece_y, ref_y])
        ax.set_yticklabels([self.sample_name, self.reference.name], fontsize=10, fontweight='bold')
        ax.set_title(self.title, pad=20)

        for spine in ['left', 'right', 'top']:
            ax.spines[spine].set_visible(False)
        ax.set_xlabel("Local piece Position (bp)")

        ax.autoscale_view()
        ax.set_ylim(-1, 3)  # Give breathing room vertically


    def plotly(self, fig: go.Figure):
        """
        Renders the complete figure on a Plotly Figure.

        Args:
            fig: Target Figure.
        """
        ref_y = 2
        piece_y = 0

        # 1. Render Reference
        if self.reference:
            self.reference.plotly(fig, y_offset=ref_y)

        # 2. Render Split pieces
        current_x = 0
        tick_vals = []
        tick_text = []

        for piece in self.pieces:
            piece.global_offset = current_x
            piece.plotly(fig, y_offset=piece_y)

            tick_vals.extend([current_x, current_x + piece.length])
            tick_text.extend(['0', f'{piece.length}'])

            current_x += piece.length + self.padding

        # 3. Render Alignments
        for aln in self.alignments:
            aln.plotly(fig, y1=ref_y, y2=piece_y)

        # 4. Layout
        fig.update_layout(
            title=self.title,
            xaxis=dict(
                title="Local piece Position (bp)",
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=[piece_y, ref_y],
                ticktext=['Assembly', 'Reference'],
                showgrid=False,
                zeroline=False,
                range=[piece_y - 1, ref_y + 1]
            ),
            plot_bgcolor='white',
            showlegend=False
        )


class KaptivePlotter(Plotter):
    """
    Main entry point for generating Kaptive visualizations.

    Converts Kaptive `TypingResult` objects into visual representations using
    either Matplotlib or Plotly.

    Examples:
        >>> from kaptive.database import load_database
        >>> from kaptive.assembly import typing_pipeline
        >>> from kaptive_plot import KaptivePlotter
        >>> db = load_database('kpsc_k')
        >>> result = typing_pipeline('assembly.fasta', db)
        >>> plotter = KaptivePlotter(db)
        >>> # Assuming 'result' is a TypingResult object
        >>> with plotter.plot(result) as (fig, ax):
        ...     fig.savefig("result.png")
    """
    __slots__ = ('_db', '_cm', '_gene_plotters', '_locus_plotters')
    def __init__(self, db: Database, cm: GeneColourMap = None):
        """
        Initializes the KaptivePlotter.

        Args:
            db: The Kaptive database used for typing.
            cm: Optional custom GeneColourMap. If None, one is selected based on DB name.
        """
        self._db: Database = db
        if cm is None:
            cm_class = GeneColourMap
            db_name = db.name.lower()
            if '_k_' in db_name: cm_class = WzyColourMap
            elif '_o_' in db_name or '_oc_' in db_name: cm_class = WzmWztColourMap
            cm = cm_class()
        self._cm = cm
        self._locus_plotters = {}
        self._gene_plotters = {}
        for locus in db.loci.values():
            length = max((int(g.end) for g in locus.genes.values()), default=0)
            gene_plotters = []
            for gene in locus.genes.values():
                gene_plotter = GenePlotter(gene.name, gene.start, gene.end, Strand.from_symbol(gene.strand), cm[gene.name])
                self._gene_plotters[gene.name] = gene_plotter
                gene_plotters.append(gene_plotter)

            locus_plotter = LocusPlotter(locus.name, length, gene_plotters)
            self._locus_plotters[locus.name] = locus_plotter

    def _prep_result(self, result: TypingResult) -> ResultPlotter:
        if (ref_plotter := self._locus_plotters.get(result.best_match.name, None)) is None:
            raise ValueError(f'{result.sample_name} database ({result.db}) does not match {self._db}')

        piece_plotters = []
        alignment_plotters = []

        for piece in result.pieces:
            offset = int(piece.start)
            length = int(piece.end) - offset

            # Detect if the whole piece matches the reference in reverse
            is_rev_piece = getattr(piece, 'strand', '+') == '-'

            gene_plotters = []
            gene_data_cache = []  # Cache to build alignments in the next step

            for gene_result in piece:
                gene_name = gene_result.gene.name
                gene_type = gene_result.gene_type

                # 1. Calculate relative coordinates
                g_start = int(gene_result.start) - offset
                g_end = int(gene_result.end) - offset
                g_strand = Strand.from_symbol(gene_result.strand)

                # 2. Visually "untwist" the piece if it's on the negative strand
                if is_rev_piece:
                    new_start = length - g_end
                    new_end = length - g_start
                    g_start, g_end = new_start, new_end

                    if g_strand == Strand.FORWARD:
                        g_strand = Strand.REVERSE
                    elif g_strand == Strand.REVERSE:
                        g_strand = Strand.FORWARD

                # 3. Create the gene
                gene_plotter = GenePlotter(
                    name=gene_name,
                    start=g_start,
                    end=g_end,
                    strand=g_strand,
                    colour=self._cm[gene_name],
                    presence=GenePresence.UNEXPECTED if 'unexpected' in gene_type else GenePresence.EXPECTED
                )
                gene_plotters.append(gene_plotter)
                gene_data_cache.append((gene_result, g_start, g_end, g_strand))

            # 4. Create the Piece Plotter
            piece_plotter = LocusPlotter(piece.id, length, gene_plotters)
            piece_plotters.append(piece_plotter)

            # 5. Build Alignments using the actual LocusPlotter objects
            for gene_result, g_start, g_end, g_strand in gene_data_cache:
                ref_g = self._gene_plotters.get(gene_result.gene.name)
                if not ref_g:
                    continue

                pid = float(getattr(gene_result, 'percent_identity', 100.0))

                # Handle true inversions: if strands don't match, cross the alignment polygon
                aln_start, aln_end = g_start, g_end
                if ref_g.strand != Strand.UNSTRANDED and g_strand != Strand.UNSTRANDED:
                    if ref_g.strand != g_strand:
                        aln_start, aln_end = g_end, g_start

                alignment_plotter = AlignmentPlotter(
                    track1=ref_plotter,
                    start1=ref_g.start,
                    end1=ref_g.end,
                    track2=piece_plotter,
                    start2=aln_start,
                    end2=aln_end,
                    percent_id=pid,
                    colour=ref_g.colour
                )
                alignment_plotters.append(alignment_plotter)

        return ResultPlotter(result.sample_name, piece_plotters, alignment_plotters, ref_plotter,
                             title=result.confidence)

    @contextmanager
    def plot(self, result: TypingResult, width: int = 14, height: int = 5) -> Iterator[tuple[plt.Figure, plt.Axes]]:
        """
        Context manager for generating a Matplotlib figure.

        Args:
            result: The Kaptive typing result to plot.
            width: The width of the figure.
            height: The height of the figure.

        Yields:
            A tuple (Figure, Axes). The figure is automatically closed after the context exits.
        """
        fig, ax = plt.subplots(figsize=(width, height))
        result_plotter = self._prep_result(result)
        result_plotter.plot(ax)
        try: yield fig, ax
        finally: plt.close(fig)

    def plotly(self, result: TypingResult, width: int = 800, height: int = 400) -> go.Figure:
        """
        Generates a Plotly figure.

        Args:
            result: The Kaptive typing result to plot.
            width: The width of the figure.
            height: The height of the figure.

        Returns:
            A plotly.graph_objects.Figure object.
        """
        fig = go.Figure()
        result_plotter = self._prep_result(result)
        result_plotter.plotly(fig)
        fig.update_layout(width=width, height=height)
        return fig

