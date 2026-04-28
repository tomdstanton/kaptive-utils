"""
Module to handle query (contigs) and target (features) IO.
"""
from bz2 import open as bzopen
from collections.abc import Iterator
from dataclasses import dataclass
from gzip import open as gzopen
from lzma import open as lzopen
from pathlib import Path
from re import compile as re_compile
from typing import IO

from kaptive_utils.io import FastaReader, GfaReader, SeqRecord
from kaptive_utils.core.graph import Edge


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class GenomeAssembly:
    """
    Container for a genome assembly, including contigs and their graph topology.

    Handles FASTA and GFA formats, with support for transparent decompression.

    Example:
        >>> assembly = GenomeAssembly.from_file("assembly.gfa.gz")
        >>> for contig_id, seq in assembly:
        >>>     print(contig_id, len(seq))
    """
    _SEQUENCE_FILE_REGEX = re_compile(
        r'\.('
        r'(?P<fasta>f(asta|a|na|fn|as|aa))|'
        r'(?P<gfa>gfa)|'
        r')\.?(?P<compression>(gz|bz2|xz))?$'
    )
    _OPENERS = {'gz': gzopen, 'bz2': bzopen, 'xz': lzopen}
    id: str
    contigs: dict[str, bytes]
    edges: list[Edge]
    contig_depths: dict[str, float]
    contig_lengths: dict[str, int]

    def __len__(self):
        """Total number of base pairs in the assembly."""
        return sum(len(i) for i in self.contigs.values())

    def __iter__(self) -> Iterator[tuple[str, bytes]]:
        """Iterate over contig IDs and sequences."""
        return iter(self.contigs.items())

    def __str__(self):
        return self.id

    def __getitem__(self, item: str) -> bytes:
        """Access a contig sequence by its ID."""
        return self.contigs[item]

    @classmethod
    def from_file(cls, file: str | Path):
        """
        Load an assembly from a FASTA or GFA file.

        Args:
            file: Path to the file. Supports .gz, .bz2, and .xz compression.
        """
        file = Path(file) # type: Path
        if not (m := cls._SEQUENCE_FILE_REGEX.search(file.name)):
            raise NotImplementedError(f'Unsupported format: {file}')
        reader = FastaReader if m.group('fasta') else GfaReader
        with cls._OPENERS.get(m.group('compression'), open)(file, mode='rt') as handle:
            return cls.from_stream(handle, reader, file.name.rstrip(m.group()))

    @classmethod
    def from_stream(cls, handle: IO[str], reader, id_: str | None = None):
        """Load an assembly from an open file stream using the specified reader."""
        contigs, edges, depths, lengths = {}, [], {}, {}
        for record in reader(handle):
            if isinstance(record, SeqRecord):
                contigs[record.id] = record.seq
                depths[record.id] = record.annotations.get('DP', record.annotations.get('depth', 1.0))
                lengths[record.id] =  len(record)
            elif isinstance(record, Edge):
                edges.append(record)
        return cls(id_ or handle.name, contigs, edges, depths, lengths)

