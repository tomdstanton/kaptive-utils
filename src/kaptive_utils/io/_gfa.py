from kaptive_utils.io._base import _Handle, SeqRecord
from kaptive_utils.lib.graph import Edge


class GfaReader(_Handle):
    """
    Reader for Graphical Fragment Assembly (GFA) files.

    Parses Segment (S) and Link (L) lines into SeqRecord and Edge objects.
    """

    @staticmethod
    def _parse_segment(parts: list[bytes]):
        name = parts[0].decode()
        seq = parts[1]  # Stays as binary bytes
        tags = {}
        for item in parts[2:]:
            tag, typ, val = item.split(b':', maxsplit=2)
            typ_str = typ.decode()
            if typ_str == 'f':
                val_parsed = float(val)
            elif typ_str == 'i':
                val_parsed = int(val)
            else:
                val_parsed = val.decode()
            tags[tag.decode()] = val_parsed
        return SeqRecord(seq=seq, id=name, annotations=tags)

    @staticmethod
    def _parse_link(parts: list[str]):
        u = parts[0]
        u_strand = Strand(parts[1])
        v = parts[2]
        v_strand = Strand(parts[3])
        cigar = Cigar(parts[4])
        overlap = next((n for op, n, _, _, _ in cigar if op == 'M'), 0)
        return Edge(u, u_strand, v, v_strand, overlap)

    @classmethod
    def _parse_line(cls, line: bytes):
        if line.startswith(b'S\t'):
            return cls._parse_segment(line[2:].rstrip().split(b'\t'))
        elif line.startswith(b'L\t'):
            # Decode Link lines to text for easier string parsing downstream
            return cls._parse_link(line[2:].rstrip().decode().split('\t'))
        else:
            return None

    def __next__(self):
        while True:  # Will naturally raise StopIteration when the handle is exhausted
            if (parsed := self._parse_line(next(self._handle))) is not None:
                return parsed

    def __iter__(self):
        for line in self._handle:
            if (parsed := self._parse_line(line)) is not None:
                yield parsed


# Writers --------------------------------------------------------------------------------------------------------------
class GfaWriter(_Handle):
    """Writer for GFA format files."""

    def write(self, item: Edge | SeqRecord) -> int:
        """Writes an Edge (Link) or SeqRecord (Segment) to the file."""
        if isinstance(item, Edge):
            line = f"L\t{item.u}\t{item.u_strand}\t{item.v}\t{item.v_strand}\t*\n"
            return self._handle.write(line.encode())
        elif isinstance(item, SeqRecord):
            return self._handle.write(b"S\t" + item.id.encode() + b"\t" + item.seq + b"\n")
        raise TypeError(item)
