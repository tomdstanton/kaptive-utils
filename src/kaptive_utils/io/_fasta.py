from collections.abc import Iterator

from kaptive_utils.io._base import _Handle, SeqRecord


class FastaReader(_Handle):
    """
    A high-performance FASTA file reader.

    Assumes the provided handle is opened in binary mode ('rb') to return the
    sequence as a byte string, which is highly efficient and ideal for wrapped
    FASTA lines.
    """
    def __iter__(self) -> Iterator[SeqRecord]:
        header: str = ""
        seq_chunks: list[bytes] = []

        # Local variable caching speeds up lookups inside the tight loop
        join_bytes = b"".join

        for line in self._handle:
            # rstrip() is faster than strip() and removes \r\n or \n
            line = line.rstrip()
            if not line:
                continue

            # 62 is the ASCII integer value for '>'
            if line[0] == 62:
                if header:
                    # Yield the previous record
                    yield SeqRecord(seq=join_bytes(seq_chunks), id=header, annotations={})

                seq_chunks.clear()
                # Decode the header (skip the '>' character)
                header = line[1:].decode("utf-8", errors="replace")
            else:
                seq_chunks.append(line)

        # Yield the final record once the loop ends
        if header:
            yield SeqRecord(seq=join_bytes(seq_chunks), id=header, annotations={})