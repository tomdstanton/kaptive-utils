from typing import IO, NamedTuple


# Data containers ------------------------------------------------------------------------------------------------------
class _Handle:
    def __init__(self, handle: IO):
        self._handle = handle

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._handle.close()


# Readers --------------------------------------------------------------------------------------------------------------
class SeqRecord(NamedTuple):
    seq: bytes
    id: str
    annotations: dict
