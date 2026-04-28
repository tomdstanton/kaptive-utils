from dataclasses import dataclass
from typing import IO
from warnings import warn

import numpy as np
import gb_io
from kaptive.database import _LOCUS_REGEX


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Database:
    loci: list[np.ndarray]
    locus_names: list[str]
    genes: list[bytes]
    proteins: list[bytes]
    gene_ids: list[str]
    gene_names: list[str]

    @classmethod
    def from_stream(cls, handle: IO[bytes]):
        global_gene_counter = 0

        loci, locus_names, genes, proteins, gene_ids, gene_names = [], [], [], [], [], []

        for rec in gb_io.iter(handle):  # type: gb_io.Record

            notes = (i.value for i in rec.features[0].qualifiers if i.key == 'note')
            if locus_name := next((m.group() for i in notes if (m := _LOCUS_REGEX.search(i))), None) is None:
                continue

            locus_names.append(locus_name)
            locus = []
            local_gene_idx = 0

            for feat in rec.features:
                if feat.kind == 'CDS':

                    gene_name = f'{locus_name}_{local_gene_idx + 1:02}'
                    if g := _get_qual(feat, 'gene'):
                        gene_name += f'_{g:}'

                    if (prot := _get_qual(feat, 'translation')):
                        prots.append(f'>{gene_name}\n{prot}\n')

                    else:
                        warn(f'Gene does not have a proper translation {feat!r}')

                    locus.append(global_gene_idx)

                    local_gene_idx += 1
                    global_gene_idx += 1



# Functions ------------------------------------------------------------------------------------------------------------
def _get_qual(feature, qual, default=None):
    return next((i.value for i in feature.qualifiers if i.key == qual), default)