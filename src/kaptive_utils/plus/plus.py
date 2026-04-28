from typing import Generator, Iterable, TextIO, Literal
from io import TextIOBase
from itertools import groupby, chain
from operator import attrgetter
from concurrent.futures import ThreadPoolExecutor

from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

from pyhmmer import hmmsearch, hmmscan
from pyhmmer.plan7 import OptimizedProfileBlock, HMMFile
from pyhmmer.easel import TextSequence, Alphabet, TextSequenceBlock, SequenceFile

from kaptive.database import load_database, Database
from kaptive.assembly import typing_pipeline, parse_assembly, Assembly, parse_result, _ASSEMBLY_FASTA_REGEX
from kaptive.typing import GeneResult, TypingResult


# Constants ------------------------------------------------------------------------------------------------------------
_ALPHABET = Alphabet.amino()
_HEADER = ('Assembly\tLocus\tPhenotype\tConfidence\tContig\tStart\tEnd\tStrand\tGene\tBest_HMM\tBest_HMM_Score\t'
           'Locus_gene\tLocus_gene_type\tGene_cluster\tProblems\n')


# Classes --------------------------------------------------------------------------------------------------------------
class KaptivePlusResult:
    def __init__(self, assembly: Assembly, typing_result: TypingResult = None, *args, **kwargs):
        """Container for a KaptivePlus run on an assembly so all information can be returned from a pipeline function"""
        self.assembly = assembly
        self.typing_result = typing_result or typing_pipeline(assembly, *args, **kwargs)
        self.annotated = []
        self.gene_clusters = []

    def __repr__(self):
        return self.typing_result.__repr__() if self.typing_result else self.assembly.__repr__()

    def __len__(self):
        return len(self.annotated)

    def find_genes(self, gene_finder: GeneFinder, verbose: bool = False):
        if not gene_finder.training_info:
            log(f"Training gene finder on {self}", verbose=verbose)
            gene_finder.train(*(bytes(i.seq) for i in self.assembly.contigs.values()))
        log(f"Predicting ORFs in {self}", verbose=verbose)
        with ThreadPoolExecutor(min(cpu_count() + 4, len(self.assembly.contigs), 32)) as pool:
            # Note: this does make gene finding very fast
            pool.map(lambda i: i.find_genes(gene_finder), self.assembly.contigs.values())
        return

    def annotate(self, profiles, verbose: bool = False, **options):
        genes, proteins = {}, TextSequenceBlock()  # Keep all genes in dict for easy retrieval, we won't need this again
        for contig in self.assembly.contigs.values():  # Get genes from the contig features
            for gene in contig.features:
                proteins.append(TextSequence((i := gene.id.encode()), sequence=gene.qualifiers['translation'][0]))
                genes[i] = gene  # This is for easily retrieving the gene later
        if isinstance(profiles, OptimizedProfileBlock):  # Run hmmscan if HMMs have been pressed
            log(f"Annotating {len(proteins)} proteins with hmmscan", verbose=verbose)
            for hits in hmmscan(proteins.digitize(_ALPHABET), profiles, **options):
                if hits:
                    self.annotated.append(gene := genes[hits.query.name])
                    # They may already be sorted, do this just to be sure
                    gene.best_hmm = (best_hit := max(hits, key=attrgetter('score'))).name.decode()
                    gene.best_hmm_score = best_hit.score
        else:
            log(f"Annotating {len(proteins)} proteins with hmmsearch", verbose=verbose)
            for gene, hits in groupby(sorted(chain.from_iterable(  # Run hmmsearch with all profiles, group by query
                    i for i in hmmsearch(profiles, proteins.digitize(_ALPHABET), **options) if i),
                    key=attrgetter('name')), key=attrgetter('name')):  # Group results from all HMMs and sort by query
                self.annotated.append(gene := genes[gene])  # Retrieve the gene and add to annotated
                # Get the best hmm per query with max score and setting the name as 'product'
                gene.best_hmm = (best_hit := max(hits, key=attrgetter('score'))).hits.query.name.decode()
                gene.best_hmm_score = best_hit.score

    def identify_locus_genes(self, overlap_frac: float = 0.8):
        grs = {  # Gene Results, grouped by contig
            k: list(v) for k, v in groupby(sorted(self.typing_result, key=attrgetter('id')), key=attrgetter('id'))
        }  # Sort annotated genes by contig
        for contig, genes in groupby(sorted(self.annotated, key=attrgetter('contig')), key=attrgetter('contig')):
            if grs_on_contig := grs.get(contig):  # Get ORFs that are in the locus that Kaptive found
                for i in genes:
                    for gr in grs_on_contig:  # type: GeneResult
                        if range_overlap((i.location.start, i.location.end), (gr.start, gr.end),
                                         skip_sort=True) / len(gr) >= overlap_frac:
                            i.gene_result = gr
                            break  # No other genes should overlap to break here

    def get_gene_clusters(self, min_n_genes: int, skip_n_unannotated: int, verbose: bool = False):
        """This could probably be a regular function but convenient to make it a class function"""
        log(f'Finding gene clusters in {self}; {min_n_genes=}; {skip_n_unannotated=}', verbose=verbose)
        for contig in self.assembly.contigs.values():
            x = [i for i, c in enumerate(contig.features)
                 if c.best_hmm and (not c.gene_result or c.gene_result.gene_type == 'extra_genes')]
            for cluster in filter(lambda i: len(i) >= min_n_genes, grouper(x, distance=skip_n_unannotated)):
                self.gene_clusters.append(gene_cluster := contig[contig.features[cluster[0]].location.start:
                                                                 contig.features[cluster[-1]].location.end + 1])
                gene_cluster.annotations["molecule_type"] = "DNA"
                gene_cluster.id = f"{contig.id}_{len(self.gene_clusters):06d}"
                for gene_index in cluster:
                    contig.features[gene_index].gene_cluster = gene_cluster

    def format(self, format_spec) -> str:
        if format_spec in {'tsv', 'ffn', 'faa'}:
            tsv_prefix = (f"{self.assembly}\t"
                          f"{self.typing_result.best_match}\t"
                          f"{self.typing_result.phenotype}\t"
                          f"{self.typing_result.confidence}\t") if format_spec == 'tsv' else ''
            return ''.join(i.format(format_spec, tsv_prefix) for i in self.annotated)
        elif format_spec in {'fasta', 'genbank'}:
            return ''.join(i.format(format_spec) for i in self.gene_clusters)

    def write(self,
              tsv: TextIO | None = None,
              faa: str | PathLike | TextIO | None = None,
              ffn: str | PathLike | TextIO | None = None,
              genbank: str | PathLike | TextIO | None = None,
              fasta: str | PathLike | TextIO | None = None):
        """Write the typing result to files or file handles."""
        for fmt, arg in locals().items():
            if fmt in {'tsv', 'ffn' 'faa', 'genbank', 'fasta'}:
                if isinstance(arg, TextIOBase):
                    arg.write(self.format(fmt))
                elif isinstance(arg, (PathLike, str)):
                    with open(path.join(arg, f'{self.assembly}_kaptiveplus.{fmt}'), 'wt') as handle:
                        handle.write(self.format(fmt))


class Contig(SeqRecord):
    def __init__(self, *args, **kwargs):
        super(Contig, self).__init__(*args, **kwargs)

    def find_genes(self, gene_finder: GeneFinder):
        previous_orf = None  # ORF to the left
        for orf in gene_finder.find_genes(bytes(self.seq)):
            self.features.append(i := CDS.from_pyrodigal(self.id, orf, id=f'{self.id}_{len(self.features) + 1:06d}'))
            if previous_orf:
                i.neighbours[0].append(previous_orf)
                previous_orf.neighbours[1].append(i)
            previous_orf = i


class CDS(SeqFeature):
    def __init__(self, contig: str = None, sequence: str = None, neighbours: list[list[CDS], list[CDS]] = None,
                 gene_result: GeneResult = None, problems: list[str] = None, best_hmm: str = None,
                 best_hmm_score: float = None, gene_cluster: SeqRecord = None, *args, **kwargs):
        super(CDS, self).__init__(*args, **kwargs)
        self.contig = contig or ''
        self.sequence = sequence or ''
        self.neighbours = neighbours or [[], []]
        self.gene_result = gene_result
        self.problems = problems or []
        self.best_hmm = best_hmm or ''
        self.best_hmm_score = best_hmm_score
        self.gene_cluster = gene_cluster
        self.type = 'CDS'

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def from_pyrodigal(cls, contig: str, gene: Gene, *args, **kwargs) -> CDS:
        self = cls(
            contig=contig, location=SimpleLocation(gene.begin - 1, gene.end, gene.strand),
            qualifiers={'transl_table': [gene.translation_table], 'translation': [gene.translate()],
                        'inference': [f'ab initio prediction:Pyrodigal:{pyrodigal_version}']},
            problems=[k for k, v in (('partial_begin', gene.partial_begin), ('partial_end', gene.partial_end)) if v],
            sequence=gene.sequence(), *args, **kwargs
        )
        self.qualifiers['locus_tag'] = self.id
        return self

    def format(self, format_spec, tsv_prefix: str = '') -> str:
        if format_spec == 'tsv':
            return (f'{tsv_prefix}'
                    f'{self.contig}\t'
                    f'{self.location.start}\t'
                    f'{self.location.end}\t'
                    f'{self.location.strand}\t'
                    f'{self.id}\t'
                    f'{self.best_hmm}\t'
                    f'{self.best_hmm_score}\t'
                    f'{self.gene_result if self.gene_result else ""}\t'
                    f'{self.gene_result.gene_type if self.gene_result else ""}\t'
                    f'{self.gene_cluster.id if self.gene_cluster else ""}\t'
                    f'{",".join(self.problems)}'
                    f'\n')
        elif format_spec == 'faa':
            return f">{self.id}\n{self.qualifiers['translation'][0]}\n"
        elif format_spec == 'ffn':
            return f">{self.id}\n{self.sequence}\n"


# Functions ------------------------------------------------------------------------------------------------------------
def parse_args(a: list[str]) -> Namespace:
    parser = ArgumentParser(
        usage="%(prog)s <db> <hmm> <assembly> [<assembly> ...] [options]",
        description=get_logo('A Kaptive add-on for annotating genes\nin the context of locus typing results'),
        add_help=False, prog='kaptiveplus', epilog=f"%(prog)s v{__version__}", formatter_class=RawTextHelpFormatter
    )
    input_parser = parser.add_argument_group(bold('Inputs'), "")
    input_parser.add_argument('db', metavar='db path/keyword', help='Kaptive database path or keyword')
    input_parser.add_argument('hmm',
                              help='HMMER-formatted profile HMM file for hmmsearch\n'
                                   'Note if pressed, hmmscan will be performed instead',
                              type=lambda i: check_file(i, panic=True))
    input_parser.add_argument('assembly', nargs='+', help='Assemblies in fasta(.gz|.xz|.bz2) format',
                              type=lambda i: check_file(i, panic=True))
    input_parser.add_argument('--kaptive-results', metavar='', type=lambda i: check_file(i, panic=True),
                              help='Optional pre-computed Kaptive results in JSON format\n'
                                   'Note, this speeds up the pipeline')
    output_parser = parser.add_argument_group(bold('Output Options'), "")
    output_parser.add_argument('--tsv', type=FileType('at'), default='-', metavar='file',
                               help='Output file to write/append per-gene tabular results to (default: %(default)s)')
    output_parser.add_argument('--faa', nargs='?', default=None, const='.', type=check_out,
                               metavar='dir/file', help='Turn on gene protein fasta output\n'
                                                        'Accepts a single file or a directory (default: %(const)s)')
    output_parser.add_argument('--ffn', nargs='?', default=None, const='.', type=check_out,
                               metavar='dir/file', help='Turn on gene nucleotide fasta output\n'
                                                        'Accepts a single file or a directory (default: %(const)s)')
    output_parser.add_argument('--genbank', nargs='?', default=None, const='.', type=check_out,
                               metavar='dir/file', help='Turn on Gene Cluster Genbank output\n'
                                                        'Accepts a single file or a directory (default: %(const)s)')
    output_parser.add_argument('--fasta', nargs='?', default=None, const='.', type=check_out,
                               metavar='dir/file', help='Turn on Gene Cluster nucleotide fasta output\n'
                                                        'Accepts a single file or a directory (default: %(const)s)')
    orf_parser = parser.add_argument_group(bold('ORF Options'), "\nOptions for tuning Pyrodigal")
    orf_parser.add_argument('--training-info', metavar='',
                            help="Pyrodigal training info (default: %(default)s)")
    orf_parser.add_argument('--good-assembly', metavar='',
                            help="Assembly to use for training the GeneFinder (default: %(default)s)")
    orf_parser.add_argument('--min-gene', type=int, default=90, metavar='',
                            help="The minimum gene length (default: %(default)s)")
    orf_parser.add_argument('--min-edge-gene', type=int, default=60, metavar='',
                            help="The minimum edge gene length (default: %(default)s)")
    orf_parser.add_argument('--max-overlap', type=int, default=60, metavar='',
                            help="The maximum number of nucleotides that can overlap between two genes on the same\n"
                                 "strand (default: %(default)s)")
    orf_parser.add_argument('--min-mask', type=int, default=50, metavar='',
                            help="The minimum mask length, when region masking is enabled. Regions shorter than the\n"
                                 "given length will not be masked, which may be helpful to prevent masking of single\n"
                                 "unknown nucleotides (default: %(default)s)")
    orf_parser.add_argument('--meta', action='store_true',
                            help="Run in metagenomic mode, using a pre-trained profiles for better results with\n"
                                 "metagenomic or progenomic inputs (default: %(default)s)")
    orf_parser.add_argument('--closed', action='store_true',
                            help="Consider sequences ends closed, which prevents proteins from running off edges\n"
                                 "(default: %(default)s)")
    orf_parser.add_argument('--mask', action='store_true',
                            help="Prevent genes from running across regions containing unknown nucleotides\n"
                                 "(default: %(default)s)")
    hmm_parser = parser.add_argument_group(bold('HMM Options'), "\nOptions for tuning PyHMMER")
    hmm_parser.add_argument('--E', type=float, default=1e-20, metavar='',
                            help="The per-target E-value threshold for reporting a hit (default: %(default)s)")
    hmm_parser.add_argument('--bit-cutoffs', metavar='', choices={'noise', 'gathering', 'trusted'},
                            help="The model-specific thresholding option to use for reporting hits\n"
                                 "(default: %(default)s)")
    gene_cluster_parser = parser.add_argument_group(bold('Gene Cluster Options'), "")
    gene_cluster_parser.add_argument('--min-n-genes', type=int, metavar='',
                                     help="Minimum number of genes in each cluster (default: number of HMM files)")
    gene_cluster_parser.add_argument('--skip-n-unannotated', type=int, metavar='', default=2,
                                     help="Skip N unannotated genes when grouping neighbouring annotated genes\n"
                                          "together in a cluster (default: %(default)s)")
    other_parser = parser.add_argument_group(bold('Other Options'), "")
    other_parser.add_argument('--no-header', action='store_true', help='Suppress header line')
    other_parser.add_argument('-t', '--threads', type=check_cpus, default=check_cpus(), metavar='int',
                              help="Number of alignment threads or 0 for all available (default: 0)")
    other_parser.add_argument('-V', '--verbose', action='store_true',
                              help='Print debug messages to stderr')
    other_parser.add_argument('-v', '--version', help='Show version number and exit', action='version',
                              version=f'%(prog)s {__version__}')
    other_parser.add_argument('-h', '--help', action='help', help='show this help message and exit')
    return parser.parse_args(a)


def grouper(pos_list: Iterable[int], distance: int, skip_sort: bool = False) -> Generator[int, None, None]:
    """This function groups numbers in a list that are close to each other
    From stack exchange: https://stackoverflow.com/a/15801233"""
    prev, group = None, []
    for current in (pos_list if skip_sort else sorted(pos_list)):
        if prev is None or current - prev <= distance:
            group.append(current)
        else:
            yield group
            group = [current]
        prev = current
    if group:
        yield group


def get_logo(message: str, width: int = 43) -> str:  # 43 is the width of the logo
    return bold_cyan(f'{_LOGO}\n{message.center(width)}')


# def symbol_type(s: str) -> Literal['gene', 'protein', 'unknown']:
#     """Evaluates a symbol using standard bacterial gene nomenclature"""
#     if len(s) == 3 and s[1:].islower():
#         return 'protein' if s[0].isupper() else 'gene'
#     if len(s) == 4 and s[1:len(s) - 1].islower() and s[-1].isupper():
#         return 'protein' if s[0].isupper() else 'gene'
#     return 'unknown'


# def swap_symbol(s: str) -> str:
#     """Swaps a gene symbol to protein and vice versa if the symbol follows standard bacterial gene nomenclature"""
#     if (i := symbol_type(s)) == 'gene':
#         return f'{s[0].upper()}{s[1:]}'
#     elif i == 'protein':
#         return f'{s[0].lower()}{s[1:]}'
#     return s


def assembly_name(file: str | PathLike) -> str:
    if match := _ASSEMBLY_FASTA_REGEX.search(basename := path.basename(file)):
        return basename.rstrip(match.group())
    return ''


def get_gene_finder(training_info: str | PathLike | None, good_assembly: str | PathLike | None, verbose: bool = False,
                    **options) -> GeneFinder:
    if training_info:
        try:
            with opener(training_info, verbose=verbose, mode='rb') as f:
                training_info = TrainingInfo.load(f)
        except Exception as e:
            quit_with_error(f"Could not load {training_info=}: {e}")
    if (gene_finder := GeneFinder(training_info, **options)).training_info is None:
        if good_assembly:  # Assuming GeneFinder will be trained in the first instance
            try:  # Use pyhmmer to parse the good assembly fasta file
                with opener(good_assembly, verbose=verbose, mode='rb') as f:
                    log(f"Training gene finder on {good_assembly=}", verbose=verbose)
                    gene_finder.train(*(i.sequence.encode() for i in SequenceFile(f, format='fasta')))
                    return gene_finder
            except Exception as e:
                quit_with_error(f"Could not train gene finder on {good_assembly=}: {e}")
        else:
            return gene_finder


def load_hmms(file: str | PathLike, verbose: bool = False):
    log(f'Loading HMMs from {file}', verbose=verbose)
    try:
        with HMMFile(file) as f:
            return OptimizedProfileBlock(_ALPHABET, f.optimized_profiles()) if f.is_pressed() else list(f)
            # return list(f)
    except Exception as e:
        quit_with_error(f"Could not load HMMs from {file}: {e}")


def write_headers(tsv: TextIO = None, no_header: bool = False) -> int:
    """Write appropriate header to a file handle."""
    if tsv and not no_header and (tsv.name == '<stdout>' or fstat(tsv.fileno()).st_size == 0):
        return tsv.write(_HEADER)


def plus_pipeline(
        assembly: str | PathLike | Assembly, db: str | PathLike | Database, profiles, gene_finder: GeneFinder = None,
        threads: int = 0, E: float = 1e-20, bit_cutoffs: Literal['noise', 'gathering', 'trusted'] = None,
        min_n_genes: int = None, skip_n_unannotated: int = 2, verbose: bool = False, typing_result: TypingResult = None
) -> KaptivePlusResult | None:

    # assembly = parse_assembly(args.assembly[0])
    # db = load_database(args.db)
    # threads = args.threads
    # bit_cutoffs = args.bit_cutoffs
    # min_n_genes = args.min_n_genes
    # skip_n_unannotated = args.skip_n_unannotated
    # verbose = args.verbose

    if not isinstance(db, Database) and not (db := load_database(db, verbose=verbose)):
        return None
    if not isinstance(assembly, Assembly) and not (assembly := parse_assembly(assembly, verbose=verbose)):
        return None
    result = KaptivePlusResult(assembly, db=db, threads=threads, verbose=verbose, typing_result=typing_result)
    if not result.typing_result:
        return warning(f"No Kaptive results for {assembly}; cannot provide context for results")
    # Replace assembly contigs (kaptive.assembly.Contig) with SeqRecord subclass (kaptiveplus.Contig)
    assembly.contigs = {k: Contig(v.seq, k, description=v.desc) for k, v in assembly.contigs.items()}
    result.find_genes(gene_finder, verbose)
    result.annotate(profiles, verbose, E=E, bit_cutoffs=bit_cutoffs)
    if not result.annotated:
        return warning(f"No annotated proteins for {assembly}")
    result.identify_locus_genes()
    result.get_gene_clusters(min_n_genes, skip_n_unannotated, verbose)
    log(f'Finished annotating {assembly}; {len(result.annotated)} annotated genes, '
        f'{len(result.gene_clusters)} extra loci', verbose=verbose)
    return result

