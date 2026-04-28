from sys import stdin, argv
import argparse
from importlib.metadata import version
from pathlib import Path

from kaptive.database import load_database
from kaptive.assembly import parse_result

from ._core import KaptivePlotter


# Constants -----------------------------------------------------------------------------------------------------------
_DIST = 'kaptive-plot'
_URL = f'https://tomdstanton.github.io/{_DIST}'
__version__ = version(_DIST)


# Functions -----------------------------------------------------------------------------------------------------------
def _get_parser():
    parser = argparse.ArgumentParser(
        description='Plotting library for Kaptive, the tool for in silico serotyping',
        usage="%(prog)s <db> <json>",
        prog=_DIST, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=f'For more help, visit: {_URL}',
        add_help=False
    )
    input_group = parser.add_argument_group('Inputs', '')
    input_group.add_argument('db', metavar='db', help='Database path or keyword used to generate results')
    input_group.add_argument('json', metavar='json', help='Kaptive results JSON file or "-" for stdin')

    option_group = parser.add_argument_group('Outputs', '')
    option_group.add_argument('-o', '--out', type=Path, help='Output path', default='.')
    option_group.add_argument('-f', '--format', choices=['png', 'svg', 'html', 'json'], metavar='',
                              default='png', help='Output format: png, svg or html')
    option_group.add_argument('-p', '--plotly', action='store_true',
                              help='Use plotly instead of matplotlib')

    other_group = parser.add_argument_group('Other options', '')
    other_group.add_argument('-v', '--version', action='version', version=__version__)
    other_group.add_argument('-h', '--help', action='help', help="Show this help message and exit")
    return parser


# Main -----------------------------------------------------------------------------------------------------------------
def main():
    parser = _get_parser()
    args = parser.parse_args()
    if not args.out.is_dir():
        args.out.mkdir(parents=True)
    db = load_database(args.db)
    plotter = KaptivePlotter(db)
    json_file, json_handle = None, None
    if args.json == '-':
        json_handle = stdin
    else:
        json_file = Path(args.json)
        assert json_file.exists() and json_file.suffix == '.json' and json_file.is_file()
        json_handle = open(args.json, 'rt')

    for line in json_handle:
        result = parse_result(line, db)
        out_path = args.out / f'{result.sample_name}_{result.best_match.name}_kaptive_plot.{args.format}'
        if args.plotly:
            fig = plotter.plotly(result)
            if args.format == 'html':
                fig.write_html(out_path)
            elif args.format == 'json':
                fig.write_json(out_path)
            else:
                fig.write_image(out_path)
        else:
            with plotter.plot(result) as (fig, ax):
                fig.savefig(out_path)

    if json_file is not None:
        json_handle.close()
