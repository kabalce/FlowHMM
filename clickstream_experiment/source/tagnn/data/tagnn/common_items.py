import pandas as pd
import click
from pathlib import Path


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    input_filepath = Path(input_filepath)
    output_filepath = Path(output_filepath)
    buys_pdf = pd.read_csv(
        input_filepath / 'yoochoose-buys.dat',
        header=None,
        names=['sessionID', 'timestamp', 'itemID', 'price', 'category'],
        dtype={'sessionID': int, 'itemID': int, 'price': int, 'category': str},
    )

    clicks_pdf = pd.read_csv(
        input_filepath / 'yoochoose-clicks.dat',
        header=None,
        names=['sessionID', 'timestamp', 'itemID', 'category'],
        dtype={'sessionID': int, 'itemID': int, 'category': str},
    )

    common_items = clicks_pdf['itemID'].value_counts() >= 500
    common_items = common_items.index[common_items]

    common_items_clicks_pdf = clicks_pdf[clicks_pdf['itemID'].isin(common_items)]
    common_items_buys_pdf = buys_pdf[buys_pdf['itemID'].isin(common_items)]

    common_items_clicks_pdf.to_csv(
        output_filepath / 'yoochoose-clicks.dat',
        header=False,
        index=False,
    )

    common_items_buys_pdf.to_csv(
        output_filepath / 'yoochoose-buys.dat',
        header=False,
        index=False,
    )


if __name__ == '__main__':
    main()
