# -*- coding: utf-8 -*-
import logging
import os
import re
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from data.split import core_train_test_split, frac_train_test_split, const_train_test_split, time_train_test_split

from data.split_utils import save_subsets



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

@click.option('--file_format', type=click.STRING, default='parquet')
def main(input_filepath: Path, output_filepath: Path, file_format: str):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    input_filepath, output_filepath = Path(input_filepath), Path(output_filepath)

    if not output_filepath.exists():
        os.mkdir(output_filepath)

    if input_filepath.is_dir():
        # Concatenate data from every file in the directory that
        # ends with clean.parquet and doesn't start with meta
        pattern = re.compile("^(?!meta).*clean\.parquet$")
        data = []
        for file in input_filepath.iterdir():
            if pattern.match(file.name):
                pdf = pd.read_parquet(file)
                data.append(pdf)
        data_pdf = pd.concat(data)
    else:
        data_pdf = pd.read_parquet(input_filepath)


    data_pdf = data_pdf.sort_values('unixReviewTime', na_position='first')
    data_pdf = data_pdf.drop_duplicates(subset=['reviewerID', 'asin'], keep='last')

    for name, func in [
        ('5-core', core_train_test_split),
        ('const', const_train_test_split),
        ('frac', frac_train_test_split),
        ('time', time_train_test_split),
    ]:
        logger.info(f'creating {name} dataset')
        save_subsets(output_filepath, name, func(data_pdf), file_format=file_format)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
