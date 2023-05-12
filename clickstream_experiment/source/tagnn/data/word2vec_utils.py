from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def create_core(train_pdf: pd.DataFrame, save_path: Path):
    """
    Saves a core file used by word2vec in save_path

    Args:
        train_pdf: Dataframe from which to create core
        save_path: Path to a file in which to save
    """
    grouped_pdf = train_pdf.groupby('reviewerID').progress_apply(
        lambda pdf: list(map(str, pdf['asin'])) if len(pdf) > 1 else None
    ).dropna()

    with open(save_path, 'w') as out_file:
        for items in tqdm(grouped_pdf.values):
            for _ in range(int(np.sqrt(len(items)))):
                out_file.write(' '.join(np.random.permutation(items)) + '\n')
