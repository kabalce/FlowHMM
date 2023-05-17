import argparse
from pathlib import Path

import pandas as pd
from sklearn.manifold import TSNE


def parse_args():
    parser = argparse.ArgumentParser(description="Run TSNE on embeddings.")
    parser.add_argument(
        'filename',
        type=str,
        help='base name of embeddings file, path should contain {filename}_full_embeddings.parquet file',
    )
    parser.add_argument(
        '--path',
        dest='path',
        type=str,
        default='/pio/scratch/1/i308362/recommender_system/data/interim/word2vec',
        help='path in which full embeddings file is, and in which tsne embeddings will be saved',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='print messages to stdin',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.path)

    clothes_full_pdf = pd.read_parquet(
        data_root / f'{args.filename}_full_embeddings.parquet'
    )
    if args.verbose:
        print("Finished reading the data, starting TSNE")

    if args.verbose:
        tsne = TSNE(init='pca', learning_rate='auto', verbose=10)
    else:
        tsne = TSNE(init='pca', learning_rate='auto')

    clothes_tsne_pdf = pd.DataFrame(tsne.fit_transform(clothes_full_pdf.values))
    clothes_tsne_pdf.index = clothes_full_pdf.index
    clothes_tsne_pdf.columns = ['x', 'y']

    if args.verbose:
        print(f"Finished TSNE, saving to {str(data_root / f'{args.filename}_tsne_embeddings.parquet')}")

    clothes_tsne_pdf.to_parquet(data_root / f'{args.filename}_tsne_embeddings.parquet')


if __name__ == '__main__':
    main()
