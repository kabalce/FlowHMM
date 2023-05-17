import argparse

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Amazon data.")
    parser.add_argument(
        'train',
        type=str,
        help='pandas dataframe containing training interactions in parquet format'
    )
    parser.add_argument(
        'test',
        type=str,
        help='pandas dataframe containing testing interactions in parquet format'
    )
    parser.add_argument(
        'path',
        type=str,
        help='path in which to store the results',
    )
    return parser.parse_args()


def remap_pdfs(train_pdf, test_pdf, save_path):
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    train_pdf['reviewerID'] = user_encoder.fit_transform(train_pdf['reviewerID'])
    train_pdf['asin'] = item_encoder.fit_transform(train_pdf['asin'])

    test_pdf['reviewerID'] = user_encoder.transform(test_pdf['reviewerID'])
    test_pdf['asin'] = item_encoder.transform(test_pdf['asin'])

    user_list_pdf = pd.DataFrame(zip(user_encoder.classes_, range(len(user_encoder.classes_))))
    user_list_pdf.columns = ['org_id', 'remap_id']
    user_list_pdf.to_csv(save_path / 'user_list.txt', sep=' ', index=False, header=True)

    item_list_pdf = pd.DataFrame(zip(item_encoder.classes_, range(len(item_encoder.classes_))))
    item_list_pdf.columns = ['org_id', 'remap_id']
    item_list_pdf.to_csv(save_path / 'item_list.txt', sep=' ', index=False, header=True)

    return train_pdf, test_pdf


def main():
    args = parse_args()
    tqdm.pandas()

    save_path = Path(args.path)
    train_pdf = pd.read_parquet(args.train)
    test_pdf = pd.read_parquet(args.test)

    # Drop users and items not present in the training set
    test_pdf = test_pdf[test_pdf['reviewerID'].isin(train_pdf['reviewerID']) & test_pdf['asin'].isin(train_pdf['asin'])]

    train_pdf, test_pdf = remap_pdfs(train_pdf, test_pdf, save_path)

    grouped_train_pdf = train_pdf.groupby('reviewerID').progress_apply(
        lambda pdf: f"{pdf.iloc[0]['reviewerID']} " + ' '.join(map(str, pdf['asin']))
    )

    grouped_test_pdf = test_pdf.groupby('reviewerID').progress_apply(
        lambda pdf: f"{pdf.iloc[0]['reviewerID']} " + ' '.join(map(str, pdf['asin']))
    )

    grouped_train_pdf.to_csv(save_path / 'train.txt', header=False, index=False)
    grouped_test_pdf.to_csv(save_path / 'test.txt', header=False, index=False)


if __name__ == '__main__':
    main()
