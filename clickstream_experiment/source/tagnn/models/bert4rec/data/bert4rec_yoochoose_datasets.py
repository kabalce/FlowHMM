import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Union

import numpy as np
import pandas as pd

from src.utils.utils import flatten


def get_raw_data(data_path: Union[Path, str], dataset_version='yoochoose1_64'):
    if isinstance(data_path, str):
        data_path = Path(data_path)

    train_sequences = pickle.load(open(data_path / dataset_version / "all_train_seq.txt", 'rb'))
    test_data = pickle.load(open(data_path / dataset_version / "test.txt", 'rb'))

    return train_sequences, test_data


class YooChoosePreprocessor:
    def __init__(
            self,
            train_sequences: List[List[int]],
            test_data: Tuple[List[List[int]], List[int]],
            max_len: int,
            mask_prob: float,
            dupe_factor: int = 3,
            test_only: bool = False,
    ):
        self.raw_train_sequences = train_sequences
        self.raw_test_data = test_data

        self.max_len = max_len
        self.mask_prob = mask_prob
        self.dupe_factor = dupe_factor
        self.test_only = test_only

        self.train_user_count = 0
        self.test_user_count = 0
        self.item_count = 0
        self.mask_token = 0

    def get_processed_dataframes(
            self,
    ) -> Dict[str, Any]:
        df = self._preprocess()
        return df

    def _preprocess(self) -> Dict[str, Any]:
        """
        Assumes data cleaning of train_sequences and test_data has already been done.
        """
        train_sequences, test_data = self.raw_train_sequences, self.raw_test_data
        train_sequences, test_data, smap = self._densify_index(train_sequences, test_data)
        train, val = self._train_val_split(train_sequences)

        self.train_user_count: int = len(train)
        self.test_user_count: int = len(test_data[0])
        self.item_count = len(smap)
        self.mask_token = self.item_count + 1

        df = {"train": train, "test": test_data, "val": val, "umap": None, "smap": smap}
        final_df = self._mask_and_labels(self._generate_negative_samples(df))
        return final_df

    def _densify_index(
            self,
            train_sequences: List[List[int]],
            test_data: Tuple[List[List[int]], List[int]],
    ):
        all_items = flatten(train_sequences) + flatten(test_data)
        smap = {s: i + 1 for i, s in enumerate(set(all_items))}

        test_sequences, test_labels = test_data

        train_sequences = [[smap[item] for item in sequence] for sequence in train_sequences]
        test_sequences = [[smap[item] for item in sequence] for sequence in test_sequences]
        test_labels = [smap[item] for item in test_labels]

        return train_sequences, (test_sequences, test_labels), smap

    def _train_val_split(self, train_sequences):
        print("Splitting")
        train, val = defaultdict(list), defaultdict(list)
        for user, sequence in enumerate(train_sequences):
            if self.test_only:
                train[user] = sequence
            else:
                train[user], val[user] = sequence[:-1], sequence[-1:]
                if len(val[user]) == 0:
                    raise RuntimeError(
                        f"val set for user {user} is empty, consider increasing the data sample"
                    )

        return train, val

    def _generate_negative_samples(
            self,
            df: Dict[str, Any],
    ) -> Dict[str, Any]:
        # follow the paper, no negative samples in training set
        # 100 negative samples in test set, 2 for random to save time
        test_set_sample_size = 100

        # use popularity random sampling align with paper
        popularity = Counter()
        for user in range(self.train_user_count):
            popularity.update(df["train"][user])
            popularity.update(df["val"][user])
        (items_list, freq) = zip(*popularity.items())
        freq_sum = sum(freq)
        prob = [float(i) / freq_sum for i in freq]
        val_negative_samples = {}
        min_size = test_set_sample_size
        print("Sampling negative items")
        for user in range(self.train_user_count):
            seen = set(df["train"][user])
            seen.update(df["val"][user])
            samples = []
            while len(samples) < test_set_sample_size:
                sampled_ids = np.random.choice(
                    items_list, test_set_sample_size * 2, replace=False, p=prob
                )
                sampled_ids = [x for x in sampled_ids if x not in seen]
                samples.extend(sampled_ids[:])
            min_size = min_size if min_size < len(samples) else len(samples)
            val_negative_samples[user] = samples
        if min_size == 0:
            raise RuntimeError(
                "we sampled 0 negative samples for a user, please increase the data size"
            )
        val_negative_samples = {
            key: value[:min_size] for key, value in val_negative_samples.items()
        }
        df["val_negative_samples"] = val_negative_samples
        return df

    def _generate_masked_train_set(
            self,
            train: Dict[int, List[int]],
            dupe_factor: int,
            need_padding: bool = True,
    ) -> pd.DataFrame:
        df = []
        for user, seq in train.items():
            sliding_step = (int)(0.1 * self.max_len)
            beg_idx = list(
                range(
                    len(seq) - self.max_len,
                    0,
                    -sliding_step if sliding_step != 0 else -1,
                    )
            )
            beg_idx.append(0)
            seqs = [seq[i : i + self.max_len] for i in beg_idx[::-1]]
            for seq in seqs:
                for _ in range(dupe_factor):
                    tokens = []
                    labels = []
                    for s in seq:
                        prob = random.random()
                        if prob < self.mask_prob:
                            prob /= self.mask_prob

                            if prob < 0.8:
                                tokens.append(self.mask_token)
                            else:
                                tokens.append(random.randint(1, self.item_count))
                            labels.append(s)
                        else:
                            tokens.append(s)
                            labels.append(0)
                    if need_padding:
                        mask_len = self.max_len - len(tokens)
                        tokens = [0] * mask_len + tokens
                        labels = [0] * mask_len + labels
                    df.append([user, tokens, labels])
        return pd.DataFrame(df, columns=["user", "seqs", "labels"])

    def _generate_labeled_val_set(
            self,
            train: Dict[int, List[int]],
            eval: Dict[int, List[int]],
            negative_samples: Dict[int, List[int]],
            need_padding: bool = True,
    ) -> pd.DataFrame:
        df = []
        for user, seqs in train.items():
            answer = eval[user]
            negs = negative_samples[user]
            candidates = answer + negs
            labels = [1] * len(answer) + [0] * len(negs)
            tokens = seqs
            tokens = tokens + [self.mask_token]
            tokens = tokens[-self.max_len :]
            if need_padding:
                padding_len = self.max_len - len(tokens)
                tokens = [0] * padding_len + tokens
            df.append([user, tokens, candidates, labels])
        return pd.DataFrame(df, columns=["user", "seqs", "candidates", "labels"])

    def _generate_labeled_test_set(
            self,
            test_data: Tuple[List[List[int]], List[int]],
            all_items: Set[int],
            need_padding: bool = True,
    ) -> pd.DataFrame:
        df = []
        for idx, (sequence, target) in enumerate(zip(*test_data)):
            user = self.train_user_count + idx + 1
            negs = list(all_items - {target})
            candidates = [target] + negs
            labels = [1] + [0] * len(negs)
            tokens = sequence + [self.mask_token]
            tokens = tokens[-self.max_len:]
            if need_padding:
                padding_len = self.max_len - len(tokens)
                tokens = [0] * padding_len + tokens
            df.append([user, tokens, candidates, labels])
        return pd.DataFrame(df, columns=["user", "seqs", "candidates", "labels"])

    def _mask_and_labels(
            self,
            df: Dict[str, Any],
    ) -> Dict[str, Any]:
        masked_train_set = self._generate_masked_train_set(
            df["train"],
            dupe_factor=self.dupe_factor,
        )
        labeled_val_set = self._generate_labeled_val_set(
            df["train"],
            df["val"],
            df["val_negative_samples"],
        )

        all_items = set(df["smap"].values())

        labeled_test_set = self._generate_labeled_test_set(
            df["test"],
            all_items
        )

        masked_df = {
            "train": masked_train_set,
            "val": labeled_val_set,
            "test": labeled_test_set,
            "umap": None,
            "smap": df["smap"],
        }
        return masked_df
