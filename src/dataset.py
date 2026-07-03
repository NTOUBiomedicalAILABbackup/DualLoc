"""
dataset.py
==========
Data loading, preprocessing, and CustomDataset definition.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, DATA_FILE, N_SAMPLES, RANDOM_STATE, TEST_SIZE,
    LABEL_COLUMNS, MAX_LEN, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE,
)


class CustomDataset(Dataset):
    """
    Wraps protein sequences ("combined") and multi-label targets (target_list)
    into a PyTorch Dataset.
    """

    def __init__(self, df, tokenizer, max_len, target_list):
        self.tokenizer = tokenizer
        self.df = df
        self.title = list(df["combined"])
        self.targets = self.df[target_list].values
        self.max_len = max_len
        self.indices = df.index.values

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        # The ProtT5 tokenizer requires residues to be whitespace-separated
        title = " ".join(title)

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "targets": torch.FloatTensor(self.targets[index]),
            "title": title,
            "index": self.indices[index],
        }


def load_raw_data(data_dir=DATA_DIR, data_file=DATA_FILE, n_samples=N_SAMPLES,
                   random_state=RANDOM_STATE):
    """
    Load the raw TSV data, subsample it, and prepare the sequence column ("combined").
    """
    df_data = pd.read_csv(os.path.join(data_dir, data_file), sep="\t")
    df_data = df_data.sample(n=n_samples, random_state=random_state)

    # Merge the sequence column and drop unused columns
    df_data["combined"] = df_data["Sequence"]
    df_data.drop(columns=["Sequence"], inplace=True)
    df_data.drop(columns=["Membrane"], inplace=True)

    return df_data


def split_train_test(df_data, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split the dataset into train and test sets.
    """
    df_train, df_test = train_test_split(
        df_data, random_state=random_state, test_size=test_size, shuffle=True
    )
    print(f"Train: {df_train.shape}, Test: {df_test.shape}")
    return df_train, df_test


def get_label_counts(df, label_columns=LABEL_COLUMNS):
    """
    Count the positive/negative sample distribution for each label column.
    """
    counts = []
    for label_col in label_columns:
        counts.append(df[label_col].value_counts().rename(label_col))
    return pd.concat(counts, axis=1)


def build_dataloaders(df_train, df_test, tokenizer, target_list,
                       max_len=MAX_LEN,
                       train_batch_size=TRAIN_BATCH_SIZE,
                       test_batch_size=TEST_BATCH_SIZE):
    """
    Build CustomDataset and DataLoader instances for the train/test DataFrames.
    """
    train_dataset = CustomDataset(df_train, tokenizer, max_len, target_list)
    test_dataset = CustomDataset(df_test, tokenizer, max_len, target_list)

    train_data_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0
    )

    return train_dataset, test_dataset, train_data_loader, test_data_loader


def get_target_list(df_data):
    """
    Get the list of label columns for multi-label classification
    (excludes the last "combined" column).
    """
    target_list = list(df_data.columns)
    return target_list[:-1]
