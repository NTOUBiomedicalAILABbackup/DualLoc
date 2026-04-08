#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer
import os

class CustomDataset(Dataset):
    "Custom Data Set Categories"
    
    def __init__(self, df, tokenizer, max_len, target_list):
        self.tokenizer = tokenizer
        self.df = df
        self.title = list(df['combined'])
        self.targets = self.df[target_list].values
        self.max_len = max_len
        self.indices = df.index.values

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title)
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'targets': torch.FloatTensor(self.targets[index]),
            'title': title,
            'index': self.indices[index]
        }

class DataManager:
    """Data Management Category"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_name)
        
    def load_data(self, file_path, sample_size=None):
        
        df_data = pd.read_csv(file_path, sep=",")
        
        if sample_size:
            df_data = df_data.sample(n=sample_size)
            
        # Data Preprocessing
        df_data["combined"] = df_data["Sequence"]
        df_data.drop(columns=["Sequence"], axis=1, inplace=True)
        
        return df_data
    
    def split_data(self, df_data, test_size=0.30, random_state=77):
        """Data Segmentation"""
        df_train, df_test = train_test_split(
            df_data, 
            random_state=random_state, 
            test_size=test_size, 
            shuffle=True
        )
        return df_train, df_test
    
    def get_label_counts(self, df, label_columns):
        """Get Tag Statistics"""
        counts = []
        for label_col in label_columns:
            counts.append(df[label_col].value_counts().rename(label_col))
        return pd.concat(counts, axis=1)
    
    def create_dataset(self, df, target_list):
        """Creating a Data Set"""
        return CustomDataset(df, self.tokenizer, self.config.MAX_LEN, target_list)
    
    def create_dataloader(self, dataset, batch_size, shuffle=True):
        """Create a data loader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
    
    def get_target_list(self, df_data):
        """Get the list of target tags"""
        target_list = list(df_data.columns)
        return target_list[:-1]  
    
    def balance_dataset(self, df, label, random_state=77):
        """Balanced Data Set"""
        positive_samples = df[df[label] == 1]
        negative_samples = df[df[label] == 0]
        
        num_positive = len(positive_samples)
        num_negative = len(negative_samples)
        
        if num_positive <= num_negative:
            negative_samples = negative_samples.sample(n=num_positive, random_state=random_state)
        else:
            positive_samples = positive_samples.sample(n=num_negative, random_state=random_state)
        
        return pd.concat([positive_samples, negative_samples])