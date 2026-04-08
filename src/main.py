#!/usr/bin/env python
# coding: utf-8

"""
Multi-label text classification main program; 
using the T5 model for protein localization prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# Import custom modules
from config import Config
from dataset import DataManager
from model import ModelManager
from trainer import Trainer



class MultiLabelClassifier:
    """Main Category of Multi-Label Classifier"""
    
    def __init__(self):
        self.config = Config()
        self.data_manager = DataManager(self.config)
        self.model_manager = ModelManager(self.config)

        
        
    def load_and_prepare_data(self):
        """Loading and preparing data"""
        print("Loading and preparing data...")
        
        # Load data
        csv_path = os.path.join(self.config.data_dir, "signal_sorting.csv")
        df_data = self.data_manager.load_data(csv_path, sample_size=1868)
        
        # Data splitting
        df_train, df_test = self.data_manager.split_data(df_data)
        
        # Get target tags
        target_list = self.data_manager.get_target_list(df_data)
        
        print(f"training set: {df_train.shape}, test set: {df_test.shape}")
        
        # Display tag statistics
        train_counts = self.data_manager.get_label_counts(df_train, self.config.label_columns)
        test_counts = self.data_manager.get_label_counts(df_test, self.config.label_columns)
        
        print("training set tag statistics:")
        print(train_counts)
        print("\ntest set tag statistics:")
        print(test_counts)
        
        return df_train, df_test, target_list
    
    def create_data_loaders(self, df_train, df_test, target_list):

        print("create_data_loaders...")
        
        # Creating a Data Set
        train_dataset = self.data_manager.create_dataset(df_train, target_list)
        test_dataset = self.data_manager.create_dataset(df_test, target_list)
        
        # Create a data loader
        train_loader = self.data_manager.create_dataloader(
            train_dataset, self.config.TRAIN_BATCH_SIZE, shuffle=True
        )
        test_loader = self.data_manager.create_dataloader(
            test_dataset, self.config.TEST_BATCH_SIZE, shuffle=False
        )
        
        return train_loader, test_loader
    
    def setup_model_and_optimizer(self):
        """Setting up the model and optimizer"""
        print("Setting up the model and optimizer...")
        
        # Create a model
        model = self.model_manager.create_model()
        
        # Create optimizer
        optimizer = self.model_manager.create_optimizer(model)
        
        return model, optimizer
    
    def train_model(self, model, optimizer, train_loader, test_loader, target_list):
        print("Start training...")
        
        trainer = Trainer(self.config, self.model_manager)
        trained_model = trainer.train(model, optimizer, train_loader, test_loader, target_list)
        
        return trained_model, trainer.history
    
    def _calculate_average_metrics(self, all_results):
        metrics_list = [result['metrics'] for result in all_results]
        
        avg_metrics = {}
        for key in ['f1_micro', 'f1_macro', 'jaccard_micro', 'jaccard_macro', 'hamming_loss', 'subset_accuracy']:
            values = [metrics[key] for metrics in metrics_list]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
        
        return avg_metrics


    def predict_raw_text(self, model, raw_text, target_list):
        """Predicting from the original text"""
        print("Predicting from the original text...")
        
        # Preprocessed text
        raw_text = " ".join(raw_text)
        
        # coding
        encoded_text = self.data_manager.tokenizer.encode_plus(
            raw_text,
            max_length=self.config.MAX_LEN,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # predict
        model.eval()
        with torch.no_grad():
            input_ids = encoded_text['input_ids'].to(self.config.device)
            attention_mask = encoded_text['attention_mask'].to(self.config.device)
            
            output = model(input_ids, attention_mask)
            output = torch.sigmoid(output).detach().cpu()
            output = output.flatten().round().numpy()
        
        # Display results
        print(f"text: {raw_text[:100]}...")
        print("Predicted Labels:")
        for idx, p in enumerate(output):
            if p == 1:
                print(f"  - {target_list[idx]}")
        
        return output
    
    def run_complete_pipeline(self):
        """Execute the complete process"""
        print("=" * 50)
        print("Start executing the multi-label text classification process")
        print("=" * 50)
        
        try:
            # 1. Loading and preparing data
            df_train, df_test, target_list = self.load_and_prepare_data()
            
            # 2. Create a data loader
            train_loader, test_loader = self.create_data_loaders(df_train, df_test, target_list)
            
            # 3. Setting up the model and optimizer
            model, optimizer = self.setup_model_and_optimizer()
            
            # 4. Training the model
            trained_model, history = self.train_model(model, optimizer, train_loader, test_loader, target_list)
            
            # 5. Test Original Text Prediction
            sample_text = "MYWSNQITRRLGERVQGFMSGISPQQMGEPEGSWSGKNPGTMGASRLYTLVLVLQPQRVLLGMKKRGFGAGRWNGFGGKVQEGETIEDGARRELQEESGLTVDALHKVGQIVFEFVGEPELMDVHVFCTDSIQGTPVESDEMRPCWFQLDQIPFKDMWPDDSYWFPLLLQKKKFHGYFKFQGQDTILDYTLREVDTV"
            self.predict_raw_text(trained_model, sample_text, target_list)
            
            print("=" * 50)
            print("Process completed！")
            print("=" * 50)
            
            return {
                'model': trained_model,
                'target_list': target_list
            }
            
        except Exception as e:
            print(f"An error occurred during execution.: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    # Create a classifier instance
    classifier = MultiLabelClassifier()
    
    # Execute the complete process
    results = classifier.run_complete_pipeline()
    
    if results:
        print("All results have been saved to:", classifier.config.data_dir)
    else:
        print("Execution failed, please check the error message.")

if __name__ == "__main__":
    main()