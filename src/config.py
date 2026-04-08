#!/usr/bin/env python
# coding: utf-8

import os
import torch

class Config:
    """Configuration Management Category"""
    
    def __init__(self):
        # Model hyperparameters
        self.MAX_LEN = 512
        self.TRAIN_BATCH_SIZE = 1
        self.VALID_BATCH_SIZE = 1
        self.TEST_BATCH_SIZE = 1
        self.EPOCHS = 10
        self.LEARNING_RATE = 2e-04
        self.THRESHOLD = 0.5
        
        # Equipment Configuration
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        
        # Data path
        self.data_dir = './data'
        
        # Tag list
        self.label_columns = ['CH', ' GPI', ' MT', ' NES', ' NLS', ' PTS', ' SP', ' TH', ' TM']
        
        # Model Configuration
        self.model_name = 'Rostlab/prot_t5_xl_uniref50'
        self.dropout_rate = 0.3
        self.hidden_size = 1024
        self.num_labels = 9
        
        # Training configuration
        self.high_confidence_threshold = 0.9
        self.stability_threshold = 3
        self.num_runs = 5
        
        # File path
        self.checkpoint_path = os.path.join(self.data_dir, "checkpoint_base_ProtT5_Dynamic_Label_Predict.pth")
        self.model_path = os.path.join(self.data_dir, "MLTC_model_state_base_ProtT5_Dynamic_Label_Predict.bin")
        self.log_path = os.path.join(self.data_dir, "training_log.txt")