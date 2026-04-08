#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from transformers import T5EncoderModel

class T5Class(nn.Module):
    """T5 Multi-Label Classification Model"""
    
    def __init__(self, config):
        super(T5Class, self).__init__()
        self.config = config
        
        # Load the pre-trained model
        self.t5_model = T5EncoderModel.from_pretrained(config.model_name)
        self.t5_fine_model = T5EncoderModel.from_pretrained(config.model_name)
        
        # Freeze the parameters of the base model
        for param in self.t5_model.parameters():
            param.requires_grad = False
            
        self.dropout = nn.Dropout(config.dropout_rate)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attn_mask):
        encoder_outputs = self.t5_fine_model.encoder(
            input_ids=input_ids,
            attention_mask=attn_mask
        )
        
        # Take the last hidden state of the encoder
        hidden_state = encoder_outputs.last_hidden_state
        
        # The average of the hidden states of all tokens is used as the representation of the sentence.
        sentence_representation = hidden_state.mean(dim=1)
        output_dropout = self.dropout(sentence_representation)
        output = self.linear(output_dropout)
        
        return output

class ModelManager:
    """Model Management Category"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        
    def create_model(self):

        self.model = T5Class(self.config)
        self.model.to(self.config.device)
        return self.model
    
    def create_optimizer(self, model):

        from transformers import AdamW
        self.optimizer = AdamW(model.parameters(), lr=self.config.LEARNING_RATE)
        return self.optimizer
    
    def loss_fn(self, outputs, targets):

        return nn.BCEWithLogitsLoss()(outputs, targets)
    
    def save_model(self, model, path):

        torch.save(model.state_dict(), path)
        
    def load_model(self, model, path):

        model.load_state_dict(torch.load(path))
        return model
    
    def save_checkpoint(self, model, optimizer, epoch, best_accuracy, history, path):

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
            'history': history,
        }, path)
    
    def load_checkpoint(self, model, optimizer, path):

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return (
            checkpoint['epoch'],
            checkpoint['best_accuracy'],
            checkpoint['history']
        )