"""
model.py
========
T5Class: a ProtT5-encoder-based multi-label classifier with attention pooling.
"""

import scipy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel

from config import PRETRAINED_MODEL_NAME


def smooth_tensor_1d(input_tensor, smooth_sigma):
    """
    Smooth an input tensor along the second dimension using a Gaussian filter.

    Arguments:
        input_tensor: an (A, B) tensor to smooth along dimension B
        smooth_sigma: standard deviation of the Gaussian filter; the smoothing
            window is 1 + (2 * sigma). sigma=0 means no smoothing.

    Returns:
        A tensor with the same shape as the input, smoothed along dimension B.
    """
    if smooth_sigma == 0:
        sigma, truncate = 1, 0
    else:
        sigma, truncate = smooth_sigma, 1

    base = np.zeros(1 + (2 * sigma))
    base[sigma] = 1  # Center of the window is 1, everywhere else is 0
    kernel = scipy.ndimage.gaussian_filter(
        base, sigma=sigma, truncate=truncate
    ).astype(np.float32)
    kernel = torch.tensor(kernel, device=input_tensor.device)

    input_tensor = torch.unsqueeze(input_tensor, dim=1)
    kernel = torch.unsqueeze(torch.unsqueeze(kernel, dim=0), dim=1).float()
    padded_input = F.pad(input_tensor, (sigma, sigma), "replicate")
    smoothed = torch.nn.functional.conv1d(padded_input, kernel)
    return torch.squeeze(smoothed, dim=1)


class AttentionHead(nn.Module):
    """
    A (possibly multi-head) attention pooling layer that compresses a sequence
    representation into a fixed-length vector.
    """

    def __init__(self, hidden_dim, n_heads):
        super(AttentionHead, self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.preattn_ln = nn.LayerNorm(hidden_dim // n_heads)
        self.Q = nn.Linear(hidden_dim // n_heads, n_heads, bias=False)
        torch.nn.init.normal_(self.Q.weight, mean=0.0, std=1 / (hidden_dim // n_heads))

    def forward(self, x, np_mask, lengths):
        # x: (batch, seq_len, embed)
        n_heads = self.n_heads
        hidden_dim = self.hidden_dim
        x = x.view(x.size(0), x.size(1), n_heads, hidden_dim // n_heads)
        x = self.preattn_ln(x)
        mul = (x * self.Q.weight.view(1, 1, n_heads, hidden_dim // n_heads)).sum(-1)

        # Smooth and pad each sample individually
        mul_score_list = []
        for i in range(mul.size(0)):
            mul_score_list.append(
                F.pad(
                    smooth_tensor_1d(mul[i, : lengths[i], 0].unsqueeze(0), 2).unsqueeze(0),
                    (0, mul.size(1) - lengths[i]),
                    "constant",
                ).squeeze(0)
            )

        mul = torch.cat(mul_score_list, dim=0).unsqueeze(-1)
        mul = mul.masked_fill(~np_mask.unsqueeze(-1), float("-inf"))
        attns = F.softmax(mul, dim=1)  # (b, l, nh)
        x = (x * attns.unsqueeze(-1)).sum(1)
        x = x.view(x.size(0), -1)
        return x, attns.squeeze(2)


class T5Class(torch.nn.Module):
    """
    Multi-label classification model built on a ProtT5 encoder backbone,
    followed by attention pooling and a linear classification head.
    """

    def __init__(self, pretrained_model_name=PRETRAINED_MODEL_NAME,
                 hidden_dim=1024, n_heads=1, num_labels=10, dropout=0.3):
        super(T5Class, self).__init__()
        # Keeps the original pretrained weights (not used in forward, kept for reference/comparison)
        self.t5_model = T5EncoderModel.from_pretrained(pretrained_model_name)
        # The model actually used for training; its weights are re-initialized below
        self.t5_fine_model = T5EncoderModel.from_pretrained(pretrained_model_name)
        self.t5_fine_model.apply(self._init_weights)

        self.attn_head = AttentionHead(hidden_dim, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.clf_head = nn.Linear(hidden_dim, num_labels)

        # Note: the original notebook's forward() referenced self.initial_ln and
        # self.lin without ever defining them in __init__. They are added here
        # so the model is actually runnable.
        self.initial_ln = nn.LayerNorm(hidden_dim)
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def _init_weights(self, module):
        """Re-initialize weights for a given module."""
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attn_mask):
        encoder_outputs = self.t5_fine_model.encoder(
            input_ids=input_ids, attention_mask=attn_mask
        )
        hidden_state = encoder_outputs.last_hidden_state
        lengths = attn_mask.sum(dim=1).cpu().tolist()

        x = self.initial_ln(hidden_state)
        x = self.lin(x)
        np_mask = attn_mask.bool()
        x_pool, x_attns = self.attn_head(x, np_mask, lengths)
        x_pool = self.dropout(x_pool)
        x_pred = self.clf_head(x_pool)

        return x_pred


def build_model(device):
    """
    Build a T5Class model and move it to the specified device.
    """
    model = T5Class()
    model.to(device)
    return model
