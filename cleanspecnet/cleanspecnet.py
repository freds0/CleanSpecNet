import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch import Tensor
#from torch.nn import TransformerEncoderLayer

class AttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        
        # Position-wise Feedforward Network
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu

    def forward(self, x, attn_mask=None):
        # Multi-head Self-Attention
        residual = x
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + self.dropout1(x)
        x = self.norm1(x)
        
        # Position-wise Feedforward Network
        residual = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout2(x)
        x = self.norm2(x)
        
        return x


class CleanSpecNet(nn.Module):
    def __init__(self, input_channels=513, num_conv_layers=5, kernel_size=4, stride=1, conv_hidden_dim=64, hidden_dim=512, num_attention_layers=5, num_heads=8, dropout=0.1):
        super(CleanSpecNet, self).__init__()

        self.input_layer = nn.Conv1d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        # Convolutional Layers
        conv_input_channels = input_channels
        self.conv_layers = nn.ModuleList()
        for _ in range(num_conv_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(conv_input_channels, conv_hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Conv1d(conv_hidden_dim, conv_hidden_dim * 2, kernel_size=kernel_size, stride=stride, padding=1),
                nn.GLU(dim=1)
            ))
            conv_input_channels = conv_hidden_dim

        self.tsfm_projection = nn.Linear(conv_hidden_dim, hidden_dim)

        # Self-Attention Layers
        self.attention_layers = nn.ModuleList()
        for _ in range(num_attention_layers):
            self.attention_layers.append(AttentionBlock(hidden_dim, num_heads, dropout))

        # Final conv Layer to project back to input dimensions
        self.output_layer = nn.Conv1d(hidden_dim, input_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x: (batch_size, freq_bins, time_steps)
        x = self.input_layer(x)
        # Convolutional Layers
        for conv in self.conv_layers:
            x = conv(x)  # (batch_size, channels, time_steps)
        # Prepare for Attention Layers
        x = x.transpose(1, 2)  # (batch_size, time_steps, channels)
        x = self.tsfm_projection(x)
        # Attention Layers
        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        for attn in self.attention_layers:
            x = attn(x, causal_mask)
        x = x.transpose(1, 2)
        # Final projection
        x = self.output_layer(x)  # (batch_size, time_steps, input_channels)
        return x


# Exemplo de uso:
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import json
    with open("configs/config_cleanspecnet.json") as f:
        data = f.read()
    config = json.loads(data)

    network_config = config["network_config"] 
    
    model = CleanSpecNet(**network_config).to(device)
    
    # Simulação de entrada de espectrograma
    input_data = torch.randn(2, 513, 1024).to(device)  # (batch_size, freq_bins, time)
    
    output = model(input_data)
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
