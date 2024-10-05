import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EmbeddingLayer, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        assert d_model % 2 == 0, 'd_model must be even'

        # Create a matrix of shape (max_len, d_model) to store positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even indices in the positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices in the positional encoding matrix
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension and register as a buffer (not a parameter but a constant in the model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input embedding
        x = x + self.pe[:, :x.size(1), :]
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.shape[-1]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_scores = F.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        output = torch.matmul(attention_scores, value)
        return output

    def forward(self, query, key, value, mask=None):  # Updated to accept query, key, value
        batch_size = query.size(0)

        # Linear projections for queries, keys, and values
        queries = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply scaled dot-product attention
        out = MultiHeadAttention.attention(queries, keys, values, mask, self.dropout)
        
        # Concatenate heads and apply final linear layer
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.fc_out(out)

    
class AddNorm(nn.Module):
    def __init__(self, d_model, dropout):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        return self.norm(x + self.dropout(residual))
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout) 
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderBlock, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.addNorm1 = AddNorm(d_model, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.addNorm2 = AddNorm(d_model, dropout)

    def forward(self, x, mask):
        attention_output = self.multihead_attention(x, x, x, mask)
        x = self.addNorm1(x, attention_output)
        feed_forward_output = self.feed_forward(x)
        x = self.addNorm2(x, feed_forward_output)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout, max_len, vocab_size):
        super(TransformerEncoder, self).__init__()
        self.embedding_layer = EmbeddingLayer(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.embedding_layer(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderBlock, self).__init__()
        self.masked_multihead_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.addNorm1 = AddNorm(d_model, dropout)
        self.multihead_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.addNorm2 = AddNorm(d_model, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.addNorm3 = AddNorm(d_model, dropout)

    def forward(self, x, encoder_output, trg_mask, src_mask):
        # Masked self-attention (decoder attends to itself)
        masked_attention_output = self.masked_multihead_attention(x, x, x, trg_mask)
        x = self.addNorm1(x, masked_attention_output)

        # Cross-attention (decoder attends to encoder output)
        attention_output = self.multihead_attention(x, encoder_output, encoder_output, src_mask) 
        x = self.addNorm2(x, attention_output)

        # Feed-forward network
        feed_forward_output = self.feed_forward(x)
        x = self.addNorm3(x, feed_forward_output)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout, max_len, vocab_size):
        super(TransformerDecoder, self).__init__()
        self.embedding_layer = EmbeddingLayer(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        x = self.embedding_layer(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, trg_mask, src_mask)
        return self.fc_out(self.norm(x))

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout, max_len, src_vocab_size)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_layers, dropout, max_len, tgt_vocab_size)
    
    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).unsqueeze(0).unsqueeze(0)
        return trg_mask.to(trg.device)
    
    def forward(self, src, trg, src_mask=None, trg_mask=None):
        trg_mask = self.make_trg_mask(trg)
        encoder_output = self.encoder(src, src_mask)
        output = self.decoder(trg, encoder_output, src_mask, trg_mask)
        return output
