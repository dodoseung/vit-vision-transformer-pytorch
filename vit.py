import torch.nn.functional as F
import torch.nn as nn
import torch
import math

# Vision Transformer
class ViT(nn.Module):
    def __init__(self, in_img=[3, 224, 224], out_dim=10, patch_size=16, layers=12, emb_size=768, d_ff=3072, num_heads=12, drop_rate=0.1):
        super(ViT, self).__init__()
        self.layers = layers
        
        # Modules
        self.embedding = PatchEmbedding(in_img, patch_size, emb_size)
        self.transformer = [Transformer(emb_size, num_heads, d_ff, drop_rate) for _ in range(layers)]
        self.transformer = nn.ModuleList(self.transformer)
        self.classification = ClassificationHead(emb_size, out_dim)
        
    def forward(self, x):
        x = self.embedding(x)
        for i in range (self.layers):
            x = self.transformer[i](x)
        x = self.classification(x)
        
        return x

# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, in_img, patch_size, emb_size):
        super(PatchEmbedding, self).__init__()
        self.in_dim = in_img[0]
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.patch = (in_img[1] // patch_size) * (in_img[2] // patch_size)
        
        # Projection
        self.conv = nn.Conv2d(self.in_dim, self.emb_size, kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
        
        # Cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_size))

        # Positional token
        self.pos_token = nn.Parameter(torch.randn((self.patch + 1, self.emb_size)))

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Projection
        x = self.conv(x)
        x = x.view(-1, self.patch, self.emb_size)
        
        # Cls token
        rep_cls_token = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat([rep_cls_token, x], dim=1)
        
        # Positional token
        x = x + self.pos_token
        
        return x

# Transformer
class Transformer(nn.Module):
    def __init__(self, emb_size, num_heads, d_ff, drop_rate):
        super(Transformer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(emb_size=emb_size, num_heads=num_heads)
        self.position_wise_feed_forward = PositionWiseFeedForward(emb_size=emb_size, d_ff=d_ff)
        
        self.multi_head_layer_norm = nn.LayerNorm(emb_size)
        self.feed_forward_layer_norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(drop_rate)
        
    def forward(self, x):
        residual = x
        x = self.multi_head_layer_norm(x)
        x = self.multi_head_attention(x, x, x)
        x = self.dropout(x) + residual
        
        residual = x
        x = self.feed_forward_layer_norm(x)
        x = self.position_wise_feed_forward(x)
        x = self.dropout(x) + residual
        
        return x

# Multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.d_k = self.emb_size // self.num_heads
        
        # Define w_q, w_k, w_v, w_o
        self.weight_q = nn.Linear(self.emb_size, self.emb_size)
        self.weight_k = nn.Linear(self.emb_size, self.emb_size)
        self.weight_v = nn.Linear(self.emb_size, self.emb_size)
        self.weight_o = nn.Linear(self.emb_size, self.emb_size)
    
    def forward(self, query, key, value, mask=None):
        # Batch size
        batch_size, patch, _ = query.shape

        # (batch, patch, emb_size) -> (batch, patch, emb_size)
        query = self.weight_q(query)
        key = self.weight_k(key)
        value = self.weight_v(value)

        # (batch, patch, emb_size) -> (batch, patch, h, d_k)
        query = query.view(batch_size, patch, self.num_heads, self.d_k)
        key = key.view(batch_size, patch, self.num_heads, self.d_k)
        value = value.view(batch_size, patch, self.num_heads, self.d_k)
        
        # (batch, patch, h, d_k) -> (batch, h, patch, d_k)
        query = torch.transpose(query, 1, 2)
        key = torch.transpose(key, 1, 2)
        value = torch.transpose(value, 1, 2)
        
        # Get the scaled attention
        # (batch, h, patch, d_k) -> (batch, patch, h, d_k)
        scaled_attention = self.scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = torch.transpose(scaled_attention, 1, 2).contiguous()

        # Concat the splitted attentions
        # (batch, patch, h, d_k) -> (batch, patch, emb_size)
        concat_attention = scaled_attention.view(batch_size, -1, self.emb_size)
        
        # Get the multi head attention
        # (batch, patch, emb_size) -> (batch, patch, emb_size)
        multihead_attention = self.weight_o(concat_attention)
        
        return multihead_attention
    
    # Query, key, and value size: (batch, num_heads, seq_len, d_k)
    # Mask size(optional): (batch, 1, seq_len, seq_len)   
    def scaled_dot_product_attention(self, query, key, value, mask):
        # Get the q matmul k_t
        # (batch, h, patch, d_k) dot (batch, h, d_k, key_len)
        # -> (batch, h, patch, key_len)
        attention_score = torch.matmul(query, torch.transpose(key, -2, -1))

        # Get the attention wights
        attention_score = attention_score.masked_fill(mask==0, -1e10) if mask is not None else attention_score
        attention_weights = F.softmax(attention_score, dim=-1, dtype=torch.float)
        
        # Get the attention score
        emb_size = query.size(1) * query.size(-1)
        attention_score = attention_score / math.sqrt(emb_size)

        # Get the attention value
        # (batch, h, patch, key_len) -> (batch, h, patch, d_k)
        attention_value = torch.matmul(attention_weights, value)
        
        return attention_value
    
# Position wise feed forward
class PositionWiseFeedForward(nn.Module):
    def __init__(self, emb_size, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(emb_size, d_ff)
        self.fc2 = nn.Linear(d_ff, emb_size)

    def forward(self, x):
        out = F.gelu(self.fc1(x))
        out = self.fc2(out)
        
        return out
 
# Classification Head
class ClassificationHead(nn.Module):
    def __init__(self, emb_size, out_dim):
        super(ClassificationHead, self).__init__()
        self.emb_size = emb_size
        self.out_dim = out_dim
        
        self.layer_norm = nn.LayerNorm(self.emb_size)
        self.fc = nn.Linear(self.emb_size, self.out_dim)
        
    def forward(self, x):
        x = x.mean(axis=1)
        x = self.layer_norm(x)
        x = self.fc(x)
        
        return x