import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0, "MultiHeadAttentionLayer: number of hidden dimensions should be divisible by the number of heads."
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(name='scale', tensor=torch.FloatTensor([self.head_dim]))

    def forward(self, query, key, value, mask=None):
        # query = (batch_size, query_len, hid_dim)
        # key = (batch_size, key_len, hid_dim)
        # value = (batch_size, value_len, hid_dim)
        
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q = (batch_size, query_len, hid_dim)
        # K = (batch_size, key_len, hid_dim)
        # V = (batch_size, value_len, hid_dim)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q = (batch_size, n_heads, query_len, head_dim)
        # K = (batch_size, n_heads, key_len, head_dim)
        # V = (batch_size, n_heads, value_len, head_dim)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = (batch_size, n_heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        # attention = (batch_size, n_heads, query_len, key_len)

        x = torch.matmul(self.dropout(attention), V)
        # x = (batch_size, n_heads, query_len, head_dim)

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = (batch_size, query_len, n_heads, head_dim)

        x = x.view(batch_size, -1, self.hid_dim)
        # x = (batch_size, query_len, hid_dim)

        x = self.fc_o(x)
        # x = (batch_size, query_len, hid_dim)

        return x, attention

