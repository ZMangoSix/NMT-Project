import math
import torch
from torch import nn

'''
Q (Query) : (batch_size, no. of queries, embedding dimension)
K (Key)   : (batch_size, no. of key-value pairs, embedding dimension)
V (Value) : (batch_size, no. of key-value pairs, embedding dimension)

Dot product attention: softmax( Q*K^T/sqrt(d) )
additive attention: softmax( w_v*tanh(W_q*q+W_k*k) )
'''


### Masked softmax operation
def masked_softmax(X, valid_lens):
    '''
    Perform softmax operation by masking elements on the last axis.
    X : (number of batch, no. of queries, no. of key-value pairs)
    valid_len: 1D or 2D tensor
    '''
    def _sequence_mask(X, valid_len, value=0):
        # X : (number of batch*no. of queries, no. of key-value pairs)
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)

        # On the last axis, replace masked elements with a very large negative value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)



### Dot product attention
class DotProductAttention(nn.Module):
    '''
    Scaled dot product attention
    '''

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)



### Additive attention
class AdditiveAttention(nn.Module):
    '''
    additive attention.
    '''
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        queries, keys = self.W_q(queries) , self.W_k(keys)
        # After dimesion expansion
        # queries: (batch_size, no. of queries, 1, num_hiddens)
        # keys: (batch_size, 1, no. of key-value pairs, num_hiddens)
        # features: (batch_size, no. of queries, no. of key-value pairs, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of self.w_v
        # self.w_v(features): (batch_size, no. of queries, no. of key-value pairs, 1)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)



### Multi-head attention
class MultiHeadAttention(nn.Module):
    '''
    Multi-head attention
    '''
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        # Shape of queries, keys, or values after hidden layer:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        queries = self.transpose_qkv(self.W_q(queries))
        keys    = self.transpose_qkv(self.W_k(keys))
        values  = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # (batch_size,) or (batch_size, no. of queries) to (batch_size*num_heads,) or (batch_size*num_heads, no. of queries)
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # Shape of output: (batch_size * num_heads, no. of queries, num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Sahpe or output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

    def transpose_qkv(self, X):
        '''
        Transposition for parallel computation of multiple attention heads
        '''
        # Input :(batch_size, no. of queries or key-value pairs, num_hiddens)
        # (batch_size, no. of queries or key-value pairs, num_heads, num_hiddens/num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # (batch_size, num_heads, no. of queries or key-value pairs, num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)
        # (batch_size*num_heads, no. of queries or key-value pairs, num_hiddens/num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])
    
    def transpose_output(self, X):
        '''
        Reverse the operation of transpose_qkv
        '''
        X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)



if __name__ == '__main__':
    # Use 1D valid_lens
    print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
    # Use 2D valid_lens
    print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

    # Batch Matrix Multiplication
    Q = torch.ones((2, 3, 4))
    K = torch.ones((2, 6, 4))
    torch.bmm(Q, K.transpose(1, 2))

    ### Test dot product attention
    queries = torch.normal(0, 1, (2, 1, 2))
    keys = torch.normal(0, 1, (2, 10, 2))
    values = torch.normal(0, 1, (2, 10, 4))
    valid_lens = torch.tensor([2, 6])

    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    attention(queries, keys, values, valid_lens)
    # d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))

    ### Test additive attention
    queries = torch.normal(0, 1, (2, 1, 20))

    attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
    attention.eval()
    attention(queries, keys, values, valid_lens)
    # d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))

    ### Test multi-head attention
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
    batch_size, num_queries, num_kvpairs = 2, 4, 6
    valid_lens = torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    attention(X, Y, Y, valid_lens)
