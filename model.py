import torch
import torch.nn as nn
import config

class MyModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM)
        self.rnn = nn.RNN(config.EMBEDDING_DIM,config.HIDDEN_SIZE,batch_first=True)
        self.linear = nn.Linear(config.HIDDEN_SIZE, vocab_size)
    
    def forward(self, x):
        # input: (batch_size, seq_len), output: (batch_size, seq_len, embedding_dim)
        x = self.embedding(x)
        # input: (batch_size, seq_len, embedding_dim), output: (batch_size, seq_len, hidden_size)
        x, _ = self.rnn(x)
        # input: (batch_size, seq_len, hidden_size), output: (batch_size, vocab_size)
        x = self.linear(x[:, -1, :])
        return x