import math
from turtle import forward
import torch
import torch.nn as nn
from torch.nn.modules.module import _forward_unimplemented
import config


class PostionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.arange(0, d_model, 2)
        div_term = torch.exp(div_term * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model) 或 (seq_len, d_model)
        # pe shape: (max_seq_len, d_model)
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]
class Translator(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, d_model, zh_pad_id=0, en_pad_id=0):
        super().__init__()
        self.zhembedding = nn.Embedding(zh_vocab_size, d_model, padding_idx=zh_pad_id)
        self.enembedding = nn.Embedding(en_vocab_size, d_model, padding_idx=en_pad_id)
        self.pos_encoder = PostionalEncoder(d_model, config.MAX_SEQ_LEN)
        self.transformer = nn.Transformer(d_model, config.NHEAD, config.NUM_ENCODER_LAYERS, config.NUM_DECODER_LAYERS, config.DIM_FEEDFORWARD, config.DROPOUT, config.ACTIVATION, batch_first=True)
        self.fc = nn.Linear(d_model, en_vocab_size)
        # 保存pad_id，forward内部生成padding mask需要
        self.zh_pad_id = zh_pad_id
        self.en_pad_id = en_pad_id

    def forward(self, src, tgt):
        src_padding_mask = (src == self.zh_pad_id)
        memory_key_padding_mask = (src == self.zh_pad_id)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=tgt.device, dtype=torch.float32)
        memory = self.encode(src, src_padding_mask)
        output = self.decode(tgt, tgt_mask, memory, memory_key_padding_mask)
        # output shape: (batch_size, seq_len, en_vocab_size)
        # memory shape: (batch_size, seq_len, d_model)
        return output

    def encode(self, src, src_padding_mask):
        src_embed = self.zhembedding(src)  # 中文输入使用中文embedding
        src_embed = self.pos_encoder(src_embed)
        # batch_first=True 时输入形状为 (batch_size, seq_len, d_model)
        return self.transformer.encoder(src=src_embed, src_key_padding_mask=src_padding_mask)
    
    def decode(self, tgt, tgt_mask, memory, memory_key_padding_mask):
        tgt_embed = self.enembedding(tgt)
        tgt_embed = self.pos_encoder(tgt_embed)
        # batch_first=True 时输入形状为 (batch_size, seq_len, d_model)
        decoded = self.transformer.decoder(tgt=tgt_embed, tgt_mask=tgt_mask, memory=memory, memory_key_padding_mask=memory_key_padding_mask)
        # decoded shape: (batch_size, seq_len, d_model)
        # fc_output shape: (batch_size, seq_len, en_vocab_size)
        return self.fc(decoded)
