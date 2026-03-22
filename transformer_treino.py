# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

D_MODEL    = 128
D_FF       = 512
N_CAMADAS  = 2
VOCAB_SIZE = 119547
MAX_LEN    = 32
BATCH_SIZE = 16
N_EPOCHS   = 15
LR         = 1e-3
PAD_IDX    = 0

dataset = load_dataset("Helsinki-NLP/opus_books", "de-en", split="train")
subset  = dataset.select(range(1000))

tokenizador = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
TOKEN_START = tokenizador.cls_token_id
TOKEN_EOS   = tokenizador.sep_token_id


def tokenizar_par(exemplo: dict) -> dict:
    ids_origem = tokenizador(
        exemplo["translation"]["de"],
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )["input_ids"].squeeze(0)

    ids_destino_raw = tokenizador(
        exemplo["translation"]["en"],
        max_length=MAX_LEN - 2,
        truncation=True
    )["input_ids"]

    ids_destino = [TOKEN_START] + ids_destino_raw + [TOKEN_EOS]
    ids_destino += [PAD_IDX] * (MAX_LEN - len(ids_destino))
    ids_destino = ids_destino[:MAX_LEN]

    return {
        "ids_origem":  ids_origem.tolist(),
        "ids_destino": ids_destino
    }


dados_tokenizados = [tokenizar_par(ex) for ex in subset]

class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, mascara=None):
        dimensao_k = K.shape[-1]
        scores = Q @ K.transpose(-2, -1) / np.sqrt(dimensao_k)
        if mascara is not None:
            scores = scores + mascara
        pesos = torch.softmax(scores, dim=-1)
        return pesos @ V


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_query = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_key   = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_value = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.atencao = ScaledDotProductAttention()
        self.ffn     = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.ReLU(),
            nn.Linear(D_FF, D_MODEL)
        )
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)

    def forward(self, X):
        Q, K, V = self.W_query(X), self.W_key(X), self.W_value(X)
        X = self.norm1(X + self.atencao(Q, K, V))
        X = self.norm2(X + self.ffn(X))
        return X


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_query_self  = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_key_self    = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_value_self  = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_query_cross = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_key_cross   = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_value_cross = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.atencao = ScaledDotProductAttention()
        self.ffn     = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.ReLU(),
            nn.Linear(D_FF, D_MODEL)
        )
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.norm3 = nn.LayerNorm(D_MODEL)

    def forward(self, Y, Z):
        seq_len = Y.shape[1]
        mascara = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

        Q, K, V = self.W_query_self(Y), self.W_key_self(Y), self.W_value_self(Y)
        Y = self.norm1(Y + self.atencao(Q, K, V, mascara))

        Q, K, V = self.W_query_cross(Y), self.W_key_cross(Z), self.W_value_cross(Z)
        Y = self.norm2(Y + self.atencao(Q, K, V))

        Y = self.norm3(Y + self.ffn(Y))
        return Y


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_enc = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_IDX)
        self.embedding_dec = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_IDX)
        self.encoder_stack = nn.ModuleList([EncoderBlock() for _ in range(N_CAMADAS)])
        self.decoder_stack = nn.ModuleList([DecoderBlock() for _ in range(N_CAMADAS)])
        self.projecao_final = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, ids_enc, ids_dec):
        Z = self.embedding_enc(ids_enc)
        for bloco in self.encoder_stack:
            Z = bloco(Z)

        Y = self.embedding_dec(ids_dec)
        for bloco in self.decoder_stack:
            Y = bloco(Y, Z)

        return self.projecao_final(Y)