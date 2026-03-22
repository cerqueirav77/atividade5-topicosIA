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