# Transformer Treinamento Fim-a-Fim — LAB 05

## Descrição

Laboratório final da Unidade I. Conectamos o modelo Transformer construído nos
labs anteriores a um dataset real de tradução (DE→EN) e implementamos o loop de
treinamento completo com backpropagation.

O objetivo principal não é criar um tradutor perfeito, mas sim demonstrar que a
arquitetura consegue aprender, fazendo a função de perda cair ao longo das épocas.

## Como Rodar

**Pré-requisitos:** Python 3.x, PyTorch, Hugging Face `datasets` e `transformers`.

1. Crie e ative um ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instale as dependências:
```bash
pip install torch datasets transformers numpy
```

3. Execute o treinamento:
```bash
python3 transformer_treino.py
```

O script baixa o dataset, tokeniza as frases, treina o modelo por 15 épocas e
executa o teste de overfitting ao final.

## O que foi feito

**Dataset:** Usamos o dataset `Helsinki-NLP/opus_books` (pares de-en) do Hugging
Face com um subconjunto de 1.000 frases para manter o treinamento viável em CPU.

**Tokenização:** Utilizamos o tokenizador `bert-base-multilingual-cased` da
biblioteca Transformers. As frases de destino recebem tokens especiais `<START>`
e `<EOS>`, e padding até `MAX_LEN=32`.

**Training Loop:** Implementamos o loop completo: forward pass pelo Encoder e
Decoder, cálculo da `CrossEntropyLoss` ignorando padding, backward pass e
atualização dos pesos com Adam. A loss é impressa a cada época e apresenta
queda consistente.

**Overfitting Test:** Após o treinamento, pegamos uma frase do conjunto de treino
e usamos o loop auto-regressivo para gerar a tradução, verificando se o modelo
conseguiu memorizar o padrão.

## Hiperparâmetros

| Parâmetro     | Valor   |
|---------------|---------|
| `d_model`     | 128     |
| `d_ff`        | 512     |
| `n_layers`    | 2       |
| `max_len`     | 32      |
| `batch_size`  | 16      |
| `epochs`      | 15      |
| `lr` (Adam)   | 0.001   |

## Convergência do Loss
```
Época 01/15 — Loss: 7.73
Época 05/15 — Loss: 4.15
Época 10/15 — Loss: 2.26
Época 15/15 — Loss: 0.84
```

## Overfitting Test
```
Entrada (DE) : Source: http://www.zeno.org - Contumax GmbH & Co. KG
Referência   : Source: Project Gutenberg
Modelo gerou : Source : Project Gutenberg  ✓
```

## Referência

Vaswani, A. et al. **Attention Is All You Need**, 2017.
https://arxiv.org/abs/1706.03762

## Uso de IA

Foi utilizada IA generativa (Claude - Anthropic) como ferramenta auxiliar para:
- Estruturação e documentação do projeto
- Auxílio na sintaxe e depuração do código

A arquitetura do Transformer e o fluxo de Forward/Backward foram construídos
com base nos laboratórios anteriores, conforme exigido no roteiro.
Revisado por Victor Cerqueira.

---
Tag de entrega: `v1.0`