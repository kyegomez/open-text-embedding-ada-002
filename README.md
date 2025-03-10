
# Transformer-based Embedding Model with Multi-Query Attention and GEGLU FFN

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


[![GitHub stars](https://img.shields.io/github/stars/The-Swarm-Corporation/Legal-Swarm-Template?style=social)](https://github.com/The-Swarm-Corporation/Legal-Swarm-Template)
[![Swarms Framework](https://img.shields.io/badge/Built%20with-Swarms-blue)](https://github.com/kyegomez/swarms)

This repository presents a production-grade implementation of a transformer-based text embedding model inspired by OpenAI's *text-embedding-ada-002*. Our implementation incorporates advanced design choices including multi-query self-attention and a GEGLU-activated feed-forward network (FFN) to achieve state-of-the-art efficiency and performance. The code is written in PyTorch and is designed with extensive type annotations, inline documentation, and detailed logging via loguru. This work also serves as an academic exploration and open source replication of the Ada model, originally detailed by Kye Gomez in his open source replication efforts.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
- [Code Structure](#code-structure)
- [Performance and Optimizations](#performance-and-optimizations)
- [Citation](#citation)
- [License](#license)

## Introduction

The aim of this project is to replicate and enhance the capabilities of OpenAI's text embedding model, *text-embedding-ada-002*. The implementation presented here employs a GPT-style decoder-only transformer, where the final hidden state corresponding to the end-of-sequence ([EOS]) token is used as the text embedding. The architecture has been refined with two major innovations:

1. **Multi-Query Self-Attention**: By sharing key and value projections across all heads, our multi-query attention reduces memory consumption and improves computational efficiency without compromising on performance.
2. **GEGLU Feed-Forward Network**: This state-of-the-art FFN variant applies a gated GELU mechanism, yielding superior results compared to traditional two-layer MLPs.

Our work builds on the open source replication initiatives led by Kye Gomez and integrates rigorous academic research and best practices from the latest transformer research.

## Features

- **Transformer-based Architecture**: A decoder-only transformer that efficiently processes tokenized text and extracts embeddings.
- **Multi-Query Self-Attention**: Optimizes the attention mechanism by sharing keys and values across heads.
- **GEGLU FFN**: Incorporates a gated activation function (GEGLU) for enhanced non-linear modeling.
- **Extensive Documentation and Type Hints**: Designed for readability, maintainability, and ease of integration in research pipelines.
- **Advanced Logging**: Uses loguru for detailed logging and traceability during model execution.
- **Highly Configurable**: Model hyperparameters are managed via a dedicated configuration dataclass, allowing easy experimentation.

## Model Architecture

The model follows a GPT-style transformer design, with the following key components:

- **Embedding Layers**: Token and positional embeddings with a configurable vocabulary size and maximum sequence length.
- **Multi-Query Attention**: Queries are projected into multiple heads while keys and values are shared across heads.
- **Transformer Blocks**: Stacked layers that combine multi-query attention with GEGLU-enhanced feed-forward networks, each preceded by layer normalization.
- **Embedding Extraction**: The final [EOS] token's hidden state is extracted to serve as the fixed-dimensional text embedding.

This design ensures that the model is both computationally efficient and highly performant for a range of text embedding tasks.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (>=1.8)
- loguru
- Other dependencies as specified in `requirements.txt` (if available)

### Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/your_username/transformer-embedding-model.git
cd transformer-embedding-model
pip install -r requirements.txt
```

### Usage

Import the model, set up the configuration, and run a forward pass on tokenized inputs:

```python
from torch import tensor
from model import TransformerConfig, TextEmbeddingModel

# Define model configuration
config = TransformerConfig(
    vocab_size=100000,
    max_seq_len=8192,
    embd_dim=1536,
    num_layers=6,  # Adjust for production use
    num_heads=16,
    dropout=0.1,
    mlp_ratio=4.0,
    layer_norm_eps=1e-5,
)


input = torch.randint(0, 10000, (1, 10))

with torch.no_grad():
    output = model(input)
    
print(output.shape)

```

## Code Structure

- **`model.py`**: Contains the implementation of the transformer architecture, including multi-query self-attention, GEGLU FFN, and the overall embedding model.
- **`config.py`**: Defines the `TransformerConfig` dataclass used to manage hyperparameters.
- **`utils.py`**: (Optional) Utility functions for tokenization, logging, and other support tasks.
- **`tests/`**: Unit tests and integration tests for model verification.

## Performance and Optimizations

The model leverages several performance optimizations:

- **Efficient Tensor Operations**: The extraction of the final [EOS] token embedding is fully vectorized.
- **Reduced Memory Footprint**: Multi-query attention minimizes redundant computation by sharing key/value projections.
- **Scalability**: Configurable model depth and width allow for adaptation to a variety of hardware and performance requirements.
- **Logging and Debugging**: Extensive logging via loguru ensures that model execution can be traced and debugged effectively.

## Citation

If you use this implementation or derive insights from this repository in your research, please cite the following work:

```bibtex
@misc{kye2023ada,
  title = {Open Source Replication of OpenAI's text-embedding-ada-002},
  author = {Kye Gomez},
  year = {2023},
  note = {Accessed: 2025-03-09},
  url = {https://github.com/kyegomez/open-text-embedding-ada-002}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
