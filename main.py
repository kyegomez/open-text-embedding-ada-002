#!/usr/bin/env python3
"""
Optimized Transformer-based Embedding Model with Multi-Query Attention and GEGLU FFN in PyTorch

This module implements a transformer-based embedding model inspired by OpenAI's text-embedding-ada-002 architecture.
Key improvements include:
  - **Multi-Query Attention:** Instead of projecting keys and values per head, a single key/value projection is shared
    across all heads while queries remain multi-headed. This reduces memory footprint and speeds up attention computation.
  - **GEGLU FeedForward Network:** The feed-forward block uses the GEGLU variant for improved performance over a standard MLP.
  
The model follows a GPT-style decoder-only transformer design where the hidden state corresponding to the final [EOS] token
is used as the fixed-dimensional embedding.

Usage:
    1. Configure the model using the TransformerConfig dataclass.
    2. Instantiate the TextEmbeddingModel.
    3. Preprocess text to token IDs (using an appropriate tokenizer).
    4. Run a forward pass to obtain embeddings.
    
Author: Your Name
Date: 2025-03-09
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


@dataclass
class TransformerConfig:
    """Configuration parameters for the transformer model."""
    vocab_size: int = 100000
    max_seq_len: int = 8192
    embd_dim: int = 1536
    num_layers: int = 24
    num_heads: int = 16
    dropout: float = 0.1
    mlp_ratio: float = 4.0  # Factor for hidden dimension in FFN
    layer_norm_eps: float = 1e-5


class MultiQuerySelfAttention(nn.Module):
    """
    Multi-Query Self-Attention module.

    In Multi-Query Attention, the query is projected into multiple heads as usual,
    but keys and values are projected once (i.e. shared across heads). This reduces memory usage
    and computational overhead.
    """

    def __init__(self, embd_dim: int, num_heads: int, dropout: float) -> None:
        """
        Args:
            embd_dim (int): Dimensionality of the embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super().__init__()
        assert embd_dim % num_heads == 0, "embd_dim must be divisible by num_heads"
        self.embd_dim = embd_dim
        self.num_heads = num_heads
        self.head_dim = embd_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query is projected into multiple heads
        self.q_proj = nn.Linear(embd_dim, embd_dim)
        # Keys and values are shared among heads
        self.k_proj = nn.Linear(embd_dim, self.head_dim)
        self.v_proj = nn.Linear(embd_dim, self.head_dim)
        self.out_proj = nn.Linear(embd_dim, embd_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for multi-query self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embd_dim).
            mask (Optional[torch.Tensor]): Attention mask broadcastable to (batch_size, num_heads, seq_length, seq_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, embd_dim).
        """
        batch_size, seq_length, _ = x.size()

        # Compute queries and reshape for multi-head attention.
        q = self.q_proj(x)  # (batch_size, seq_length, embd_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)

        # Keys and values: shared projections (no head dimension) and then unsqueeze to broadcast.
        k = self.k_proj(x)  # (batch_size, seq_length, head_dim)
        v = self.v_proj(x)  # (batch_size, seq_length, head_dim)
        # Expand keys and values along the head dimension.
        k = k.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (batch_size, num_heads, seq_length, head_dim)
        v = v.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (batch_size, num_heads, seq_length, head_dim)

        # Compute scaled dot-product attention.
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_length, seq_length)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)  # (batch_size, num_heads, seq_length, head_dim)
        # Concatenate heads.
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, self.embd_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)
        return attn_output


class GEGLUFeedForward(nn.Module):
    """
    FeedForward network with GEGLU activation.

    GEGLU (Gated GELU) is an alternative FFN design where the input is split into two parts;
    one is passed through GELU and then multiplied elementwise with the other.
    This variant has been shown to yield improvements over the standard FFN.
    """

    def __init__(self, embd_dim: int, mlp_ratio: float, dropout: float) -> None:
        """
        Args:
            embd_dim (int): Dimensionality of the input embedding.
            mlp_ratio (float): Multiplier for determining the hidden layer size.
            dropout (float): Dropout probability.
        """
        super().__init__()
        hidden_dim = int(embd_dim * mlp_ratio)
        # Instead of a single linear layer, we project to 2 * hidden_dim and then apply GEGLU.
        self.proj = nn.Linear(embd_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, embd_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GEGLU feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embd_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        x_proj = self.proj(x)  # (batch_size, seq_length, 2 * hidden_dim)
        # Split projections into two halves.
        x_proj, gate = x_proj.chunk(2, dim=-1)  # both: (batch_size, seq_length, hidden_dim)
        # Apply GELU activation on the first half and gate it with the second half.
        x_out = F.gelu(x_proj) * gate
        x_out = self.dropout(x_out)
        x_out = self.out_proj(x_out)
        x_out = self.dropout(x_out)
        return x_out


class TransformerBlock(nn.Module):
    """
    A single transformer block using multi-query attention and a GEGLU-based FFN.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize a transformer block.

        Args:
            config (TransformerConfig): Model configuration.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embd_dim, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.embd_dim, eps=config.layer_norm_eps)
        self.attn = MultiQuerySelfAttention(config.embd_dim, config.num_heads, config.dropout)
        self.ffn = GEGLUFeedForward(config.embd_dim, config.mlp_ratio, config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for a transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embd_dim).
            mask (Optional[torch.Tensor]): Optional attention mask.

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        # Multi-query attention with residual connection.
        x = x + self.attn(self.ln1(x), mask)
        # GEGLU feed-forward network with residual connection.
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    """
    Transformer model that encodes token sequences into contextual embeddings.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the transformer model.

        Args:
            config (TransformerConfig): Model configuration.
        """
        super().__init__()
        self.config = config

        # Token and positional embeddings.
        self.token_embedding = nn.Embedding(config.vocab_size, config.embd_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embd_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_final = nn.LayerNorm(config.embd_dim, eps=config.layer_norm_eps)

    def forward(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the transformer model.

        Args:
            token_ids (torch.Tensor): Tensor of token IDs (batch_size, seq_length).
            mask (Optional[torch.Tensor]): Optional attention mask.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, embd_dim).
        """
        batch_size, seq_length = token_ids.size()
        logger.debug("Transformer input: batch_size={}, seq_length={}", batch_size, seq_length)

        positions = torch.arange(seq_length, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_length)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for idx, layer in enumerate(self.layers):
            x = layer(x, mask)
            logger.debug("Completed transformer layer {}", idx + 1)

        x = self.ln_final(x)
        return x


class TextEmbeddingModel(nn.Module):
    """
    Text Embedding Model that converts tokenized text input into fixed-dimensional embeddings.

    The model uses a GPT-style decoder-only transformer. The embedding is extracted from the hidden state
    corresponding to the final [EOS] token in each sequence.
    """

    def __init__(self, config: TransformerConfig, eos_token_id: int = 2) -> None:
        """
        Initialize the TextEmbeddingModel.

        Args:
            config (TransformerConfig): Transformer configuration.
            eos_token_id (int): Token ID representing the end-of-sequence ([EOS]) marker.
        """
        super().__init__()
        self.config = config
        self.eos_token_id = eos_token_id
        self.transformer = TransformerModel(config)
        logger.info("TextEmbeddingModel initialized with embd_dim={} and {} transformer layers.",
                    config.embd_dim, config.num_layers)

    def forward(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to obtain text embeddings.

        Args:
            token_ids (torch.Tensor): Tensor of token IDs (batch_size, seq_length).
            mask (Optional[torch.Tensor]): Optional attention mask.

        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embd_dim).
        """
        transformer_out = self.transformer(token_ids, mask)  # (batch_size, seq_length, embd_dim)
        logger.debug("Transformer output shape: {}", transformer_out.shape)

        # Efficient extraction of the last occurrence of the EOS token for each sequence.
        eos_mask = (token_ids == self.eos_token_id)
        # Create a tensor of positions [0, 1, ..., seq_length-1] and apply the mask.
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0).expand_as(token_ids)
        eos_positions = torch.where(eos_mask, positions, torch.full_like(positions, -1))
        final_positions, _ = torch.max(eos_positions, dim=1)
        # If no EOS token is found, use the last token.
        final_positions = torch.where(final_positions == -1, torch.full_like(final_positions, token_ids.size(1) - 1), final_positions)
        batch_indices = torch.arange(token_ids.size(0), device=token_ids.device)
        embeddings = transformer_out[batch_indices, final_positions]  # (batch_size, embd_dim)
        logger.info("Generated embeddings for batch of size {}", embeddings.size(0))
        return embeddings


# model = TextEmbeddingModel(TransformerConfig())

# input = torch.randint(0, 10000, (1, 10))
# output = model(input)
# print(output.shape)
