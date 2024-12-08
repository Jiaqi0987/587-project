import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from dataclasses import dataclass

# -----------------------------------------------------------------------------

class MultiHeadCausalAttention(nn.Module):
    """Implements multi-head attention with causal masking."""
    def __init__(self, config):
        super(MultiHeadCausalAttention, self).__init__()
        if config.embedding_dim % config.num_heads != 0:
            raise ValueError("Embedding size must be divisible by the number of heads.")
        # Linear transformations for query, key, and value
        self.qkv_projection = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        # Linear transformation for output
        self.final_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.final_projection.SCALE_FACTOR = 1  # Custom scaling for initialization

        # Number of attention heads and dimensions per head
        self.num_heads = config.num_heads
        self.dim_per_head = config.embedding_dim // config.num_heads

    def forward(self, inputs):
        batch_size, seq_length, embed_dim = inputs.size()
        # Compute query, key, and value tensors
        qkv = self.qkv_projection(inputs)
        q, k, v = torch.chunk(qkv, 3, dim=2)

        # Reshape tensors for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.dim_per_head).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.dim_per_head).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.dim_per_head).transpose(1, 2)

        # Perform scaled dot-product attention with causal masking
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape and project back to original embedding size
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)
        return self.final_projection(attn_output)

class PositionwiseFeedForward(nn.Module):
    """Applies a feed-forward network to the input tensor."""
    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(config.embedding_dim, 4 * config.embedding_dim)
        self.activation_fn = nn.GELU(approximate='tanh')
        self.linear2 = nn.Linear(4 * config.embedding_dim, config.embedding_dim)
        self.linear2.SCALE_FACTOR = 1  # Custom scaling for initialization

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        return self.linear2(x)

class TransformerLayer(nn.Module):
    """Defines a single layer of the transformer."""
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        # Normalization layers
        self.pre_attn_norm = nn.LayerNorm(config.embedding_dim)
        self.pre_ffn_norm = nn.LayerNorm(config.embedding_dim)

        # Attention and feed-forward components
        self.causal_attention = MultiHeadCausalAttention(config)
        self.feedforward = PositionwiseFeedForward(config)

    def forward(self, x):
        # Residual connection with attention
        x = x + self.causal_attention(self.pre_attn_norm(x))
        # Residual connection with feed-forward
        x = x + self.feedforward(self.pre_ffn_norm(x))
        return x

@dataclass
class TransformerConfig:
    """Configuration for the Transformer model."""
    seq_len: int = 1024  # Maximum sequence length
    vocab_size: int = 50257  # Vocabulary size
    num_layers: int = 12  # Number of transformer layers
    num_heads: int = 12  # Number of attention heads
    embedding_dim: int = 768  # Dimensionality of embeddings

    def get_block_size(self):
        """Returns the block size for the transformer."""
        return self.seq_len

class GPT(nn.Module):
    """Full GPT model with transformer architecture."""
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config

        # Embedding layers
        self.token_embed = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_embed = nn.Embedding(config.seq_len, config.embedding_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.embedding_dim)

        # Output layer
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # Weight sharing with token embedding

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Custom initialization for model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = 0.02
                if hasattr(module, 'SCALE_FACTOR'):
                    std *= (2 * self.config.num_layers) ** -0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        """Forward pass of the GPT model."""
        B, T = input_ids.size()
        if T > self.config.get_block_size():
            raise ValueError(f"Input sequence length {T} exceeds block size {self.config.get_block_size()}.")

        # Embedding lookup
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.positional_embed(positions)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Apply final normalization
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def load_pretrained(cls, model_type):
        """Load pretrained GPT-2 weights from HuggingFace."""
        from transformers import GPT2LMHeadModel

        # Supported GPT-2 models
        model_mapping = {
            "gpt2": {"num_layers": 12, "num_heads": 12, "embedding_dim": 768},
            "gpt2-medium": {"num_layers": 24, "num_heads": 16, "embedding_dim": 1024},
            "gpt2-large": {"num_layers": 36, "num_heads": 20, "embedding_dim": 1280},
            "gpt2-xl": {"num_layers": 48, "num_heads": 25, "embedding_dim": 1600},
        }

        # Check if model type is valid
        if model_type not in model_mapping:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Set configuration
        config_params = model_mapping[model_type]
        config_params.update({"vocab_size": 50257, "seq_len": 1024})
        config = TransformerConfig(**config_params)

        # Initialize our model
        model = cls(config)

        # Load HuggingFace GPT-2 model
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_state_dict = hf_model.state_dict()

        # Transfer weights from HuggingFace model
        own_state_dict = model.state_dict()
        for name, param in hf_state_dict.items():
            if name in own_state_dict and param.size() == own_state_dict[name].size():
                own_state_dict[name].copy_(param)

        return model

    def configure_optimizers(self, weight_decay, lr, device_type):
        """Configure optimizer with weight decay for model parameters."""
        # Separate parameters into groups for weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if param.ndimension() >= 2:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use fused AdamW if supported
        try:
            use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        except Exception:
            use_fused = False

        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optimizer_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer

