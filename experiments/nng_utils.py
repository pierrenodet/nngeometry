import torch.nn.functional as F
from torch import nn

from nngeometry.layercollection import (
    Conv1dLayer,
    Conv2dLayer,
    EmbeddingLayer,
    LayerCollection,
    LinearLayer,
)


def last_layer(layer_collection: LayerCollection) -> LayerCollection:
    name = next(reversed(layer_collection.layers))
    last_layer_collection = LayerCollection()
    last_layer_collection.add_layer(name, layer_collection.layers[name])
    return last_layer_collection


def supported_kfac_layers(layer_collection: LayerCollection) -> LayerCollection:
    filtered_layer_collection = LayerCollection()
    for name, layer in layer_collection.layers.items():
        if isinstance(
            layer,
            (LinearLayer, Conv2dLayer, Conv1dLayer, EmbeddingLayer),
        ):
            filtered_layer_collection.add_layer(name, layer)
    return filtered_layer_collection


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=1):
        super().__init__()
        if num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by a positive num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.batch_first = True
        self.in_proj_bias = None
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, q, k, v, **kwargs):
        if (
            kwargs.get("attn_mask") is not None
            or kwargs.get("key_padding_mask") is not None
            or kwargs.get("is_causal", False)
        ):
            raise NotImplementedError("Only unmasked attention is supported")
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        q, k, v = [
            tensor.reshape(tensor.size(0), -1, self.num_heads, self.head_dim).transpose(
                1, 2
            )
            for tensor in (q, k, v)
        ]
        attn_weights = F.softmax(q @ k.transpose(-2, -1) / (self.head_dim**0.5), dim=-1)
        output = (
            (attn_weights @ v).transpose(1, 2).reshape(q.size(0), -1, self.hidden_size)
        )
        weights = attn_weights.mean(dim=1) if kwargs.get("need_weights", True) else None
        return self.out_proj(output), weights


def replace_torch_mha_with_nng_mha(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.MultiheadAttention):
        return (
            MultiHeadAttention(model.embed_dim, model.num_heads)
            .to(model.out_proj.weight)
            .train(model.training)
        )

    for name, child in list(model.named_children()):
        model.add_module(name, replace_torch_mha_with_nng_mha(child))
    return model
