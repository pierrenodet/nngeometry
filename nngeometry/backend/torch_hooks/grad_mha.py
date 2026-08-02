import torch
import torch.nn.functional as F


def mha_backward_to_qkv(mod, x, gy):
    bs, seq, embed_dim = x.size()
    num_heads = mod.num_heads
    head_dim = embed_dim // num_heads

    def split_heads(t):
        return t.view(bs, seq, num_heads, head_dim).transpose(1, 2)

    def merge_heads(t):
        return t.transpose(1, 2).reshape(bs, seq, embed_dim)

    with torch.no_grad():
        qkv = F.linear(x, mod.in_proj_weight, mod.in_proj_bias)
        q, k, v = (split_heads(t) for t in qkv.chunk(3, dim=-1))

        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
        attn = F.softmax(scores, dim=-1)

        gy_o = torch.matmul(gy, mod.out_proj.weight)
        g_O = split_heads(gy_o)

        g_attn = torch.matmul(g_O, v.transpose(-2, -1))
        g_v = torch.matmul(attn.transpose(-2, -1), g_O)

        g_scores = attn * (g_attn - (g_attn * attn).sum(dim=-1, keepdim=True))

        g_q = torch.matmul(g_scores, k) / (head_dim**0.5)
        g_k = torch.matmul(g_scores.transpose(-2, -1), q) / (head_dim**0.5)

        gy_qkv = torch.cat(
            [merge_heads(g_q), merge_heads(g_k), merge_heads(g_v)], dim=-1
        )

    return gy_qkv
